from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Node

from mdc_llm_deploy.errors import QuantizationConfigError
from mdc_llm_deploy.graph.fx import ownership as ownership_module
from mdc_llm_deploy.graph.lifecycle import (
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    TensorAbi,
    set_metadata,
)
from mdc_llm_deploy.quantization import materialization as materialization_module
from mdc_llm_deploy.quantization.algorithms.math import quantize
from mdc_llm_deploy.quantization.calibration import (
    _CalibrationInterpreter,
    _CalibrationOwnershipSnapshot,
    collect_calibration_samples,
)
from mdc_llm_deploy.quantization.config import ActivationSpec, WeightSpec
from mdc_llm_deploy.quantization.materialization import (
    MaterializationContext,
    _require_same_device,
    materialize_alias_group,
    materialize_target,
)
from mdc_llm_deploy.quantization.planning import (
    CalibrationPlan,
    TargetPlan,
    plan_calibration,
)

_WEIGHT = WeightSpec(bits=8, granularity="per_channel")
_ACTIVATION = ActivationSpec(bits=8, granularity="per_tensor", mode="static")


def _target(
    fqn: str,
    *,
    target_type: str = "linear",
    algorithm: str = "minmax",
    parameter_name: str | None = None,
    weight: WeightSpec | None = None,
    activation: ActivationSpec | None = None,
) -> TargetPlan:
    return TargetPlan(
        fqn=fqn,
        target_type=target_type,
        algorithm=algorithm,
        modifier_index=0,
        parameter_name=parameter_name,
        weight=weight,
        activation=activation,
    )


def _capture(*fqns: str) -> CalibrationPlan:
    return CalibrationPlan(frozenset(fqns))


def _graph() -> GraphModule:
    root = nn.Module()
    root.add_module("linear", nn.Linear(2, 2, bias=False))
    graph = Graph()
    value = graph.placeholder("x")
    weight = graph.get_attr("linear.weight")
    value = graph.call_function(
        torch.ops.aten.linear.default,
        (value, weight, None),
    )
    graph.output(value)
    module = GraphModule(root, graph)
    set_metadata(
        module,
        GraphMetadata(
            schema_version=1,
            stage=GraphStage.FLOAT_PREFILL,
            model_kind="dense",
            input_abi=(TensorAbi("x", "float32", (1, 2)),),
            output_abi=(TensorAbi("output", "float32", (1, 2)),),
            sequence_length=2,
        ),
    )
    return module


def _graph_with_bias_input() -> GraphModule:
    root = nn.Module()
    root.add_module("linear", nn.Linear(2, 2, bias=False))
    graph = Graph()
    value = graph.placeholder("x")
    bias = graph.placeholder("bias")
    weight = graph.get_attr("linear.weight")
    value = graph.call_function(
        torch.ops.aten.linear.default,
        (value, weight, bias),
    )
    graph.output(value)
    module = GraphModule(root, graph)
    set_metadata(
        module,
        GraphMetadata(
            schema_version=1,
            stage=GraphStage.FLOAT_PREFILL,
            model_kind="dense",
            input_abi=(
                TensorAbi("x", "float32", (1, 2)),
                TensorAbi("bias", "float32", (2,)),
            ),
            output_abi=(TensorAbi("output", "float32", (1, 2)),),
            sequence_length=2,
        ),
    )
    return module


def _two_linear_graph() -> GraphModule:
    root = nn.Module()
    root.add_module("first", nn.Linear(2, 2, bias=False))
    root.add_module("second", nn.Linear(2, 2, bias=False))
    graph = Graph()
    value = graph.placeholder("x")
    first_weight = graph.get_attr("first.weight")
    value = graph.call_function(
        torch.ops.aten.linear.default,
        (value, first_weight, None),
    )
    second_weight = graph.get_attr("second.weight")
    value = graph.call_function(
        torch.ops.aten.linear.default,
        (value, second_weight, None),
    )
    graph.output(value)
    module = GraphModule(root, graph)
    set_metadata(
        module,
        GraphMetadata(
            schema_version=1,
            stage=GraphStage.FLOAT_PREFILL,
            model_kind="dense",
            input_abi=(TensorAbi("x", "float32", (1, 2)),),
            output_abi=(TensorAbi("output", "float32", (1, 2)),),
            sequence_length=2,
        ),
    )
    return module


def _fanout_linear_graph(*, tied: bool = False) -> GraphModule:
    root = nn.Module()
    root.add_module("first", nn.Linear(2, 2, bias=False))
    root.add_module("second", nn.Linear(2, 2, bias=False))
    if tied:
        root.second.weight = root.first.weight
    graph = Graph()
    value = graph.placeholder("x")
    first_weight = graph.get_attr("first.weight")
    first = graph.call_function(
        torch.ops.aten.linear.default,
        (value, first_weight, None),
    )
    second_weight = graph.get_attr("second.weight")
    second = graph.call_function(
        torch.ops.aten.linear.default,
        (value, second_weight, None),
    )
    graph.output(graph.call_function(torch.ops.aten.add.Tensor, (first, second)))
    module = GraphModule(root, graph)
    set_metadata(
        module,
        GraphMetadata(
            schema_version=1,
            stage=GraphStage.FLOAT_PREFILL,
            model_kind="dense",
            input_abi=(TensorAbi("x", "float32", (1, 2)),),
            output_abi=(TensorAbi("output", "float32", (1, 2)),),
            sequence_length=2,
        ),
    )
    return module


def _attention_graph() -> GraphModule:
    root = nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "ordinary"):
        root.add_module(name, nn.Linear(2, 2, bias=False))
    graph = Graph()
    value = graph.placeholder("x")
    for name in ("q_proj", "k_proj", "v_proj", "ordinary"):
        weight = graph.get_attr(f"{name}.weight")
        value = graph.call_function(
            torch.ops.aten.linear.default,
            (value, weight, None),
        )
        value.name = name
        if name != "ordinary":
            value.meta["nn_module_stack"] = {
                "owner": ("self_attn", object())
            }
    value = graph.call_function(torch.ops.aten.mul.Tensor, (value, 0.5))
    value.name = "score"
    value.meta["nn_module_stack"] = {"owner": ("self_attn", object())}
    value = graph.call_function(torch.ops.aten.neg.default, (value,))
    value.name = "non_score"
    value.meta["nn_module_stack"] = {"owner": ("self_attn", object())}
    graph.output(value)
    module = GraphModule(root, graph)
    set_metadata(
        module,
        GraphMetadata(
            schema_version=1,
            stage=GraphStage.FLOAT_PREFILL,
            model_kind="dense",
            input_abi=(TensorAbi("x", "float32", (1, 1, 2, 2)),),
            output_abi=(TensorAbi("output", "float32", (1, 1, 2, 2)),),
            boundaries=(FusionBoundary("attention", "self_attn"),),
            sequence_length=2,
        ),
    )
    return module


def _moe_ownership_graph(owner: object | None) -> tuple[GraphModule, Node]:
    graph = Graph()
    values = [graph.placeholder(name) for name in ("x", "ids", "routing", "weights")]
    node = graph.call_function(
        torch.ops.mdc_llm_deploy.moe_expert.default,
        tuple(values),
    )
    if owner is not None:
        node.meta["nn_module_stack"] = {"owner": owner}
    graph.output(node)
    return GraphModule(nn.Module(), graph), node


def test_plan_calibration_derives_all_materialization_requirements() -> None:
    targets = (
        _target("weight_only", parameter_name="weight_only.weight", weight=_WEIGHT),
        _target(
            "activation_only",
            parameter_name="activation_only.weight",
            activation=_ACTIVATION,
        ),
        _target(
            "weight_activation",
            parameter_name="weight_activation.weight",
            weight=_WEIGHT,
            activation=_ACTIVATION,
        ),
        _target(
            "gptq_weight",
            algorithm="gptq",
            parameter_name="gptq_weight.weight",
            weight=_WEIGHT,
        ),
        _target(
            "block.expert_weights",
            target_type="moe",
            parameter_name="block.expert_weights",
            weight=_WEIGHT,
        ),
        _target(
            "ordinary_moe",
            target_type="moe",
            parameter_name="ordinary_moe.weight",
            weight=_WEIGHT,
        ),
        _target(
            "attention.query",
            target_type="attention",
            activation=_ACTIVATION,
        ),
    )

    result = plan_calibration(targets)

    assert result.required_fqns == frozenset(
        {
            "activation_only",
            "weight_activation",
            "gptq_weight",
            "block.expert_weights",
            "attention.query",
        }
    )


def test_plan_calibration_preserves_every_gptq_alias_target() -> None:
    targets = (
        _target(
            "first",
            algorithm="gptq",
            parameter_name="first.weight",
            weight=_WEIGHT,
        ),
        _target(
            "second",
            algorithm="gptq",
            parameter_name="second.weight",
            weight=_WEIGHT,
        ),
    )

    assert plan_calibration(targets).required_fqns == frozenset(
        {"first", "second"}
    )


def test_plan_calibration_accepts_empty_target_tuple() -> None:
    assert plan_calibration(()).required_fqns == frozenset()


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("self_attn", "self_attn"),
        ("self_attn.rotary", "self_attn"),
        ("self_attn2", None),
        ("block.self_attn", None),
    ],
)
def test_calibration_attention_owner_uses_shared_fqn_rule(
    candidate: str,
    expected: str | None,
) -> None:
    graph = _graph()
    node = next(item for item in graph.graph.nodes if item.op == "call_function")
    node.meta["nn_module_stack"] = {"owner": (candidate, object())}

    snapshot = _CalibrationOwnershipSnapshot.capture(
        graph,
        ("self_attn",),
        (),
    )

    assert snapshot.attention_by_node[node] == expected


def test_calibration_attention_owner_ignores_malformed_stack() -> None:
    graph = _graph()
    node = next(item for item in graph.graph.nodes if item.op == "call_function")
    node.meta["nn_module_stack"] = {
        "string": "self_attn",
        "empty": (),
        "non_string": (1, object()),
    }

    snapshot = _CalibrationOwnershipSnapshot.capture(
        graph,
        ("self_attn",),
        (),
    )

    assert snapshot.attention_by_node[node] is None


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        (("block", "block.self_attn"), "block"),
        (("block.self_attn", "block"), "block.self_attn"),
    ],
)
def test_calibration_attention_owner_preserves_boundary_order(
    candidates: tuple[str, ...],
    expected: str,
) -> None:
    graph = _graph()
    node = next(item for item in graph.graph.nodes if item.op == "call_function")
    node.meta["nn_module_stack"] = {
        "owner": ("block.self_attn.proj", object())
    }

    snapshot = _CalibrationOwnershipSnapshot.capture(graph, candidates, ())

    assert snapshot.attention_by_node[node] == expected
    with pytest.raises(TypeError):
        snapshot.attention_by_node[node] = None  # type: ignore[index]


@pytest.mark.parametrize(
    ("owner", "moe_fqns", "expected"),
    [
        (("block.experts", object()), (), None),
        (None, ("only",), "only"),
        (None, ("first", "second"), None),
        (("second.experts", object()), ("first", "second"), "second"),
        (("first.experts", object()), ("first", "second"), "first"),
        ("broken", ("first", "second"), None),
    ],
)
def test_calibration_moe_owner_preserves_match_and_fallback(
    owner: object | None,
    moe_fqns: tuple[str, ...],
    expected: str | None,
) -> None:
    graph, node = _moe_ownership_graph(owner)

    snapshot = _CalibrationOwnershipSnapshot.capture(graph, (), moe_fqns)

    assert snapshot.moe_by_node[node] == expected


@pytest.mark.parametrize("batch_count", [1, 4])
def test_calibration_ownership_resolves_once_per_function_node(
    monkeypatch: pytest.MonkeyPatch,
    batch_count: int,
) -> None:
    graph = _attention_graph()
    lookups: list[str] = []
    records: list[str] = []
    original_lookup = ownership_module.node_owner_fqns
    original_record = _CalibrationInterpreter._record
    metadata_before = tuple(
        (
            node,
            tuple(node.meta),
            dict(node.meta),
            tuple(id(value) for value in node.meta.values()),
        )
        for node in graph.graph.nodes
    )

    def counted_lookup(node: Node) -> tuple[str, ...]:
        lookups.append(node.name)
        return original_lookup(node)

    def traced_record(
        self: _CalibrationInterpreter,
        node: Node,
        value: Any,
    ) -> None:
        if node in self.boundaries.targets_by_node:
            records.append(node.name)
        original_record(self, node, value)

    monkeypatch.setattr(ownership_module, "node_owner_fqns", counted_lookup)
    monkeypatch.setattr(_CalibrationInterpreter, "_record", traced_record)
    collect_calibration_samples(
        graph,
        [
            {"x": torch.ones(1, 1, 2, 2)}
            for _ in range(batch_count)
        ],
        CalibrationPlan(
            frozenset(
                {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "ordinary",
                    "self_attn.query",
                    "self_attn.key",
                    "self_attn.value",
                    "self_attn.score",
                }
            )
        ),
    )

    function_nodes = tuple(
        node.name for node in graph.graph.nodes if node.op == "call_function"
    )
    assert lookups == list(function_nodes)
    assert records == [
        "x",
        "q_proj",
        "k_proj",
        "v_proj",
        "score",
    ] * batch_count
    for node, keys, values, identities in metadata_before:
        assert tuple(node.meta) == keys
        assert node.meta == values
        assert tuple(id(value) for value in node.meta.values()) == identities


def test_collect_calibration_samples_preserves_attention_fqn_view() -> None:
    graph = _attention_graph()
    batches = [
        {"x": torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)},
        {"x": torch.arange(4, 8, dtype=torch.float32).reshape(1, 1, 2, 2)},
    ]
    required_fqns = frozenset(
        {
            "q_proj",
            "k_proj",
            "v_proj",
            "ordinary",
            "self_attn.query",
            "self_attn.key",
            "self_attn.value",
            "self_attn.score",
        }
    )
    actual = collect_calibration_samples(
        graph,
        batches,
        CalibrationPlan(required_fqns),
    )

    assert set(actual) == required_fqns
    assert actual["q_proj"].shape == (4, 2)
    assert actual["self_attn.query"].shape == (4, 2)
    assert actual["self_attn.score"].shape == (4, 2)


def test_collect_calibration_samples_records_shared_boundary_once() -> None:
    graph = _fanout_linear_graph()
    batches = [
        {"x": torch.tensor([[1.0, 2.0]])},
        {"x": torch.tensor([[3.0, 4.0]])},
    ]

    samples = collect_calibration_samples(
        graph,
        batches,
        _capture("first", "second"),
    )

    assert samples["first"] is samples["second"]
    assert samples["first"].shape == (2, 2)
    assert samples.boundary_key("first") == samples.boundary_key("second")
    torch.testing.assert_close(
        samples["first"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_materialization_caches_shared_boundary_activation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _fanout_linear_graph()
    samples = collect_calibration_samples(
        graph,
        [{"x": torch.tensor([[1.0, 2.0]])}],
        _capture("first", "second"),
    )
    context = MaterializationContext.capture(graph)
    calls = 0
    original = materialization_module.calculate_qparams

    def counted(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(materialization_module, "calculate_qparams", counted)
    first = materialize_target(
        context,
        _target("first", activation=_ACTIVATION),
        samples,
    )
    second = materialize_target(
        context,
        _target("second", activation=_ACTIVATION),
        samples,
    )

    assert calls == 1
    assert first.activation_qparams == second.activation_qparams
    assert first.target.scale == second.target.scale


def test_shared_boundary_supports_distinct_activation_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _fanout_linear_graph()
    samples = collect_calibration_samples(
        graph,
        [{"x": torch.tensor([[1.0, 2.0]])}],
        _capture("first", "second"),
    )
    context = MaterializationContext.capture(graph)
    calls = 0
    original = materialization_module.calculate_qparams

    def counted(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(materialization_module, "calculate_qparams", counted)
    first = materialize_target(
        context,
        _target("first", activation=_ACTIVATION),
        samples,
    )
    second = materialize_target(
        context,
        _target(
            "second",
            activation=ActivationSpec(
                bits=4,
                granularity="per_tensor",
                mode="static",
            ),
        ),
        samples,
    )

    assert samples["first"] is samples["second"]
    assert calls == 2
    assert first.activation_qparams is not None
    assert second.activation_qparams is not None
    assert first.activation_qparams["bits"] == 8
    assert second.activation_qparams["bits"] == 4


def test_alias_group_deduplicates_shared_boundary_for_gptq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _fanout_linear_graph(tied=True)
    samples = collect_calibration_samples(
        graph,
        [
            {"x": torch.tensor([[1.0, 2.0]])},
            {"x": torch.tensor([[3.0, 4.0]])},
        ],
        _capture("first", "second"),
    )
    observed_shapes: list[torch.Size] = []

    def fake_gptq(
        weight: torch.Tensor,
        activations: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        observed_shapes.append(activations.shape)
        return quantize(weight, bits=kwargs["bits"], symmetric=True, axis=0)

    monkeypatch.setattr(
        materialization_module,
        "gptq_weight_quantize",
        fake_gptq,
    )
    targets = (
        _target(
            "first",
            algorithm="gptq",
            parameter_name="first.weight",
            weight=_WEIGHT,
        ),
        _target(
            "second",
            algorithm="gptq",
            parameter_name="second.weight",
            weight=_WEIGHT,
        ),
    )

    materialize_alias_group(
        MaterializationContext.capture(graph),
        targets,
        samples,
    )

    assert observed_shapes == [torch.Size((2, 2))]


def test_collect_calibration_samples_wraps_ownership_capture_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _attention_graph()
    plan = CalibrationPlan(frozenset({"self_attn.score"}))

    def fail_lookup(node: Node) -> tuple[str, ...]:
        raise RuntimeError(f"owner lookup failed at {node.name}")

    monkeypatch.setattr(ownership_module, "node_owner_fqns", fail_lookup)

    with pytest.raises(
        QuantizationConfigError,
        match="Calibration graph execution failed: owner lookup failed at q_proj",
    ) as caught:
        collect_calibration_samples(
            graph,
            [{"x": torch.ones(1, 1, 2, 2)}],
            plan,
        )

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "owner lookup failed at q_proj"


def test_collect_calibration_samples_aggregates_linear_inputs() -> None:
    first = torch.tensor([[1.0, 2.0]], requires_grad=True)
    second = torch.tensor([[3.0, 4.0]])

    samples = collect_calibration_samples(
        _graph(),
        [{"x": first}, {"x": second}],
        _capture("linear"),
    )

    torch.testing.assert_close(
        samples["linear"],
        torch.cat((first.detach(), second)),
    )
    assert samples["linear"].device.type == "cpu"
    assert not samples["linear"].requires_grad


def test_collect_calibration_samples_accepts_mapping_order_independent_of_abi() -> None:
    samples = collect_calibration_samples(
        _graph_with_bias_input(),
        [{"bias": torch.zeros(2), "x": torch.ones(1, 2)}],
        _capture("linear"),
    )

    torch.testing.assert_close(samples["linear"], torch.ones(1, 2))


def test_collect_calibration_samples_rejects_empty_dataloader_without_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_capture(*args: object, **kwargs: object) -> None:
        raise AssertionError("ownership capture must not run")

    monkeypatch.setattr(
        _CalibrationOwnershipSnapshot,
        "capture",
        fail_capture,
    )
    with pytest.raises(
        QuantizationConfigError,
        match="must yield at least one batch",
    ):
        collect_calibration_samples(_graph(), [], _capture())


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        ({"wrong": torch.ones(1, 2)}, "Calibration keys"),
        ({"x": torch.ones(2, 2)}, "Calibration shape"),
        ({"x": torch.ones(1, 2, dtype=torch.float64)}, "Calibration dtype"),
        ({"x": torch.ones(1, 2, device="meta")}, "Calibration device"),
        ({"x": torch.tensor([[float("inf"), 0.0]])}, "contains NaN or Inf"),
    ],
)
def test_collect_calibration_samples_rejects_invalid_batches(
    monkeypatch: pytest.MonkeyPatch,
    batch: dict[str, torch.Tensor],
    message: str,
) -> None:
    def fail_capture(*args: object, **kwargs: object) -> None:
        raise AssertionError("ownership capture must not run")

    monkeypatch.setattr(
        _CalibrationOwnershipSnapshot,
        "capture",
        fail_capture,
    )
    with pytest.raises(QuantizationConfigError, match=message):
        collect_calibration_samples(_graph(), [batch], _capture())


def test_collect_calibration_samples_filters_unrequired_fqns() -> None:
    value = torch.tensor([[1.0, 2.0]])

    samples = collect_calibration_samples(
        _two_linear_graph(),
        [{"x": value}],
        _capture("first"),
    )

    assert set(samples) == {"first"}
    torch.testing.assert_close(samples["first"], value)


def test_collect_calibration_samples_empty_plan_still_executes_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    calls = 0
    original_run = _CalibrationInterpreter.run

    def counted_run(
        self: _CalibrationInterpreter,
        *args: torch.Tensor,
        **kwargs: object,
    ) -> object:
        nonlocal calls
        calls += 1
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(_CalibrationInterpreter, "run", counted_run)

    assert collect_calibration_samples(
        graph,
        [{"x": torch.ones(1, 2)}],
        _capture(),
    ) == {}
    assert calls == 1


def test_collect_calibration_samples_empty_plan_wraps_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("execution failed")

    monkeypatch.setattr(_CalibrationInterpreter, "run", fail)

    with pytest.raises(
        QuantizationConfigError,
        match="Calibration graph execution failed",
    ):
        collect_calibration_samples(
            _graph(),
            [{"x": torch.ones(1, 2)}],
            _capture(),
        )


def test_collect_calibration_samples_rejects_non_mapping_with_empty_plan() -> None:
    with pytest.raises(TypeError, match="must be mappings"):
        collect_calibration_samples(
            _graph(),
            [torch.ones(1, 2)],  # type: ignore[list-item]
            _capture(),
        )


@pytest.mark.parametrize("algorithm", ["GPTQ", "MoeExpert"])
def test_materialization_rejects_calibration_device_mismatch(
    algorithm: str,
) -> None:
    with pytest.raises(
        QuantizationConfigError,
        match=rf"{algorithm} device mismatch",
    ):
        _require_same_device(
            torch.ones(2),
            torch.ones(2, device="meta"),
            algorithm=algorithm,
            fqn="target",
        )
