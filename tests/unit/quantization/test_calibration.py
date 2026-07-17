from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Node

from mdc_llm_deploy.errors import QuantizationConfigError
from mdc_llm_deploy.graph.fx import ownership as ownership_module
from mdc_llm_deploy.graph.fx.inspection import linear_weight_name
from mdc_llm_deploy.graph.fx.ownership import node_belongs_to
from mdc_llm_deploy.graph.lifecycle import (
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    TensorAbi,
    set_metadata,
)
from mdc_llm_deploy.quantization import calibration as calibration_module
from mdc_llm_deploy.quantization.calibration import (
    _CalibrationInterpreter,
    _CalibrationOwnershipSnapshot,
    collect_calibration_samples,
)
from mdc_llm_deploy.quantization.config import ActivationSpec, WeightSpec
from mdc_llm_deploy.quantization.materialization import _require_same_device
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


class _ReferenceCalibrationInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        graph: GraphModule,
        required_fqns: frozenset[str],
        ownership: _CalibrationOwnershipSnapshot,
    ) -> None:
        super().__init__(graph, garbage_collect_values=True)
        self.required_fqns = required_fqns
        graph_metadata = calibration_module.metadata(graph)
        self.attention_fqns = tuple(
            boundary.fqn
            for boundary in graph_metadata.boundaries
            if boundary.kind == "attention"
        )
        self.moe_fqns = tuple(
            boundary.fqn
            for boundary in graph_metadata.boundaries
            if boundary.kind == "moe"
        )
        self.samples: dict[str, list[torch.Tensor]] = {}

    def _record(self, name: str, value: Any) -> None:
        if name not in self.required_fqns:
            return
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            self.samples.setdefault(name, []).append(value.detach())

    def _attention_fqn(self, node: Node) -> str | None:
        return next(
            (
                fqn
                for fqn in self.attention_fqns
                if node_belongs_to(node, fqn)
            ),
            None,
        )

    def run_node(self, node: Node) -> Any:
        """Execute one node using the pre-optimization attention lookup flow."""
        args, _ = self.fetch_args_kwargs_from_env(node)
        result = super().run_node(node)
        if node.op != "call_function":
            return result
        weight_name = linear_weight_name(node)
        if weight_name is not None:
            fqn = weight_name.removesuffix(".weight")
            self._record(fqn, args[0])
            edge = {
                "q_proj": "query",
                "k_proj": "key",
                "v_proj": "value",
            }.get(fqn.rsplit(".", 1)[-1])
            attention_fqn = self._attention_fqn(node)
            if edge is not None and attention_fqn is not None:
                self._record(f"{attention_fqn}.{edge}", result)
        attention_fqn = self._attention_fqn(node)
        if (
            attention_fqn is not None
            and node.target == torch.ops.aten.mul.Tensor
            and isinstance(result, torch.Tensor)
            and result.ndim == 4
            and result.shape[-2] == result.shape[-1]
            and any(isinstance(argument, (float, int)) for argument in args)
        ):
            self._record(f"{attention_fqn}.score", result)
        if (
            node.target
            == torch.ops.mdc_llm_deploy.moe_expert.default
            and self.moe_fqns
        ):
            owner = next(
                (
                    fqn
                    for fqn in self.moe_fqns
                    if node_belongs_to(node, fqn)
                ),
                self.moe_fqns[0] if len(self.moe_fqns) == 1 else None,
            )
            if owner is not None:
                self._record(f"{owner}.expert_weights", args[0])
        return result


def _assert_samples_exact(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> None:
    assert tuple(actual) == tuple(expected)
    for name, expected_value in expected.items():
        actual_value = actual[name]
        assert actual_value.dtype == expected_value.dtype
        assert actual_value.shape == expected_value.shape
        assert actual_value.device == expected_value.device
        assert actual_value.requires_grad == expected_value.requires_grad
        assert torch.equal(actual_value, expected_value)


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
        name: str,
        value: Any,
    ) -> None:
        records.append(name)
        original_record(self, name, value)

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
        "q_proj",
        "self_attn.query",
        "k_proj",
        "self_attn.key",
        "v_proj",
        "self_attn.value",
        "ordinary",
        "self_attn.score",
    ] * batch_count
    for node, keys, values, identities in metadata_before:
        assert tuple(node.meta) == keys
        assert node.meta == values
        assert tuple(id(value) for value in node.meta.values()) == identities


@pytest.mark.parametrize(
    "required_fqns",
    [
        frozenset({"q_proj", "self_attn.query", "self_attn.score"}),
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
        ),
        frozenset(),
    ],
)
def test_collect_calibration_samples_matches_reference_attention_lookups(
    monkeypatch: pytest.MonkeyPatch,
    required_fqns: frozenset[str],
) -> None:
    graph = _attention_graph()
    batches = [
        {"x": torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)},
        {"x": torch.arange(4, 8, dtype=torch.float32).reshape(1, 1, 2, 2)},
    ]
    plan = CalibrationPlan(required_fqns)
    candidate_interpreter = _CalibrationInterpreter

    monkeypatch.setattr(
        calibration_module,
        "_CalibrationInterpreter",
        _ReferenceCalibrationInterpreter,
    )
    expected = collect_calibration_samples(graph, batches, plan)
    monkeypatch.setattr(
        calibration_module,
        "_CalibrationInterpreter",
        candidate_interpreter,
    )
    actual = collect_calibration_samples(graph, batches, plan)

    _assert_samples_exact(expected, actual)
    assert set(actual) <= required_fqns


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
