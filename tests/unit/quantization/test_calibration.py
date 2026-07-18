from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Node

import mdc_llm_deploy.quantization.calibration as calibration_module
from mdc_llm_deploy.errors import QuantizationConfigError
from mdc_llm_deploy.graph.fx import ownership as ownership_module
from mdc_llm_deploy.graph.lifecycle import (
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    TensorAbi,
    metadata,
    set_metadata,
)
from mdc_llm_deploy.quantization import (
    materialization as materialization_module,
)
from mdc_llm_deploy.quantization import (
    oneshot,
)
from mdc_llm_deploy.quantization.algorithms.math import (
    calculate_qparams,
    quantize,
)
from mdc_llm_deploy.quantization.calibration import (
    _CalibrationBoundaryMap,
    _CalibrationInterpreter,
    _CalibrationOwnershipSnapshot,
    collect_calibration_artifacts,
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
    CalibrationRequirement,
    TargetPlan,
    plan_calibration,
)

_WEIGHT = WeightSpec(bits=8, granularity="per_channel")
_ACTIVATION = ActivationSpec(bits=8, granularity="per_tensor", mode="static")
_NONFINITE_CASES = (
    ("per_tensor", float("inf")),
    ("per_tensor", float("nan")),
    ("per_token", float("inf")),
    ("per_token", float("nan")),
)


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


def _capture(
    *fqns: str,
    full_samples: bool = False,
    activation: ActivationSpec | None = _ACTIVATION,
) -> CalibrationPlan:
    targets = tuple(
        _target(
            fqn,
            algorithm="gptq" if full_samples else "minmax",
            parameter_name=f"{fqn}.weight" if full_samples else None,
            weight=_WEIGHT if full_samples else None,
            activation=activation,
        )
        for fqn in fqns
    )
    return plan_calibration(targets)


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
    assert result.requirements["activation_only"] == CalibrationRequirement(
        frozenset({_ACTIVATION}),
        False,
    )
    assert result.requirements["gptq_weight"] == CalibrationRequirement(
        frozenset(),
        True,
    )
    assert result.requirements["block.expert_weights"] == CalibrationRequirement(
        frozenset(),
        True,
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
    plan = plan_calibration(())

    assert plan.requires_collection is False
    assert plan.required_fqns == frozenset()
    assert dict(plan.requirements) == {}


def test_plan_calibration_requires_collection_for_each_artifact_kind() -> None:
    activation = plan_calibration((_target("linear", activation=_ACTIVATION),))
    gptq = plan_calibration(
        (
            _target(
                "linear",
                algorithm="gptq",
                parameter_name="linear.weight",
                weight=_WEIGHT,
            ),
        )
    )
    packed_moe = plan_calibration(
        (
            _target(
                "block.expert_weights",
                target_type="moe",
                parameter_name="block.expert_weights",
                weight=_WEIGHT,
            ),
        )
    )

    assert activation.requires_collection is True
    assert gptq.requires_collection is True
    assert packed_moe.requires_collection is True


def test_plan_calibration_merges_requirements_order_independently() -> None:
    four_bit = ActivationSpec(
        bits=4,
        granularity="per_token",
        mode="static",
        symmetric=False,
    )
    targets = (
        _target("shared", activation=_ACTIVATION),
        _target(
            "shared",
            algorithm="gptq",
            parameter_name="shared.weight",
            weight=_WEIGHT,
            activation=four_bit,
        ),
    )

    forward = plan_calibration(targets)
    reverse = plan_calibration(tuple(reversed(targets)))

    expected = CalibrationRequirement(
        frozenset({_ACTIVATION, four_bit}),
        True,
    )
    assert forward.requirements["shared"] == expected
    assert reverse.requirements["shared"] == expected
    with pytest.raises(TypeError):
        forward.requirements["other"] = expected  # type: ignore[index]


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
    collect_calibration_artifacts(
        graph,
        [
            {"x": torch.ones(1, 1, 2, 2)}
            for _ in range(batch_count)
        ],
        _capture(
            "q_proj",
            "k_proj",
            "v_proj",
            "ordinary",
            "self_attn.query",
            "self_attn.key",
            "self_attn.value",
            "self_attn.score",
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


def test_collect_calibration_artifacts_preserves_attention_fqn_view() -> None:
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
    actual = collect_calibration_artifacts(
        graph,
        batches,
        _capture(*required_fqns),
    )

    for fqn in required_fqns:
        scale, zero_point = actual.qparams(fqn, _ACTIVATION)
        assert scale.shape == torch.Size([])
        assert zero_point.shape == torch.Size([])
        with pytest.raises(KeyError):
            actual.samples(fqn)


def test_collect_calibration_artifacts_records_shared_boundary_once() -> None:
    graph = _fanout_linear_graph()
    batches = [
        {"x": torch.tensor([[1.0, 2.0]])},
        {"x": torch.tensor([[3.0, 4.0]])},
    ]

    artifacts = collect_calibration_artifacts(
        graph,
        batches,
        _capture("first", "second"),
    )

    first = artifacts.qparams("first", _ACTIVATION)
    second = artifacts.qparams("second", _ACTIVATION)
    expected = calculate_qparams(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        bits=8,
        symmetric=True,
    )
    assert first is second
    torch.testing.assert_close(first[0], expected[0])
    torch.testing.assert_close(first[1], expected[1])
    with pytest.raises(KeyError):
        artifacts.samples("first")


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("granularity", ["per_tensor", "per_token"])
def test_streamed_qparams_match_full_sample_reference(
    bits: int,
    symmetric: bool,
    granularity: str,
) -> None:
    spec = ActivationSpec(
        bits=bits,  # type: ignore[arg-type]
        granularity=granularity,  # type: ignore[arg-type]
        mode="static",
        symmetric=symmetric,
    )
    batches = (
        torch.tensor([[0.0, 2.0]]),
        torch.tensor([[-4.0, -1.0]]),
        torch.tensor([[3.0, 0.0]]),
    )
    artifacts = collect_calibration_artifacts(
        _graph(),
        [{"x": batch} for batch in batches],
        _capture("linear", activation=spec),
    )

    actual = artifacts.qparams("linear", spec)
    expected = calculate_qparams(
        torch.cat(batches),
        bits=bits,
        symmetric=symmetric,
        axis=0 if granularity == "per_token" else None,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert torch.equal(actual[1], expected[1])
    assert actual[0].dtype is torch.float32
    assert actual[1].dtype is torch.int32


@pytest.mark.parametrize(("granularity", "nonfinite"), _NONFINITE_CASES)
def test_streamed_qparams_reject_nonfinite_intermediate_activation(
    granularity: str,
    nonfinite: float,
) -> None:
    graph = _attention_graph()
    with torch.no_grad():
        graph.q_proj.weight.fill_(nonfinite)
    spec = ActivationSpec(
        bits=8,
        granularity=granularity,  # type: ignore[arg-type]
        mode="static",
    )

    with pytest.raises(ValueError, match=r"^tensor contains NaN or Inf$"):
        collect_calibration_artifacts(
            graph,
            [{"x": torch.ones(1, 1, 2, 2)}],
            _capture("self_attn.query", activation=spec),
        )


@pytest.mark.parametrize(("granularity", "nonfinite"), _NONFINITE_CASES)
def test_oneshot_preserves_nonfinite_activation_error_and_graph(
    granularity: str,
    nonfinite: float,
) -> None:
    graph = _attention_graph()
    with torch.no_grad():
        graph.q_proj.weight.fill_(nonfinite)
    graph_before = str(graph.graph)
    metadata_before = metadata(graph)
    parameter_before = graph.q_proj.weight
    config = {
        "modifiers": [
            {
                "type": "minmax",
                "attention": {
                    "query": {
                        "bits": 8,
                        "granularity": granularity,
                        "mode": "static",
                        "symmetric": True,
                    }
                },
            }
        ]
    }

    with pytest.raises(ValueError, match=r"^tensor contains NaN or Inf$"):
        oneshot(
            graph,
            config,
            [{"x": torch.ones(1, 1, 2, 2)}],
        )

    assert str(graph.graph) == graph_before
    assert metadata(graph) == metadata_before
    assert graph.q_proj.weight is parameter_before


def test_combined_requirement_produces_qparams_and_full_samples() -> None:
    target = _target(
        "linear",
        algorithm="gptq",
        parameter_name="linear.weight",
        weight=_WEIGHT,
        activation=_ACTIVATION,
    )
    batches = (
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
    )

    artifacts = collect_calibration_artifacts(
        _graph(),
        [{"x": batch} for batch in batches],
        plan_calibration((target,)),
    )

    torch.testing.assert_close(artifacts.samples("linear"), torch.cat(batches))
    actual = artifacts.qparams("linear", _ACTIVATION)
    expected = calculate_qparams(
        torch.cat(batches),
        bits=8,
        symmetric=True,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert torch.equal(actual[1], expected[1])


def test_materialization_uses_shared_boundary_activation_contract() -> None:
    graph = _fanout_linear_graph()
    samples = collect_calibration_artifacts(
        graph,
        [{"x": torch.tensor([[1.0, 2.0]])}],
        _capture("first", "second"),
    )
    context = MaterializationContext.capture(graph)
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

    assert samples.qparams("first", _ACTIVATION) is samples.qparams(
        "second",
        _ACTIVATION,
    )
    assert first.activation_qparams == second.activation_qparams
    assert first.target.scale == second.target.scale


def test_shared_boundary_supports_distinct_activation_contracts() -> None:
    graph = _fanout_linear_graph()
    four_bit = ActivationSpec(
        bits=4,
        granularity="per_tensor",
        mode="static",
    )
    samples = collect_calibration_artifacts(
        graph,
        [{"x": torch.tensor([[1.0, 2.0]])}],
        plan_calibration(
            (
                _target("first", activation=_ACTIVATION),
                _target("second", activation=four_bit),
            )
        ),
    )
    context = MaterializationContext.capture(graph)
    first = materialize_target(
        context,
        _target("first", activation=_ACTIVATION),
        samples,
    )
    second = materialize_target(
        context,
        _target(
            "second",
            activation=four_bit,
        ),
        samples,
    )

    assert first.activation_qparams is not None
    assert second.activation_qparams is not None
    assert first.activation_qparams["bits"] == 8
    assert second.activation_qparams["bits"] == 4


def test_alias_group_deduplicates_shared_boundary_for_gptq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _fanout_linear_graph(tied=True)
    samples = collect_calibration_artifacts(
        graph,
        [
            {"x": torch.tensor([[1.0, 2.0]])},
            {"x": torch.tensor([[3.0, 4.0]])},
        ],
        _capture("first", "second", full_samples=True, activation=None),
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


def test_collect_calibration_artifacts_wraps_ownership_capture_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _attention_graph()
    plan = _capture("self_attn.score")

    def fail_lookup(node: Node) -> tuple[str, ...]:
        raise RuntimeError(f"owner lookup failed at {node.name}")

    monkeypatch.setattr(ownership_module, "node_owner_fqns", fail_lookup)

    with pytest.raises(
        QuantizationConfigError,
        match="Calibration graph execution failed: owner lookup failed at q_proj",
    ) as caught:
        collect_calibration_artifacts(
            graph,
            [{"x": torch.ones(1, 1, 2, 2)}],
            plan,
        )

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "owner lookup failed at q_proj"


def test_collect_calibration_artifacts_aggregates_linear_inputs() -> None:
    first = torch.tensor([[1.0, 2.0]], requires_grad=True)
    second = torch.tensor([[3.0, 4.0]])

    artifacts = collect_calibration_artifacts(
        _graph(),
        [{"x": first}, {"x": second}],
        _capture("linear", full_samples=True, activation=None),
    )

    samples = artifacts.samples("linear")
    torch.testing.assert_close(
        samples,
        torch.cat((first.detach(), second)),
    )
    assert samples.device.type == "cpu"
    assert not samples.requires_grad


def test_collect_calibration_artifacts_accepts_mapping_order_independent_of_abi() -> None:
    artifacts = collect_calibration_artifacts(
        _graph_with_bias_input(),
        [{"bias": torch.zeros(2), "x": torch.ones(1, 2)}],
        _capture("linear", full_samples=True, activation=None),
    )

    torch.testing.assert_close(artifacts.samples("linear"), torch.ones(1, 2))


def test_collect_calibration_artifacts_rejects_empty_dataloader_without_capture(
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
        collect_calibration_artifacts(_graph(), [], _capture("linear"))


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
def test_collect_calibration_artifacts_rejects_invalid_batches(
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
        collect_calibration_artifacts(_graph(), [batch], _capture("linear"))


def test_collect_calibration_artifacts_filters_unrequired_fqns() -> None:
    value = torch.tensor([[1.0, 2.0]])

    artifacts = collect_calibration_artifacts(
        _two_linear_graph(),
        [{"x": value}],
        _capture("first"),
    )

    artifacts.qparams("first", _ACTIVATION)
    with pytest.raises(KeyError):
        artifacts.qparams("second", _ACTIVATION)


def test_collect_calibration_artifacts_empty_plan_does_not_iterate_or_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailOnIteration:
        def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:
            raise AssertionError("calibration dataloader must not be iterated")

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("calibration execution path must not run")

    monkeypatch.setattr(calibration_module, "metadata", fail)
    monkeypatch.setattr(_CalibrationOwnershipSnapshot, "capture", fail)
    monkeypatch.setattr(_CalibrationBoundaryMap, "capture", fail)
    monkeypatch.setattr(_CalibrationInterpreter, "run", fail)

    artifacts = collect_calibration_artifacts(
        _graph(),
        FailOnIteration(),
        _capture(),
    )
    with pytest.raises(KeyError):
        artifacts.qparams("linear", _ACTIVATION)
    with pytest.raises(KeyError):
        artifacts.samples("linear")
    with pytest.raises(KeyError):
        artifacts.sample_items("linear")


def test_collect_calibration_artifacts_nonempty_plan_wraps_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("execution failed")

    monkeypatch.setattr(_CalibrationInterpreter, "run", fail)

    with pytest.raises(
        QuantizationConfigError,
        match="Calibration graph execution failed",
    ):
        collect_calibration_artifacts(
            _graph(),
            [{"x": torch.ones(1, 2)}],
            _capture("linear"),
        )


def test_collect_calibration_artifacts_rejects_non_mapping_with_nonempty_plan() -> None:
    with pytest.raises(TypeError, match="must be mappings"):
        collect_calibration_artifacts(
            _graph(),
            [torch.ones(1, 2)],  # type: ignore[list-item]
            _capture("linear"),
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
