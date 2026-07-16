from __future__ import annotations

import ast
import copy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.fx import GraphModule, symbolic_trace

import mdc_llm_deploy.graph as graph
import mdc_llm_deploy.graph_contract as graph_contract
from mdc_llm_deploy.errors import GraphStateError, UnsupportedPatternError
from mdc_llm_deploy.graph import (
    GRAPH_METADATA_KEY,
    GRAPH_SCHEMA_VERSION,
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    QuantizedTarget,
    TensorAbi,
    metadata,
    set_metadata,
    transactional_update,
    validate_capability_request,
    validate_graph,
    validate_metadata,
)


@pytest.mark.parametrize(
    "name",
    [
        "GRAPH_METADATA_KEY",
        "GRAPH_SCHEMA_VERSION",
        "GraphStage",
        "TensorAbi",
        "FusionBoundary",
        "QuantizedTarget",
        "GraphMetadata",
        "validate_metadata",
        "validate_capability_request",
        "require_boundaries",
    ],
)
def test_legacy_graph_contract_exports_preserve_identity(name: str) -> None:
    assert getattr(graph, name) is getattr(graph_contract, name)


def test_graph_contract_has_only_allowed_dependency_roots() -> None:
    source_path = Path(graph_contract.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        (node.module or "").split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert imported_roots <= {
        "__future__",
        "capabilities",
        "errors",
        "graph_types",
        "graph_validation",
        "immutable_json",
        "model_properties",
    }


def test_importing_graph_contract_does_not_load_torch() -> None:
    project_root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "assert 'torch' not in sys.modules; "
                "import mdc_llm_deploy.graph_contract; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


class Scale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


def _metadata(**changes: object) -> GraphMetadata:
    value = GraphMetadata(
        schema_version=GRAPH_SCHEMA_VERSION,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="dense",
        input_abi=(TensorAbi("x", "float32", (1,)),),
        output_abi=(TensorAbi("output", "float32", (1,)),),
        boundaries=(FusionBoundary("linear", "scale"),),
        sequence_length=1,
        properties={"opset": 18},
    )
    return replace(value, **changes)


def _graph() -> GraphModule:
    graph = symbolic_trace(Scale().eval())
    set_metadata(graph, _metadata())
    return graph


def _quantized_target(**changes: object) -> QuantizedTarget:
    value = QuantizedTarget(
        fqn="linear",
        target_type="linear",
        algorithm="minmax",
        bits=8,
        granularity="per_channel",
        symmetric=True,
        scale=(0.5,),
        zero_point=(0,),
    )
    return replace(value, **changes)


def test_versioned_metadata_and_full_graph_validate() -> None:
    graph = _graph()

    assert validate_graph(graph) is metadata(graph)
    assert metadata(graph).schema_version == 1


def test_metadata_properties_are_a_deeply_immutable_snapshot() -> None:
    source = {
        "nested": {"items": [1, {"enabled": True}]},
        "labels": ["first", "second"],
    }

    value = _metadata(properties=source)
    source["nested"]["items"].append(3)
    source["nested"]["items"][1]["enabled"] = False
    source["labels"][0] = "changed"
    source["added"] = "later"

    assert value.properties == {
        "nested": {"items": (1, {"enabled": True})},
        "labels": ("first", "second"),
    }
    with pytest.raises(TypeError):
        value.properties["added"] = "blocked"  # type: ignore[index]
    with pytest.raises(TypeError):
        value.properties["nested"]["added"] = "blocked"  # type: ignore[index]
    with pytest.raises(TypeError):
        value.properties["labels"][0] = "blocked"  # type: ignore[index]


def test_replace_and_deepcopy_preserve_frozen_property_values() -> None:
    value = _metadata(properties={"nested": {"items": [1, 2]}})

    replaced = replace(value, sequence_length=2)
    copied = copy.deepcopy(value)
    graph = _graph()
    copied_graph = copy.deepcopy(graph)

    assert replaced.properties == value.properties
    assert copied == value
    assert copied is not value
    assert metadata(copied_graph) == metadata(graph)
    with pytest.raises(TypeError):
        replaced.properties["nested"]["items"][0] = 9  # type: ignore[index]


def test_set_and_get_metadata_preserve_instance_identity() -> None:
    graph = symbolic_trace(Scale().eval())
    value = _metadata()

    set_metadata(graph, value)

    assert graph.meta[GRAPH_METADATA_KEY] is value
    assert metadata(graph) is value


def test_metadata_properties_reject_circular_references_stably() -> None:
    properties: dict[str, object] = {}
    properties["self"] = properties

    with pytest.raises(GraphStateError, match="circular references"):
        _metadata(properties=properties)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 0}, "Unsupported graph metadata version"),
        ({"model_kind": "hybrid"}, "Unsupported model kind"),
        (
            {"input_abi": (TensorAbi("x", "float32", (-1,)),)},
            "static and positive",
        ),
        (
            {
                "boundaries": (
                    FusionBoundary("linear", "first", ("node",)),
                    FusionBoundary("attention", "second", ("node",)),
                )
            },
            "multiple owners",
        ),
    ],
)
def test_metadata_rejects_contract_boundaries(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(GraphStateError, match=message):
        validate_metadata(_metadata(**changes))


def test_quantized_metadata_checks_cross_module_capability() -> None:
    target = _quantized_target(
        target_type="moe",
        fqn="experts.0",
        granularity="per_tensor",
    )
    value = _metadata(
        stage=GraphStage.QUANTIZED_PREFILL,
        quantized_targets=(target,),
        config_fingerprint="a" * 64,
    )

    with pytest.raises(GraphStateError, match="Unsupported capability"):
        validate_metadata(value)


def test_decode_position_is_derived_from_sequence_length() -> None:
    value = _metadata(
        stage=GraphStage.FLOAT_DECODE,
        sequence_length=8,
        absolute_position=6,
    )

    with pytest.raises(GraphStateError, match="sequence_length - 1"):
        validate_metadata(value)


@pytest.mark.parametrize("absolute_position", [True, 7.0])
def test_decode_position_requires_exact_int(absolute_position: object) -> None:
    value = _metadata(
        stage=GraphStage.FLOAT_DECODE,
        sequence_length=8,
        absolute_position=absolute_position,
    )

    with pytest.raises(GraphStateError, match="must be an integer"):
        validate_metadata(value)


@pytest.mark.parametrize("scale", [(1,), (float("inf"),), (float("nan"),), (10**400,)])
def test_quantized_target_scale_requires_finite_positive_float(
    scale: tuple[object, ...],
) -> None:
    target = _quantized_target(scale=scale)
    value = _metadata(
        stage=GraphStage.QUANTIZED_PREFILL,
        quantized_targets=(target,),
        config_fingerprint="d" * 64,
    )

    with pytest.raises(GraphStateError, match="finite and positive"):
        validate_metadata(value)


@pytest.mark.parametrize("kind", [None, ["linear"]])
def test_fusion_boundary_kind_rejects_non_strings_stably(kind: object) -> None:
    value = _metadata(boundaries=(FusionBoundary(kind, "scale"),))

    with pytest.raises(GraphStateError, match="Unsupported fusion boundary kind"):
        validate_metadata(value)


def test_empty_fusion_boundary_nodes_remain_valid() -> None:
    validate_metadata(_metadata(boundaries=(FusionBoundary("linear", "scale", ()),)))


def test_onnx_activation_qparams_accept_frozen_mapping_and_tuples() -> None:
    from mdc_llm_deploy.onnx_export.lowering_support import activation_target

    target = _quantized_target()
    value = _metadata(
        properties={
            "activation_qparams": {
                target.fqn: {
                    "bits": 8,
                    "granularity": "per_tensor",
                    "symmetric": False,
                    "scale": [0.25],
                    "zero_point": [3],
                }
            }
        }
    )

    activated = activation_target(value, target)

    assert activated.bits == 8
    assert activated.granularity == "per_tensor"
    assert activated.symmetric is False
    assert activated.scale == (0.25,)
    assert activated.zero_point == (3,)


def test_graph_capability_request_rejects_gptq_onnx() -> None:
    target = _quantized_target(
        algorithm="gptq",
        bits=4,
        granularity="per_channel",
    )
    value = _metadata(
        stage=GraphStage.QUANTIZED_PREFILL,
        quantized_targets=(target,),
        config_fingerprint="b" * 64,
    )

    with pytest.raises(UnsupportedPatternError, match="GPTQ is FX-only"):
        validate_capability_request(value, mask_mode="masked", artifact="onnx")


def test_w4_metadata_is_valid_for_fx_but_not_onnx() -> None:
    target = _quantized_target(bits=4)
    value = _metadata(
        stage=GraphStage.QUANTIZED_PREFILL,
        quantized_targets=(target,),
        config_fingerprint="c" * 64,
    )

    assert validate_capability_request(value, mask_mode="masked", artifact="fx")
    with pytest.raises(UnsupportedPatternError, match="W4 is FX-only"):
        validate_capability_request(value, mask_mode="masked", artifact="onnx")


def test_asymmetric_attention_edge_is_fx_only() -> None:
    target = _quantized_target(
        fqn="layers.0.attention.query",
        target_type="attention",
        granularity="per_tensor",
        symmetric=False,
    )
    value = _metadata(
        stage=GraphStage.QUANTIZED_PREFILL,
        quantized_targets=(target,),
        config_fingerprint="d" * 64,
    )

    assert validate_capability_request(
        value,
        mask_mode="masked",
        artifact="fx",
    )
    with pytest.raises(
        UnsupportedPatternError,
        match="Asymmetric attention query/score is FX-only",
    ):
        validate_capability_request(
            value,
            mask_mode="masked",
            artifact="onnx",
        )


def test_non_release_rms_epsilon_is_fx_only() -> None:
    value = _metadata(
        properties={
            "opset": 18,
            "rms_norm_epsilon": 1e-5,
        }
    )

    assert validate_capability_request(
        value,
        mask_mode="masked",
        artifact="fx",
    )
    with pytest.raises(
        UnsupportedPatternError,
        match="require RmsNorm epsilon=1e-6",
    ):
        validate_capability_request(
            value,
            mask_mode="masked",
            artifact="onnx",
        )


def test_deployment_artifacts_require_attention_boundary() -> None:
    value = _metadata()

    assert validate_capability_request(
        value,
        mask_mode="masked",
        artifact="fx",
    )
    with pytest.raises(
        UnsupportedPatternError,
        match="Missing fusion boundaries: attention",
    ):
        validate_capability_request(
            value,
            mask_mode="masked",
            artifact="onnx",
        )


def test_transaction_commits_valid_candidate_and_preserves_identity() -> None:
    graph = _graph()
    identity = id(graph)

    def mutate(candidate: GraphModule) -> None:
        with torch.no_grad():
            candidate.weight.fill_(3.0)
        current = metadata(candidate)
        set_metadata(candidate, replace(current, properties={"opset": 18, "revision": 2}))

    result = transactional_update(graph, mutate)

    assert result is graph
    assert id(graph) == identity
    assert graph(torch.ones(1)).item() == pytest.approx(3.0)
    assert metadata(graph).properties["revision"] == 2


def test_transaction_recompiles_structural_graph_changes() -> None:
    graph = _graph()

    def mutate(candidate: GraphModule) -> None:
        operation = next(
            node
            for node in candidate.graph.nodes
            if node.op == "call_function"
        )
        operation.target = torch.neg
        operation.args = (operation.args[0],)

    transactional_update(graph, mutate)

    assert graph(torch.ones(1)).item() == pytest.approx(-1.0)


def test_transaction_failure_leaves_graph_parameters_and_metadata_unchanged() -> None:
    graph = _graph()
    original_code = graph.code
    original_weight = graph.weight.detach().clone()
    original_metadata = metadata(graph)

    def mutate(candidate: GraphModule) -> None:
        with torch.no_grad():
            candidate.weight.fill_(99.0)
        candidate.meta[GRAPH_METADATA_KEY] = replace(
            metadata(candidate),
            model_kind="unsupported",
        )

    with pytest.raises(GraphStateError, match="Unsupported model kind"):
        transactional_update(graph, mutate)

    assert graph.code == original_code
    assert torch.equal(graph.weight, original_weight)
    assert metadata(graph) == original_metadata
    assert graph(torch.ones(1)).item() == pytest.approx(2.0)
