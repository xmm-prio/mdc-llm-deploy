from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.fx import GraphModule, symbolic_trace

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
