"""Integration tests for Qwen3 capture and multi-layer graph contracts."""

from __future__ import annotations

import pytest
import torch

from mdc_llm_deploy.errors import UnsupportedPatternError
from mdc_llm_deploy.export import convert_to_decode, export
from mdc_llm_deploy.graph import GraphStage, metadata
from tests.model_fixtures import dense_model, moe_model

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("layers", [1, 2])
def test_dense_fx_export_preserves_internal_per_layer_kv_outputs(
    layers: int,
) -> None:
    model = dense_model(8, layers=layers)
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    expected = model(**inputs)

    graph = export(model, inputs)
    actual = graph(**inputs)

    assert len(actual) == 1 + 2 * layers
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)
    assert tuple(item.name for item in metadata(graph).output_abi) == (
        "logits",
        *tuple(
            name
            for layer_id in range(layers)
            for name in (
                f"present.{layer_id}.key",
                f"present.{layer_id}.value",
            )
        ),
    )


def test_moe_export_preserves_custom_operator() -> None:
    graph = export(
        moe_model(4, expert_count=3, top_k=2),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )

    targets = {str(node.target) for node in graph.graph.nodes}
    assert any("mdc_llm_deploy.moe_expert" in target for target in targets)
    assert metadata(graph).model_kind == "moe"


def test_decode_rewrite_handles_every_layer_cache() -> None:
    graph = export(
        dense_model(8, layers=2),
        {"input_ids": torch.arange(8).reshape(1, 8)},
    )

    convert_to_decode(graph)

    value = metadata(graph)
    assert value.stage is GraphStage.FLOAT_DECODE
    assert tuple(item.name for item in value.input_abi[1:]) == (
        "past.0.key",
        "past.0.value",
        "past.1.key",
        "past.1.value",
    )
    assert tuple(item.name for item in value.output_abi[1:]) == (
        "present.0.key",
        "present.0.value",
        "present.1.key",
        "present.1.value",
    )


def test_export_rejects_training_model() -> None:
    model = dense_model(4)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        export(model, {"input_ids": torch.arange(4).reshape(1, 4)})


def test_decode_rejects_graph_without_attention() -> None:
    class NoAttention(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids.float()

    graph = export(
        NoAttention().eval(),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )
    with pytest.raises(UnsupportedPatternError, match="attention boundary"):
        convert_to_decode(graph)
