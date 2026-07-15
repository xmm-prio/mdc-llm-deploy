"""Tests for static ATen export and decode graph rewriting."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from mdc_llm_deploy.errors import GraphStateError, UnsupportedPatternError
from mdc_llm_deploy.export import convert_to_decode, export
from mdc_llm_deploy.graph import (
    GRAPH_SCHEMA_VERSION,
    GraphStage,
    QuantizedTarget,
    metadata,
    set_metadata,
)
from mdc_llm_deploy.models.tiny import TinyAttention, TinyQwen3Dense


def _inputs(sequence: int = 8) -> dict[str, torch.Tensor]:
    return {"input_ids": (torch.arange(sequence) % 128).reshape(1, sequence)}


def test_export_is_static_aten_and_numerically_equal() -> None:
    model = TinyQwen3Dense().eval()
    inputs = _inputs()
    expected = model(**inputs)

    graph = export(model, inputs)
    actual = graph(**inputs)

    assert all(node.op not in {"call_module", "call_method"} for node in graph.graph.nodes)
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.key_cache, expected.key_cache)
    torch.testing.assert_close(actual.value_cache, expected.value_cache)
    value = metadata(graph)
    assert value.schema_version == GRAPH_SCHEMA_VERSION
    assert value.stage is GraphStage.FLOAT_PREFILL
    assert tuple(item.name for item in value.output_abi) == (
        "logits",
        "present.0.key",
        "present.0.value",
    )
    assert {"rms_norm", "rope", "attention"} <= {
        item.kind for item in value.boundaries
    }
    assert all(item.nodes for item in value.boundaries)


def test_boundary_discovery_uses_structure_not_class_name() -> None:
    class OpaqueBlock(TinyAttention):
        pass

    model = TinyQwen3Dense().eval()
    dtype = next(model.parameters()).dtype
    model.self_attn = OpaqueBlock(model.config).to(dtype=dtype).eval()

    graph = export(model, _inputs())

    assert any(item.kind == "attention" for item in metadata(graph).boundaries)


def test_decode_rewrites_attention_position_and_cache() -> None:
    model = TinyQwen3Dense().eval()
    full = _inputs()
    graph = export(model, full)
    expected = model(**full)
    prefix = model(input_ids=full["input_ids"][:, :-1])

    same = convert_to_decode(graph)
    actual = same(
        full["input_ids"][:, -1:],
        prefix.key_cache,
        prefix.value_cache,
    )

    assert same is graph
    value = metadata(graph)
    assert value.stage is GraphStage.FLOAT_DECODE
    assert value.absolute_position == 7
    assert value.properties["decode_rewrite"] is True
    assert tuple(item.name for item in value.input_abi) == (
        "input_ids",
        "past_key_values.0.key",
        "past_key_values.0.value",
    )
    assert value.input_abi[1].shape == (1, 2, 7, 16)
    assert value.output_abi[1].shape == (1, 2, 8, 16)
    torch.testing.assert_close(actual[0], expected.logits[:, -1:], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[1], expected.key_cache, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[2], expected.value_cache, atol=1e-5, rtol=1e-5)


def test_decode_int8_cache_uses_position_qparams() -> None:
    model = TinyQwen3Dense().eval()
    graph = export(model, _inputs())
    value = metadata(graph)
    targets = (
        QuantizedTarget(
            "self_attn.key",
            "attention",
            "minmax",
            8,
            "per_tensor",
            True,
            (0.01,),
            (0,),
        ),
        QuantizedTarget(
            "self_attn.value",
            "attention",
            "minmax",
            8,
            "per_tensor",
            True,
            (0.02,),
            (0,),
        ),
    )
    set_metadata(
        graph,
        replace(
            value,
            stage=GraphStage.QUANTIZED_PREFILL,
            quantized_targets=targets,
            config_fingerprint="a" * 64,
        ),
    )
    prefix = model(input_ids=_inputs()["input_ids"][:, :-1])

    convert_to_decode(graph)
    result = graph(
        _inputs()["input_ids"][:, -1:],
        torch.round(prefix.key_cache / 0.01).clamp(-128, 127).to(torch.int8),
        torch.round(prefix.value_cache / 0.02).clamp(-128, 127).to(torch.int8),
    )

    assert metadata(graph).input_abi[1].dtype == "int8"
    assert metadata(graph).output_abi[1].dtype == "int8"
    assert result[1].dtype is torch.int8
    assert result[2].dtype is torch.int8
    assert result[1].shape == (1, 2, 8, 16)


def test_decode_failures_are_atomic() -> None:
    model = TinyQwen3Dense().eval()
    graph = export(model, _inputs())
    value = metadata(graph)
    set_metadata(
        graph,
        replace(
            value,
            boundaries=tuple(
                item for item in value.boundaries if item.kind != "attention"
            ),
        ),
    )
    before = copy.deepcopy(graph)

    with pytest.raises(UnsupportedPatternError, match="attention boundary"):
        convert_to_decode(graph)

    assert str(graph.graph) == str(before.graph)
    assert metadata(graph) == metadata(before)


def test_decode_rejects_repeated_conversion() -> None:
    graph = export(TinyQwen3Dense().eval(), _inputs())
    convert_to_decode(graph)

    with pytest.raises(GraphStateError, match="prefill"):
        convert_to_decode(graph)


def test_export_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="eval"):
        export(TinyQwen3Dense().train(), _inputs())

    class NoAttention(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids.float()

    graph = export(NoAttention().eval(), _inputs())
    with pytest.raises(UnsupportedPatternError, match="attention boundary"):
        convert_to_decode(graph)
