from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import torch
from onnx import TensorProto
from transformers import Qwen3Config, Qwen3ForCausalLM

from examples.qwen3_4b_fp16_tp_export import (
    MAX_CHUNK_SIZE,
    ChunkedQwen3,
    apply_tp_sharding,
    export_stage,
    make_export_inputs,
    register_hcom_schema,
    validate_export,
)

pytestmark = pytest.mark.integration


def _tiny_rank_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        dtype=torch.float16,
    )
    model = Qwen3ForCausalLM(config).eval().to(dtype=torch.float16)
    model.set_attn_implementation("eager")
    apply_tp_sharding(model, rank=0)
    return model


def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dimension.dim_value for dimension in value.type.tensor_type.shape.dim)


def test_chunk_scatter_updates_only_actual_sequence_range() -> None:
    model = _tiny_rank_model()
    inputs = make_export_inputs(model, 2, 8, torch.device("cpu"), seed=0)
    inputs["actual_seq_len"].fill_(3)
    inputs["attention_mask"].zero_()
    inputs["past_key"].normal_()
    inputs["past_value"].normal_()
    original_key = inputs["past_key"].clone()
    original_value = inputs["past_value"].clone()

    with torch.inference_mode():
        outputs = ChunkedQwen3(model)(**inputs)

    assert inputs["actual_seq_len"].shape == ()
    assert inputs["actual_seq_len"].dtype == torch.int64
    assert inputs["attention_mask"].shape == (1, 1, MAX_CHUNK_SIZE, 8)
    assert inputs["attention_mask"].dtype == torch.float16
    assert torch.equal(inputs["past_key"], original_key)
    assert torch.equal(inputs["past_value"], original_value)
    assert torch.equal(outputs["present_key"][:, :, :, :3], original_key[:, :, :, :3])
    assert torch.equal(outputs["present_value"][:, :, :, :3], original_value[:, :, :, :3])
    assert torch.equal(outputs["present_key"][:, :, :, 5:], original_key[:, :, :, 5:])
    assert torch.equal(outputs["present_value"][:, :, :, 5:], original_value[:, :, :, 5:])
    assert not torch.equal(outputs["present_key"][:, :, :, 3:5], original_key[:, :, :, 3:5])
    assert not torch.equal(outputs["present_value"][:, :, :, 3:5], original_value[:, :, :, 3:5])


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_exported_chunk_graph_has_static_abi(
    chunk_size: int,
    tmp_path: Path,
) -> None:
    register_hcom_schema()
    model = _tiny_rank_model()
    inputs = make_export_inputs(model, chunk_size, 8, torch.device("cpu"), seed=0)
    path = tmp_path / f"chunk_{chunk_size}.onnx"

    graph = export_stage(ChunkedQwen3(model), inputs, path)
    validate_export(
        graph,
        path,
        inputs=inputs,
        num_hidden_layers=model.config.num_hidden_layers,
        vocab_size=model.config.vocab_size,
    )

    assert [value.name for value in graph.graph.input] == [
        "input_ids",
        "attention_mask",
        "actual_seq_len",
        "past_key",
        "past_value",
    ]
    assert [_shape(value) for value in graph.graph.input] == [
        (1, chunk_size),
        (1, 1, MAX_CHUNK_SIZE, 8),
        (),
        (1, 1, 2, 8, 8),
        (1, 1, 2, 8, 8),
    ]
    assert [value.type.tensor_type.elem_type for value in graph.graph.input] == [
        TensorProto.INT64,
        TensorProto.FLOAT16,
        TensorProto.INT64,
        TensorProto.FLOAT16,
        TensorProto.FLOAT16,
    ]
    assert [_shape(value) for value in graph.graph.output] == [
        (1, chunk_size, 32),
        (1, 1, 2, 8, 8),
        (1, 1, 2, 8, 8),
    ]


@pytest.mark.parametrize("chunk_size", [0, MAX_CHUNK_SIZE + 1])
def test_make_export_inputs_rejects_invalid_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be within"):
        make_export_inputs(
            _tiny_rank_model(),
            chunk_size,
            8,
            torch.device("cpu"),
            seed=0,
        )
