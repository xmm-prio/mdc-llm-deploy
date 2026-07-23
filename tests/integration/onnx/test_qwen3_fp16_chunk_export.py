from __future__ import annotations

from collections import Counter
from pathlib import Path

import onnx
import pytest
import torch
from onnx.external_data_helper import uses_external_data
from transformers import Qwen3Config, Qwen3ForCausalLM

from examples.qwen3_8b_fp16_export import (
    ChunkedQwen3,
    StageSpec,
    export_stage,
    make_stage_inputs,
    position_ids_from_mask,
)
from mdc_llm_deploy.onnx.schemas import (
    FUSED_INFER_ATTENTION_SCORE_OP,
    RMS_NORM_OP,
    ROTARY_POSITION_EMBEDDING_OP,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        dtype=torch.float32,
    )
    model = Qwen3ForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    return model


def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dimension.dim_value for dimension in value.type.tensor_type.shape.dim)


def test_chunked_prefill_and_decode_return_only_current_kv(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    module = ChunkedQwen3(tiny_model)
    prefill_spec = StageSpec("prefill", query_length=3, valid_kv_length=0, kv_capacity=16)
    decode_spec = StageSpec("decode", query_length=1, valid_kv_length=3, kv_capacity=16)
    prefill_inputs = make_stage_inputs(
        tiny_model,
        prefill_spec,
        torch.device("cpu"),
        seed=0,
    )
    original_key = prefill_inputs["past_key"].clone()

    with torch.inference_mode():
        prefill_outputs = module(**prefill_inputs)
    decode_inputs = make_stage_inputs(
        tiny_model,
        decode_spec,
        torch.device("cpu"),
        seed=1,
        initial_cache=(
            prefill_outputs["present_key"],
            prefill_outputs["present_value"],
        ),
    )
    with torch.inference_mode():
        decode_outputs = module(**decode_inputs)

    assert torch.equal(prefill_inputs["past_key"], original_key)
    assert prefill_outputs["logits"].shape == (1, 3, 32)
    assert prefill_outputs["present_key"].shape == (2, 1, 2, 3, 8)
    assert prefill_outputs["present_value"].shape == (2, 1, 2, 3, 8)
    assert decode_outputs["logits"].shape == (1, 1, 32)
    assert decode_outputs["present_key"].shape == (2, 1, 2, 1, 8)
    assert decode_outputs["present_value"].shape == (2, 1, 2, 1, 8)
    assert torch.equal(
        decode_inputs["past_key"][:, :, :, :3],
        prefill_outputs["present_key"],
    )
    assert torch.equal(
        position_ids_from_mask(prefill_inputs["attention_mask"], 3),
        torch.tensor([[0, 1, 2]]),
    )
    assert torch.equal(
        position_ids_from_mask(decode_inputs["attention_mask"], 1),
        torch.tensor([[3]]),
    )


@pytest.mark.parametrize(
    ("stage_spec", "expected_input_shapes", "expected_output_shapes"),
    [
        (
            StageSpec("prefill", query_length=3, valid_kv_length=0, kv_capacity=16),
            ((1, 3), (2, 1, 2, 16, 8), (2, 1, 2, 16, 8), (1, 19)),
            ((1, 3, 32), (2, 1, 2, 3, 8), (2, 1, 2, 3, 8)),
        ),
        (
            StageSpec("decode", query_length=1, valid_kv_length=0, kv_capacity=16),
            ((1, 1), (2, 1, 2, 16, 8), (2, 1, 2, 16, 8), (1, 17)),
            ((1, 1, 32), (2, 1, 2, 1, 8), (2, 1, 2, 1, 8)),
        ),
    ],
)
def test_exported_graph_has_static_abi_and_small_operator_attention(
    tiny_model: Qwen3ForCausalLM,
    stage_spec: StageSpec,
    expected_input_shapes: tuple[tuple[int, ...], ...],
    expected_output_shapes: tuple[tuple[int, ...], ...],
    tmp_path: Path,
) -> None:
    inputs = make_stage_inputs(
        tiny_model,
        stage_spec,
        torch.device("cpu"),
        seed=0,
    )

    model_path = tmp_path / f"{stage_spec.name}.onnx"
    graph = export_stage(ChunkedQwen3(tiny_model), inputs, output_path=model_path)

    assert model_path.is_file()
    assert (tmp_path / f"{model_path.name}.data").is_file()
    assert any(uses_external_data(tensor) for tensor in graph.graph.initializer)
    assert [value.name for value in graph.graph.input] == [
        "input_ids",
        "past_key",
        "past_value",
        "attention_mask",
    ]
    assert tuple(_shape(value) for value in graph.graph.input) == expected_input_shapes
    assert [value.name for value in graph.graph.output] == [
        "logits",
        "present_key",
        "present_value",
    ]
    assert tuple(_shape(value) for value in graph.graph.output) == expected_output_shapes
    operators = Counter(node.op_type for node in graph.graph.node)
    assert operators.keys() >= {"MatMul", "Softmax"}
    assert operators[RMS_NORM_OP] == 9
    assert operators[ROTARY_POSITION_EMBEDDING_OP] == 2
    assert operators[FUSED_INFER_ATTENTION_SCORE_OP] == 0
    assert [opset.version for opset in graph.opset_import if opset.domain in ("", "ai.onnx")] == [
        18
    ]
