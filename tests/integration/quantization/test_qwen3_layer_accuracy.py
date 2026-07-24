from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace

import onnx
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.exporters import OnnxConfig, OnnxExporter

from examples.qwen3_8b_layer_accuracy.artifacts import LAYER_ADAPTER_CONFIG
from examples.qwen3_8b_layer_accuracy.metrics import SaturationCollector, compare_tensors
from examples.qwen3_8b_layer_accuracy.modeling import Qwen3DecoderLayerHarness
from mdc_llm_deploy.onnx import OnnxAdapter
from mdc_llm_deploy.onnx.schemas import (
    FUSED_INFER_ATTENTION_SCORE_OP,
    RMS_NORM_OP,
    ROTARY_POSITION_EMBEDDING_OP,
)
from mdc_llm_deploy.quantization import MinMaxConfig, MinMaxLinear, quantize

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_SEQUENCE_LENGTH = 8


def _build_case() -> tuple[Qwen3DecoderLayerHarness, list[dict[str, torch.Tensor]]]:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        use_cache=False,
        dtype=torch.float16,
    )
    model = Qwen3ForCausalLM(config).eval().to(dtype=torch.float16)
    model.set_attn_implementation("eager")
    position_ids = torch.arange(_SEQUENCE_LENGTH).unsqueeze(0)
    attention_mask = torch.ones((1, _SEQUENCE_LENGTH), dtype=torch.long)
    cases: list[dict[str, torch.Tensor]] = []
    for scale in (0.75, 1.0, 1.25):
        hidden_states = (
            torch.randn(1, _SEQUENCE_LENGTH, config.hidden_size, dtype=torch.float16) * scale
        )
        cases.append(
            {
                "inputs_embeds": hidden_states,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
    return Qwen3DecoderLayerHarness.from_model(model), cases


def _w8a8_config(activation_granularity: str) -> MinMaxConfig:
    return MinMaxConfig(
        weight=True,
        activation=True,
        weight_granularity="per_channel",
        activation_granularity=activation_granularity,
        weight_symmetric=True,
        activation_symmetric=True,
    )


@pytest.mark.parametrize("activation_granularity", ["per_token", "per_tensor"])
def test_layer_w8a8_torch_and_mdc_export(activation_granularity: str) -> None:
    fp16_layer, cases = _build_case()
    quantized_layer = copy.deepcopy(fp16_layer)
    quantize(quantized_layer, _w8a8_config(activation_granularity), cases[:2])

    collector = SaturationCollector(quantized_layer)
    try:
        with torch.inference_mode():
            reference = fp16_layer(**cases[2])
            actual = quantized_layer(**cases[2])
    finally:
        collector.close()
    metrics = compare_tensors(reference, actual)
    assert metrics.finite
    assert metrics.cosine >= 0.99
    assert collector.report()["__all__"]["total"] > 0

    program = OnnxExporter().export(
        quantized_layer.model,
        {**cases[2], "use_cache": False},
        OnnxConfig(
            opset_version=21,
            optimize=False,
            dynamic=False,
            external_data=False,
        ),
    )
    graph = program.model_proto
    raw_counts = Counter(node.op_type for node in graph.graph.node)
    assert raw_counts["QuantizeLinear"] > 0
    assert raw_counts["QuantizeLinear"] == raw_counts["DequantizeLinear"]

    OnnxAdapter(replace(LAYER_ADAPTER_CONFIG, show_progress=False))(graph)
    lowered_counts = Counter(node.op_type for node in graph.graph.node)
    assert lowered_counts["QuantizeLinear"] == 0
    assert lowered_counts["DequantizeLinear"] == 0
    assert lowered_counts["NPUAscendQuantV2"] > 0
    assert lowered_counts["AscendDequant"] > 0
    assert lowered_counts[RMS_NORM_OP] == 4
    assert lowered_counts[ROTARY_POSITION_EMBEDDING_OP] == 1
    assert lowered_counts[FUSED_INFER_ATTENTION_SCORE_OP] == 0
    onnx.checker.check_model(graph)


def test_layer_per_token_rejects_different_sequence_length() -> None:
    layer, cases = _build_case()
    quantize(layer, _w8a8_config("per_token"), cases[:2])
    first_linear = next(module for module in layer.modules() if isinstance(module, MinMaxLinear))

    with pytest.raises(ValueError, match="per-token activation length changed"):
        first_linear(torch.randn(1, _SEQUENCE_LENGTH + 1, first_linear.in_features))
