from __future__ import annotations

from collections import Counter

import onnx
import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.exporters import OnnxConfig, OnnxExporter

from mdc_llm_deploy.onnx import process_onnx
from mdc_llm_deploy.onnx.schemas import (
    FUSED_INFER_ATTENTION_SCORE_OP,
    RMS_NORM_OP,
    ROTARY_POSITION_EMBEDDING_OP,
)
from mdc_llm_deploy.quantization import (
    MinMaxConfig,
    MinMaxLinear,
    QuantizationState,
    calibrate,
    convert,
    load_quantized_state_dict,
    prepare,
    quantization_state,
    quantize,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_INPUTS = {
    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
    "attention_mask": torch.ones((1, 3), dtype=torch.long),
    "position_ids": torch.arange(3, dtype=torch.long).unsqueeze(0),
    "use_cache": False,
}


@pytest.fixture(autouse=True)
def offline_export_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _build_dense_qwen3(
    *,
    dtype: torch.dtype = torch.float32,
    use_cache: bool = False,
) -> Qwen3ForCausalLM:
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
        use_cache=use_cache,
        pad_token_id=0,
        eos_token_id=31,
        dtype=dtype,
    )
    model = Qwen3ForCausalLM(config)
    model.set_attn_implementation("eager")
    return model.eval().to(dtype=dtype)


def _w8a8_config() -> MinMaxConfig:
    return MinMaxConfig(
        weight=True,
        activation=True,
        weight_granularity="per_channel",
        activation_granularity="per_tensor",
        weight_symmetric=True,
        activation_symmetric=True,
    )


def _calibration_batches() -> list[dict[str, object]]:
    return [dict(_INPUTS)]


def _assert_finite_inference(model: nn.Module) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(**_INPUTS).logits
    assert logits.shape == (1, 3, 32)
    assert bool(torch.isfinite(logits).all())
    return logits


def _export_config() -> OnnxConfig:
    return OnnxConfig(
        opset_version=21,
        optimize=False,
        dynamic=False,
        external_data=False,
    )


def _export_with_pytorch(model: nn.Module) -> onnx.ModelProto:
    config = _export_config()
    program = torch.onnx.export(
        model,
        (),
        kwargs=dict(_INPUTS),
        dynamo=True,
        opset_version=config.opset_version,
        external_data=config.external_data,
        optimize=config.optimize,
    )
    if program is None:
        raise RuntimeError("PyTorch ONNX export did not return an ONNXProgram")
    return program.model_proto


def _export_with_transformers(model: nn.Module) -> onnx.ModelProto:
    return OnnxExporter().export(model, dict(_INPUTS), _export_config()).model_proto


def _standard_operator_counts(model: onnx.ModelProto) -> Counter[str]:
    return Counter(node.op_type for node in model.graph.node if node.domain in ("", "ai.onnx"))


def _computational_operator_counts(model: onnx.ModelProto) -> Counter[str]:
    counts = _standard_operator_counts(model)
    for housekeeping_operator in ("Constant", "Identity"):
        del counts[housekeeping_operator]
    return counts


def test_dense_qwen3_three_stage_and_one_step_inference() -> None:
    staged = _build_dense_qwen3()
    config = _w8a8_config()

    assert prepare(staged, config) is staged
    assert quantization_state(staged) is QuantizationState.PREPARED
    assert calibrate(staged, _calibration_batches()) is staged
    assert quantization_state(staged) is QuantizationState.CALIBRATED
    assert convert(staged) is staged
    assert quantization_state(staged) is QuantizationState.CONVERTED
    assert any(isinstance(module, MinMaxLinear) for module in staged.modules())
    _assert_finite_inference(staged)

    one_step = _build_dense_qwen3()
    assert quantize(one_step, config, _calibration_batches()) is one_step
    assert quantization_state(one_step) is QuantizationState.CONVERTED
    _assert_finite_inference(one_step)


def test_dense_qwen3_quantized_state_dict_restores_frozen_model() -> None:
    config = _w8a8_config()
    source = _build_dense_qwen3()
    quantize(source, config, _calibration_batches())
    expected = _assert_finite_inference(source)
    checkpoint = {name: value.detach().clone() for name, value in source.state_dict().items()}

    restored = _build_dense_qwen3()
    assert load_quantized_state_dict(restored, config, checkpoint) is restored
    assert quantization_state(restored) is QuantizationState.CONVERTED
    actual = _assert_finite_inference(restored)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not module.training for module in restored.modules())


def test_dense_qwen3_opset21_qdq_export_and_w8a8_lowering() -> None:
    model = _build_dense_qwen3()
    quantize(model, _w8a8_config(), _calibration_batches())

    exports = {
        "pytorch": _export_with_pytorch(model),
        "transformers": _export_with_transformers(model),
    }
    raw_operator_counts = {
        exporter_name: _standard_operator_counts(exported)
        for exporter_name, exported in exports.items()
    }
    assert _computational_operator_counts(exports["pytorch"]) == _computational_operator_counts(
        exports["transformers"]
    )

    for exporter_name, exported in exports.items():
        onnx.checker.check_model(exported)
        assert next(opset.version for opset in exported.opset_import if opset.domain == "") == 21
        raw_counts = raw_operator_counts[exporter_name]
        assert raw_counts["QuantizeLinear"] == raw_counts["DequantizeLinear"]
        assert raw_counts["QuantizeLinear"] > 0

        assert process_onnx(exported) is exported
        lowered_counts = _standard_operator_counts(exported)
        assert lowered_counts["QuantizeLinear"] == 0
        assert lowered_counts["DequantizeLinear"] == 0
        assert sum(node.op_type == "NPUAscendQuantV2" for node in exported.graph.node) > 0
        assert sum(node.op_type == "AscendDequant" for node in exported.graph.node) > 0
        onnx.checker.check_model(exported)


def test_dense_qwen3_w8a8_generation_fuses_fp16_attention() -> None:
    model = _build_dense_qwen3(dtype=torch.float16, use_cache=True)
    quantize(model, _w8a8_config(), _calibration_batches())
    programs = OnnxExporter().export_for_generation(
        model,
        {"inputs": _INPUTS["input_ids"]},
        _export_config(),
    )

    for component_name in ("prefill", "decode"):
        exported = programs[component_name].model_proto

        assert process_onnx(exported) is exported

        counts = _standard_operator_counts(exported)
        assert counts["QuantizeLinear"] == 0
        assert counts["DequantizeLinear"] == 0
        assert counts["NPUAscendQuantV2"] > 0
        assert counts["AscendDequant"] > 0
        assert counts[RMS_NORM_OP] == 5
        assert counts[ROTARY_POSITION_EMBEDDING_OP] == 1
        assert counts[FUSED_INFER_ATTENTION_SCORE_OP] == 1
        onnx.checker.check_model(exported)
