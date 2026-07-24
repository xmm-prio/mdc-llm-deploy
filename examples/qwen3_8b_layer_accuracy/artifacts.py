"""Generate reproducible Torch and ONNX artifacts for MDC validation."""

from __future__ import annotations

import gc
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import torch
from torch import Tensor, nn
from torch.onnx import ONNXProgram
from transformers.exporters import OnnxConfig, OnnxExporter

from mdc_llm_deploy.onnx import AdapterConfig, OnnxAdapter
from mdc_llm_deploy.quantization import MinMaxConfig, MinMaxLinear, quantize

from .metrics import AccuracyMetrics, SaturationCollector, compare_tensors
from .modeling import (
    CALIBRATION_PROMPTS,
    EVALUATION_PROMPTS,
    MODEL_ID,
    SEQUENCE_LENGTH,
    LayerSource,
    Qwen3DecoderLayerHarness,
    load_layer_source,
    move_inputs,
    prepare_prompt_inputs,
)

ValidationMode = Literal["fp16", "w8a8_per_token", "w8a8_per_tensor"]
VALIDATION_MODES: tuple[ValidationMode, ...] = (
    "fp16",
    "w8a8_per_token",
    "w8a8_per_tensor",
)
INPUT_NAMES = ("inputs_embeds", "attention_mask", "position_ids")
OUTPUT_NAME = "logits"
LAYER_ADAPTER_CONFIG = AdapterConfig(
    fuse_fused_infer_attention_score=False,
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Configure one complete Qwen3 layer artifact generation run."""

    model_id: str = MODEL_ID
    sequence_length: int = SEQUENCE_LENGTH
    cosine_threshold: float = 0.999
    process_graph: bool = True


def _quantization_config(mode: ValidationMode) -> MinMaxConfig | None:
    if mode == "fp16":
        return None
    activation_granularity = "per_token" if mode == "w8a8_per_token" else "per_tensor"
    return MinMaxConfig(
        weight=True,
        activation=True,
        weight_granularity="per_channel",
        activation_granularity=activation_granularity,
        weight_symmetric=True,
        activation_symmetric=True,
    )


def _prepare_module(
    source: LayerSource,
    mode: ValidationMode,
    calibration_inputs: list[dict[str, Tensor]],
    device: torch.device,
) -> nn.Module:
    module = source.clone_harness().to(device)
    quantization_config = _quantization_config(mode)
    if quantization_config is not None:
        quantize(module, quantization_config, calibration_inputs)
    return module.eval()


def _run_torch(
    module: nn.Module,
    evaluation_inputs: list[dict[str, Tensor]],
) -> tuple[list[Tensor], dict[str, dict[str, float | int]]]:
    collector = SaturationCollector(module)
    try:
        with torch.inference_mode():
            outputs = [module(**inputs).detach().cpu() for inputs in evaluation_inputs]
    finally:
        collector.close()
    return outputs, collector.report()


def _save_tensor(path: Path, tensor: Tensor) -> dict[str, object]:
    array = tensor.detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), array, allow_pickle=False)
    array.tofile(path.with_suffix(".bin"))
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _save_inputs(output_dir: Path, evaluation_inputs: list[dict[str, Tensor]]) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for case_index, inputs in enumerate(evaluation_inputs):
        case_dir = output_dir / "inputs" / f"case_{case_index}"
        tensors = {
            name: _save_tensor(case_dir / name, inputs[name])
            for name in INPUT_NAMES
        }
        cases.append(
            {
                "name": f"case_{case_index}",
                "prompt": EVALUATION_PROMPTS[case_index],
                "tensors": tensors,
            }
        )
    return cases


def _export_raw(
    module: Qwen3DecoderLayerHarness,
    inputs: dict[str, Tensor],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    output_path.with_suffix(".onnx.data").unlink(missing_ok=True)
    program = OnnxExporter().export(
        module.model,
        {**inputs, "use_cache": False},
        OnnxConfig(
            opset_version=21,
            optimize=False,
            dynamic=False,
            external_data=True,
        ),
    )
    if not isinstance(program, ONNXProgram):
        raise TypeError(f"Expected ONNXProgram, got {type(program).__name__}")
    program.save(output_path, external_data=True)


def _save_processed(raw_path: Path, processed_path: Path) -> None:
    graph = onnx.load(raw_path, load_external_data=True)
    OnnxAdapter(LAYER_ADAPTER_CONFIG)(graph)
    processed_path.unlink(missing_ok=True)
    data_path = processed_path.with_suffix(".onnx.data")
    data_path.unlink(missing_ok=True)
    onnx.save_model(
        graph,
        processed_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=0,
    )
    onnx.checker.check_model(processed_path)


def _quantization_summary(module: nn.Module) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for name, child in module.named_modules():
        if not isinstance(child, MinMaxLinear):
            continue
        item: dict[str, object] = {}
        for role in ("weight", "activation"):
            scale = getattr(child, f"{role}_scale")
            if scale is None:
                continue
            item[role] = {
                "shape": list(scale.shape),
                "minimum": float(scale.min().item()),
                "maximum": float(scale.max().item()),
                "axis": getattr(child, f"{role}_axis"),
            }
        summary[name] = item
    return summary


def _local_report(
    config: GenerationConfig,
    metrics: dict[str, list[AccuracyMetrics]],
) -> str:
    lines = [
        "# Qwen3-8B 单层本地精度结果",
        "",
        f"- 序列长度: {config.sequence_length}",
        "- 权重量化: 对称 INT8 per-channel",
        "- 激活量化: 对称 INT8 静态 per-token / per-tensor",
        f"- MDC 验收门槛: 同配置 Torch 对比 cosine >= {config.cosine_threshold}",
        "",
        "## 量化 Torch 相对 FP16 Torch",
    ]
    for mode in VALIDATION_MODES[1:]:
        lines.append("")
        lines.append(f"### {mode}")
        for index, result in enumerate(metrics[mode]):
            lines.append(
                f"- case_{index}: cosine={result.cosine:.9f}, "
                f"max_abs={result.max_absolute_error:.9g}, "
                f"mean_abs={result.mean_absolute_error:.9g}, "
                f"mean_rel={result.mean_relative_error:.9g}, finite={result.finite}"
            )
    lines.extend(
        [
            "",
            "## 说明",
            "- 本报告只描述量化模型相对 FP16 的模型误差, 不作为 MDC 同配置验收结果.",
            "- MDC 结果需使用 compare 子命令与各配置 torch_output 对比。",
            "",
        ]
    )
    return "\n".join(lines)


def generate_artifacts(
    output_dir: Path,
    config: GenerationConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Generate all three validation modes and a machine-readable manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading one Qwen3 layer from {config.model_id} on {device}")
    source = load_layer_source(config.model_id, device)
    calibration_inputs = prepare_prompt_inputs(
        source,
        CALIBRATION_PROMPTS,
        sequence_length=config.sequence_length,
        device=device,
    )
    evaluation_inputs = prepare_prompt_inputs(
        source,
        EVALUATION_PROMPTS,
        sequence_length=config.sequence_length,
        device=device,
    )
    input_cases = _save_inputs(
        output_dir,
        [move_inputs(inputs, "cpu") for inputs in evaluation_inputs],
    )

    fp16_outputs: list[Tensor] | None = None
    mode_metrics: dict[str, list[AccuracyMetrics]] = {}
    mode_manifest: dict[str, object] = {}
    for mode in VALIDATION_MODES:
        print(f"Generating {mode}")
        mode_dir = output_dir / mode
        if mode_dir.exists():
            shutil.rmtree(mode_dir)
        mode_dir.mkdir(parents=True)
        prepared_module = _prepare_module(source, mode, calibration_inputs, device)
        if not isinstance(prepared_module, Qwen3DecoderLayerHarness):
            raise TypeError("Expected Qwen3DecoderLayerHarness after preparation")
        module = prepared_module
        outputs, saturation = _run_torch(module, evaluation_inputs)
        output_specs = [
            _save_tensor(mode_dir / f"torch_output_case_{index}", output)
            for index, output in enumerate(outputs)
        ]
        if fp16_outputs is None:
            fp16_outputs = outputs
            mode_metrics[mode] = []
        else:
            mode_metrics[mode] = [
                compare_tensors(reference, actual)
                for reference, actual in zip(fp16_outputs, outputs, strict=True)
            ]

        quantization = _quantization_summary(module)
        module = module.cpu()
        export_inputs = move_inputs(evaluation_inputs[0], "cpu")
        raw_path = mode_dir / "layer_raw.onnx"
        _export_raw(module, export_inputs, raw_path)
        processed_path: Path | None = None
        if config.process_graph:
            processed_path = mode_dir / "layer_mdc.onnx"
            _save_processed(raw_path, processed_path)
        mode_manifest[mode] = {
            "raw_onnx": str(raw_path.relative_to(output_dir)),
            "mdc_onnx": (
                None if processed_path is None else str(processed_path.relative_to(output_dir))
            ),
            "torch_outputs": [
                {
                    "path": f"{mode}/torch_output_case_{index}.npy",
                    **output_spec,
                }
                for index, output_spec in enumerate(output_specs)
            ],
            "quantization": quantization,
            "saturation": saturation,
            "fp16_metrics": [metric.to_dict() for metric in mode_metrics[mode]],
        }
        del module, outputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "device": str(device),
        "input_names": list(INPUT_NAMES),
        "output_name": OUTPUT_NAME,
        "cases": input_cases,
        "modes": mode_manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "local_report.md").write_text(
        _local_report(config, mode_metrics),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "INPUT_NAMES",
    "LAYER_ADAPTER_CONFIG",
    "OUTPUT_NAME",
    "VALIDATION_MODES",
    "GenerationConfig",
    "ValidationMode",
    "generate_artifacts",
]
