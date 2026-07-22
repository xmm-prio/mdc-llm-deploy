from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import onnx
import torch
from torch import nn
from transformers.exporters import OnnxConfig

from mdc_llm_deploy.quantization import MinMaxConfig, quantize


@dataclass(frozen=True, slots=True)
class MinMaxExportCase:
    dtype: torch.dtype
    config: MinMaxConfig

    @property
    def id(self) -> str:
        dtype_name = str(self.dtype).removeprefix("torch.")
        sides = (
            "w8a8"
            if self.config.weight and self.config.activation
            else "w8"
            if self.config.weight
            else "a8"
        )
        weight = (
            f"w-{self.config.weight_granularity}-"
            f"{'sym' if self.config.weight_symmetric else 'asym'}"
            if self.config.weight
            else "w-off"
        )
        activation = (
            f"a-{self.config.activation_granularity}-"
            f"{'sym' if self.config.activation_symmetric else 'asym'}"
            if self.config.activation
            else "a-off"
        )
        return f"{dtype_name}-{sides}-{weight}-{activation}"

    @property
    def lowering_supported(self) -> bool:
        return (
            self.dtype in (torch.float16, torch.float32)
            and self.config.weight
            and self.config.activation
            and self.config.weight_symmetric
        )


def _supported_configs() -> tuple[MinMaxConfig, ...]:
    configs: list[MinMaxConfig] = []
    for weight, activation in ((True, False), (False, True), (True, True)):
        weight_options = (
            tuple(
                (granularity, symmetric)
                for granularity in ("per_tensor", "per_channel")
                for symmetric in (True, False)
            )
            if weight
            else (("per_tensor", True),)
        )
        activation_options = (
            tuple(
                (granularity, symmetric)
                for granularity in ("per_tensor", "per_token")
                for symmetric in (True, False)
            )
            if activation
            else (("per_tensor", True),)
        )
        for weight_granularity, weight_symmetric in weight_options:
            for activation_granularity, activation_symmetric in activation_options:
                configs.append(
                    MinMaxConfig(
                        weight=weight,
                        activation=activation,
                        weight_granularity=weight_granularity,  # type: ignore[arg-type]
                        activation_granularity=activation_granularity,  # type: ignore[arg-type]
                        weight_symmetric=weight_symmetric,
                        activation_symmetric=activation_symmetric,
                    )
                )
    return tuple(configs)


MINMAX_EXPORT_CASES = tuple(
    MinMaxExportCase(dtype, config)
    for dtype in (torch.float16, torch.bfloat16, torch.float32)
    for config in _supported_configs()
)
LOWERING_SUPPORTED_CASES = tuple(case for case in MINMAX_EXPORT_CASES if case.lowering_supported)
LOWERING_UNSUPPORTED_CASES = tuple(
    case for case in MINMAX_EXPORT_CASES if not case.lowering_supported
)


def quantized_onnx_export_config() -> OnnxConfig:
    return OnnxConfig(
        opset_version=21,
        optimize=False,
        dynamic=False,
        external_data=False,
    )


def build_quantized_linear(case: MinMaxExportCase) -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3, bias=False, dtype=case.dtype)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.linspace(0.1, 1.2, 12, dtype=case.dtype).reshape(3, 4))
    inputs = torch.linspace(-2, 2, 24, dtype=case.dtype).reshape(2, 3, 4)
    batches = [{"input": inputs}] if case.config.activation else ()
    quantize(model, case.config, batches)
    return model, inputs


def export_quantized_linear(case: MinMaxExportCase) -> onnx.ModelProto:
    return onnx.load_from_string(_export_quantized_linear_bytes(case))


@cache
def _export_quantized_linear_bytes(case: MinMaxExportCase) -> bytes:
    model, inputs = build_quantized_linear(case)
    export_config = quantized_onnx_export_config()
    program = torch.onnx.export(
        model,
        (inputs,),
        dynamo=True,
        opset_version=export_config.opset_version,
        external_data=export_config.external_data,
        optimize=export_config.optimize,
    )
    if program is None:
        raise RuntimeError("quantized ONNX export did not return an ONNXProgram")
    return program.model_proto.SerializeToString()


__all__ = [
    "LOWERING_SUPPORTED_CASES",
    "LOWERING_UNSUPPORTED_CASES",
    "MINMAX_EXPORT_CASES",
    "MinMaxExportCase",
    "build_quantized_linear",
    "export_quantized_linear",
    "quantized_onnx_export_config",
]
