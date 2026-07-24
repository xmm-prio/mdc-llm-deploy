"""Extensible in-place quantization API."""

from .algorithms.minmax import MinMaxConfig, MinMaxLinear, MinMaxQuantizer
from .lifecycle import (
    CalibrationBatch,
    QuantizationConfig,
    QuantizationState,
    Quantizer,
    TargetSelector,
)
from .lifecycle.api import (
    calibrate,
    convert,
    load_quantized_state_dict,
    prepare,
    quantization_state,
    quantize,
)

__all__ = [
    "CalibrationBatch",
    "MinMaxConfig",
    "MinMaxLinear",
    "MinMaxQuantizer",
    "QuantizationConfig",
    "QuantizationState",
    "Quantizer",
    "TargetSelector",
    "calibrate",
    "convert",
    "load_quantized_state_dict",
    "prepare",
    "quantization_state",
    "quantize",
]
