"""Independently loaded RmsNorm custom-operator plugin."""

from .contract import MAX_TRITON_BLOCK_SIZE, SUPPORTED_DTYPES
from .fake import fake
from .kernels import cpu, cuda
from .onnx import ONNX_SCHEMA, translate, validate_onnx_inputs
from .registration import PLUGIN, rms_norm

__all__ = [
    "MAX_TRITON_BLOCK_SIZE",
    "ONNX_SCHEMA",
    "PLUGIN",
    "SUPPORTED_DTYPES",
    "cpu",
    "cuda",
    "fake",
    "rms_norm",
    "translate",
    "validate_onnx_inputs",
]
