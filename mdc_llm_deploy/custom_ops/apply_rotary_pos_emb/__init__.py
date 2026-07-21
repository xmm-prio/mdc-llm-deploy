"""Independently loaded ApplyRotaryPosEmb operator plugin."""

from .contract import QUALIFIED_NAME, TORCH_SCHEMA
from .fake import fake
from .kernels import cpu, cuda
from .onnx import ONNX_NAME, ONNX_OPSET, validate_onnx_inputs
from .registration import PLUGIN, REGISTERED_OPERATOR, apply_rotary_pos_emb

__all__ = [
    "ONNX_NAME",
    "ONNX_OPSET",
    "PLUGIN",
    "QUALIFIED_NAME",
    "REGISTERED_OPERATOR",
    "TORCH_SCHEMA",
    "apply_rotary_pos_emb",
    "cpu",
    "cuda",
    "fake",
    "validate_onnx_inputs",
]
