"""Independently loaded FusedInferAttentionScore operator plugin."""

from .contract import PLUGIN_NAME, TORCH_INPUT_SLOTS
from .fake import fake_attention
from .kernels import attention_kernel
from .onnx import ONNX_ATTRIBUTE_NAMES, ONNX_OP_NAME
from .registration import PLUGIN, REGISTERED_OPERATOR, fused_infer_attention_score

__all__ = [
    "ONNX_ATTRIBUTE_NAMES",
    "ONNX_OP_NAME",
    "PLUGIN",
    "PLUGIN_NAME",
    "REGISTERED_OPERATOR",
    "TORCH_INPUT_SLOTS",
    "attention_kernel",
    "fake_attention",
    "fused_infer_attention_score",
]
