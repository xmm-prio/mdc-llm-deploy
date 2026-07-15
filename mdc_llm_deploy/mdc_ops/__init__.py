"""MDC custom operator capsules."""

from .operators import (
    apply_rotary_pos_emb,
    ascend_dequant,
    ascend_quant_v2,
    fused_infer_attention_score,
    moe_expert,
    registered_device_dispatches,
    rms_norm,
)
from .schema import MDC_ONNX_DOMAIN, MDC_ONNX_OPSET, OPERATOR_SCHEMAS, OperatorSchema
from .symbolics import register_onnx_symbolics

__all__ = [
    "MDC_ONNX_DOMAIN",
    "MDC_ONNX_OPSET",
    "OPERATOR_SCHEMAS",
    "OperatorSchema",
    "apply_rotary_pos_emb",
    "ascend_dequant",
    "ascend_quant_v2",
    "fused_infer_attention_score",
    "moe_expert",
    "register_onnx_symbolics",
    "registered_device_dispatches",
    "rms_norm",
]
