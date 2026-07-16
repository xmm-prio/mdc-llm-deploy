"""MDC custom operator capsules."""

from ..onnx_protocol import MDC_ONNX_DOMAIN, MDC_ONNX_OPSET
from ..operator_schema import OPERATOR_SCHEMAS, OperatorSchema
from .backend import OperatorBackendStatus
from .operators import (
    apply_rotary_pos_emb,
    ascend_dequant,
    ascend_quant_v2,
    fused_infer_attention_score,
    moe_expert,
    operator_backend_status,
    registered_device_dispatches,
    rms_norm,
)
from .symbolics import register_onnx_symbolics

__all__ = [
    "MDC_ONNX_DOMAIN",
    "MDC_ONNX_OPSET",
    "OPERATOR_SCHEMAS",
    "OperatorBackendStatus",
    "OperatorSchema",
    "apply_rotary_pos_emb",
    "ascend_dequant",
    "ascend_quant_v2",
    "fused_infer_attention_score",
    "moe_expert",
    "operator_backend_status",
    "register_onnx_symbolics",
    "registered_device_dispatches",
    "rms_norm",
]
