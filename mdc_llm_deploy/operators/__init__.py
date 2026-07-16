"""Public MDC operators and their unique registration entry."""

from .contracts import (
    MDC_ONNX_DOMAIN,
    MDC_ONNX_OPSET,
    OPERATOR_SCHEMAS,
    OperatorSchema,
)
from .onnx import register_onnx_symbolics as _register_onnx_symbolics
from .runtime import (
    apply_rotary_pos_emb,
    ascend_dequant,
    ascend_quant_v2,
    fused_infer_attention_score,
    moe_expert,
    rms_norm,
)
from .torch import (
    OperatorBackendStatus,
    operator_backend_status,
    registered_device_dispatches,
)
from .torch import (
    register_torch_operators as _register_torch_operators,
)

_REGISTERED = False


def register_operators() -> None:
    """Register Torch kernels and ONNX symbolics exactly once."""
    global _REGISTERED
    if _REGISTERED:
        return
    _register_torch_operators()
    _register_onnx_symbolics()
    _REGISTERED = True


register_operators()

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
    "register_operators",
    "registered_device_dispatches",
    "rms_norm",
]
