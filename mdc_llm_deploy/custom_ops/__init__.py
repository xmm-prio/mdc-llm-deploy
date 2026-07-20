"""Public API for incrementally attached custom operators."""

from typing import Any

from .apply_rotary_pos_emb import ApplyRotaryPosEmb
from .base import CustomOp
from .fused_infer_attention_score import FusedInferAttentionScore
from .moe_expert import MoeExpert
from .registry import (
    RegisteredCustomOp,
    get_custom_op,
    register_custom_op,
    register_custom_ops,
    registered_custom_ops,
)
from .rms_norm import RmsNorm

_RMS_NORM, _APPLY_ROTARY_POS_EMB, _FUSED_INFER_ATTENTION_SCORE, _MOE_EXPERT = (
    register_custom_ops(
        RmsNorm,
        ApplyRotaryPosEmb,
        FusedInferAttentionScore,
        MoeExpert,
    )
)


def rms_norm(*args: Any, **kwargs: Any) -> Any:
    """Invoke the registered RMS normalization operator."""
    return _RMS_NORM.definition(*args, **kwargs)


def apply_rotary_pos_emb(*args: Any, **kwargs: Any) -> Any:
    """Invoke the registered rotary position embedding operator."""
    return _APPLY_ROTARY_POS_EMB.definition(*args, **kwargs)


def fused_infer_attention_score(*args: Any, **kwargs: Any) -> Any:
    """Invoke the registered fused attention operator."""
    return _FUSED_INFER_ATTENTION_SCORE.definition(*args, **kwargs)


def moe_expert(*args: Any, **kwargs: Any) -> Any:
    """Invoke the registered mixture-of-experts operator."""
    return _MOE_EXPERT.definition(*args, **kwargs)

__all__ = [
    "ApplyRotaryPosEmb",
    "CustomOp",
    "FusedInferAttentionScore",
    "MoeExpert",
    "RegisteredCustomOp",
    "RmsNorm",
    "apply_rotary_pos_emb",
    "fused_infer_attention_score",
    "get_custom_op",
    "moe_expert",
    "register_custom_op",
    "register_custom_ops",
    "registered_custom_ops",
    "rms_norm",
]
