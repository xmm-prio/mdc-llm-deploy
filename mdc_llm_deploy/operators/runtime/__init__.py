"""Validated MDC operator runtime functions."""

from .attention import fused_infer_attention_score
from .moe import moe_expert
from .normalization import apply_rotary_pos_emb, rms_norm
from .quantized_io import ascend_dequant, ascend_quant_v2

__all__ = [
    "apply_rotary_pos_emb",
    "ascend_dequant",
    "ascend_quant_v2",
    "fused_infer_attention_score",
    "moe_expert",
    "rms_norm",
]
