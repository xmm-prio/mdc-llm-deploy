"""Independent ONNX fusion passes and stable orchestration."""

from __future__ import annotations

from typing import Final

import onnx

from .apply_rotary_pos_emb import (
    APPLY_ROTARY_POS_EMB_FUSION_PASS,
    ApplyRotaryPosEmbFusionPass,
    fuse_apply_rotary_pos_emb,
)
from .base import FusionPass, FusionPassResult, FusionReport
from .fused_infer_attention_score import (
    FUSED_INFER_ATTENTION_SCORE_FUSION_PASS,
    FusedInferAttentionScoreFusionPass,
    fuse_fused_infer_attention_score,
)
from .rms_norm import RMS_NORM_FUSION_PASS, RmsNormFusionPass, fuse_rms_norm

_FUSION_PASSES: Final[tuple[FusionPass, ...]] = (
    RMS_NORM_FUSION_PASS,
    APPLY_ROTARY_POS_EMB_FUSION_PASS,
    FUSED_INFER_ATTENTION_SCORE_FUSION_PASS,
)


def run_fusion_passes(model: onnx.ModelProto) -> FusionReport:
    """Run every fusion pass in stable order, preserving prior successful rewrites."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an onnx.ModelProto")
    return FusionReport(tuple(fusion_pass.apply(model) for fusion_pass in _FUSION_PASSES))


__all__ = [
    "APPLY_ROTARY_POS_EMB_FUSION_PASS",
    "FUSED_INFER_ATTENTION_SCORE_FUSION_PASS",
    "RMS_NORM_FUSION_PASS",
    "ApplyRotaryPosEmbFusionPass",
    "FusedInferAttentionScoreFusionPass",
    "FusionPass",
    "FusionPassResult",
    "FusionReport",
    "RmsNormFusionPass",
    "fuse_apply_rotary_pos_emb",
    "fuse_fused_infer_attention_score",
    "fuse_rms_norm",
    "run_fusion_passes",
]
