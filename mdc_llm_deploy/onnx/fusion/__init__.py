"""Independent ONNX fusion passes and stable orchestration."""

from .apply_rotary_pos_emb import (
    APPLY_ROTARY_POS_EMB_FUSION_PASS,
    ApplyRotaryPosEmbFusionPass,
    fuse_apply_rotary_pos_emb,
)
from .contracts import FusionPass, FusionPassResult, FusionReport
from .fused_infer_attention_score import (
    FUSED_INFER_ATTENTION_SCORE_FUSION_PASS,
    FusedInferAttentionScoreFusionPass,
    fuse_fused_infer_attention_score,
)
from .rms_norm import RMS_NORM_FUSION_PASS, RmsNormFusionPass, fuse_rms_norm
from .runner import run_fusion_passes

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
