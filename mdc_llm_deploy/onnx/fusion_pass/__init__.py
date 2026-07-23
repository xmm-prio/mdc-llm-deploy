"""Independent ONNX fusion passes and stable orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Final

import onnx

from ..._observability import get_logger
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

_logger = get_logger(__name__)
_FUSION_PASSES: Final[tuple[FusionPass, ...]] = (
    RMS_NORM_FUSION_PASS,
    APPLY_ROTARY_POS_EMB_FUSION_PASS,
    FUSED_INFER_ATTENTION_SCORE_FUSION_PASS,
)


def run_fusion_passes(
    model: onnx.ModelProto,
    *,
    passes: Sequence[FusionPass] | None = None,
) -> FusionReport:
    """Run selected fusion passes in order, or every pass when unspecified."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an onnx.ModelProto")
    selected_passes = _FUSION_PASSES if passes is None else tuple(passes)
    results: list[FusionPassResult] = []
    for fusion_pass in selected_passes:
        started_at = perf_counter()
        result = fusion_pass.apply(model)
        results.append(result)
        _logger.info(
            "Fusion pass %s completed in %.3fs: fused_count=%d",
            result.pass_name,
            perf_counter() - started_at,
            result.fused_count,
        )
        _logger.debug(
            "Fusion pass %s fused nodes: %s",
            result.pass_name,
            result.fused_node_names,
        )
    report = FusionReport(tuple(results))
    _logger.info("Fusion passes completed: total_fused_count=%d", report.total_fused_count)
    return report


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
