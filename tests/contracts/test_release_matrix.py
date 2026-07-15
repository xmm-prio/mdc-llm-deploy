"""Release-structure coverage for all 28 FP16/MinMax ONNX combinations."""

from __future__ import annotations

from mdc_llm_deploy.capabilities import (
    CAPABILITY_MATRIX,
    Algorithm,
    Artifact,
)
from tools.release_matrix import ATC_MOE_CONFIG

ONNX_MATRIX = tuple(
    item
    for item in CAPABILITY_MATRIX
    if item.algorithm in {Algorithm.FP16, Algorithm.MINMAX}
    and item.supports(Artifact.ONNX)
)


def test_release_onnx_matrix_has_exactly_28_unique_entries() -> None:
    identities = {
        (
            item.model,
            item.algorithm,
            item.target,
            item.phase,
            item.mask_mode,
        )
        for item in ONNX_MATRIX
    }

    assert len(ONNX_MATRIX) == 28
    assert len(identities) == 28
    assert sum(item.algorithm is Algorithm.FP16 for item in ONNX_MATRIX) == 8
    assert sum(item.algorithm is Algorithm.MINMAX for item in ONNX_MATRIX) == 20


def test_release_moe_fixture_uses_atc_verified_dimensions() -> None:
    assert ATC_MOE_CONFIG.hidden_size == 256
    assert ATC_MOE_CONFIG.moe_intermediate_size == 128
