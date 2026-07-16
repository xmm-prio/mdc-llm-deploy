"""Release coverage without a redundant mask dimension."""

from __future__ import annotations

from mdc_llm_deploy.capabilities import (
    CAPABILITY_MATRIX,
    Algorithm,
    Artifact,
)
from tools.release.matrix import ATC_MOE_CONFIG

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
        )
        for item in ONNX_MATRIX
    }

    assert len(identities) == 14


def test_release_moe_fixture_uses_atc_verified_dimensions() -> None:
    assert ATC_MOE_CONFIG.hidden_size == 256
    assert ATC_MOE_CONFIG.moe_intermediate_size == 128
