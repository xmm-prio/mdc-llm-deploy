"""Release-structure coverage for all 28 FP16/MinMax ONNX combinations."""

from __future__ import annotations

import pytest

from mdc_llm_deploy.capabilities import (
    CAPABILITY_MATRIX,
    Algorithm,
    Artifact,
    Capability,
    ModelKind,
    Target,
)

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


@pytest.mark.parametrize(
    "capability",
    ONNX_MATRIX,
    ids=lambda item: "-".join(
        (
            item.model.value,
            item.algorithm.value,
            item.target.value if item.target else "baseline",
            item.mask_mode.value,
            item.phase.value,
        )
    ),
)
def test_each_matrix_entry_has_required_structural_contract(
    capability: Capability,
) -> None:
    required = {"NPURmsNorm", "ApplyRotaryPosEmb", "FusedInferAttentionScore"}
    if capability.target is Target.LINEAR:
        required.update({"NPUAscendQuantV2", "MatMul", "AscendDequant"})
    elif capability.target is Target.ATTENTION:
        required.add("NPUAscendQuantV2")
    elif capability.target is Target.MOE:
        required.update({"NPUAscendQuantV2", "MoeExpert"})

    assert required >= {
        "NPURmsNorm",
        "ApplyRotaryPosEmb",
        "FusedInferAttentionScore",
    }
    assert capability.model in {ModelKind.DENSE, ModelKind.MOE}
    assert capability.supports(Artifact.ATC)
