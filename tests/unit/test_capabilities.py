from __future__ import annotations

import pytest

from mdc_llm_deploy.capabilities import (
    CAPABILITY_MATRIX,
    Algorithm,
    Artifact,
    MaskMode,
    ModelKind,
    Phase,
    Target,
    capability_for,
    gptq_bits_for,
    gptq_granularity_for,
    require_capability,
)
from mdc_llm_deploy.errors import UnsupportedPatternError


def test_release_matrix_has_8_fp16_and_20_minmax_combinations() -> None:
    fp16 = [item for item in CAPABILITY_MATRIX if item.algorithm is Algorithm.FP16]
    minmax = [item for item in CAPABILITY_MATRIX if item.algorithm is Algorithm.MINMAX]

    assert len(fp16) == 8
    assert len(minmax) == 20
    assert all(item.artifacts == {Artifact.FX, Artifact.ONNX, Artifact.ATC} for item in fp16)
    assert all(item.artifacts == {Artifact.FX, Artifact.ONNX, Artifact.ATC} for item in minmax)


def test_gptq_target_contract_is_centralized() -> None:
    assert gptq_bits_for(Target.LINEAR) == 4
    assert gptq_bits_for(Target.MOE) == 8
    assert gptq_granularity_for(Target.LINEAR) == "per_channel"
    assert gptq_granularity_for(Target.MOE) == "per_tensor"
    with pytest.raises(KeyError):
        gptq_bits_for(Target.ATTENTION)


def test_every_matrix_entry_has_one_canonical_lookup() -> None:
    for item in CAPABILITY_MATRIX:
        assert (
            capability_for(
                item.model,
                item.algorithm,
                item.target,
                item.phase,
                item.mask_mode,
            )
            is item
        )


def test_dense_rejects_moe_target() -> None:
    assert (
        capability_for(
            ModelKind.DENSE,
            Algorithm.MINMAX,
            Target.MOE,
            Phase.PREFILL,
            MaskMode.MASKED,
        )
        is None
    )

    with pytest.raises(UnsupportedPatternError, match="Unsupported capability"):
        require_capability(
            "dense",
            "minmax",
            "moe",
            "prefill",
            "masked",
            "fx",
        )


def test_gptq_is_explicitly_fx_only() -> None:
    capability = require_capability(
        "moe",
        "gptq",
        "moe",
        "decode",
        "maskless",
        "fx",
    )

    assert capability.artifacts == {Artifact.FX}
    with pytest.raises(UnsupportedPatternError, match="GPTQ is FX-only"):
        require_capability(
            "moe",
            "gptq",
            "moe",
            "decode",
            "maskless",
            "onnx",
        )


@pytest.mark.parametrize("mask_mode", tuple(MaskMode))
@pytest.mark.parametrize("phase", tuple(Phase))
@pytest.mark.parametrize(
    ("model", "target"),
    [
        (ModelKind.DENSE, Target.LINEAR),
        (ModelKind.DENSE, Target.ATTENTION),
        (ModelKind.MOE, Target.LINEAR),
        (ModelKind.MOE, Target.ATTENTION),
        (ModelKind.MOE, Target.MOE),
    ],
)
def test_minmax_matrix_covers_phase_and_mask_dimensions(
    model: ModelKind,
    target: Target,
    phase: Phase,
    mask_mode: MaskMode,
) -> None:
    assert require_capability(
        model,
        Algorithm.MINMAX,
        target,
        phase,
        mask_mode,
        Artifact.ATC,
    )
