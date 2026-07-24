from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import product

import pytest

from mdc_llm_deploy.onnx import AdapterConfig, OnnxAdapter


def test_adapter_config_defaults_enable_complete_pipeline() -> None:
    config = AdapterConfig()

    assert config.fold_constants
    assert config.fuse_rms_norm
    assert config.fuse_apply_rotary_pos_emb
    assert config.fuse_fused_infer_attention_score
    assert config.show_progress


def test_adapter_config_is_immutable() -> None:
    config = AdapterConfig()

    with pytest.raises(FrozenInstanceError):
        config.show_progress = False  # type: ignore[misc]


@pytest.mark.parametrize(
    (
        "fuse_rms_norm",
        "fuse_apply_rotary_pos_emb",
        "fuse_fused_infer_attention_score",
    ),
    product((False, True), repeat=3),
)
def test_adapter_selects_enabled_fusion_passes_in_stable_order(
    fuse_rms_norm: bool,
    fuse_apply_rotary_pos_emb: bool,
    fuse_fused_infer_attention_score: bool,
) -> None:
    config = AdapterConfig(
        fuse_rms_norm=fuse_rms_norm,
        fuse_apply_rotary_pos_emb=fuse_apply_rotary_pos_emb,
        fuse_fused_infer_attention_score=fuse_fused_infer_attention_score,
    )

    selected = OnnxAdapter(config)._selected_fusion_passes()

    expected = tuple(
        name
        for enabled, name in (
            (fuse_rms_norm, "rms_norm"),
            (fuse_apply_rotary_pos_emb, "apply_rotary_pos_emb"),
            (fuse_fused_infer_attention_score, "fused_infer_attention_score"),
        )
        if enabled
    )
    assert tuple(fusion_pass.name for fusion_pass in selected) == expected


def test_adapter_exposes_its_configuration() -> None:
    config = AdapterConfig(show_progress=False)

    assert OnnxAdapter(config).config is config
