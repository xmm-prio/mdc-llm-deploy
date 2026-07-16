"""Typed graph model-property contracts."""

import pytest

from mdc_llm_deploy.graph.metadata.model import (
    AttentionDimensions,
    MoeDimensions,
    NormalizationProperties,
)


def test_attention_dimensions_require_explicit_valid_values() -> None:
    dimensions = AttentionDimensions.from_properties(
        {
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
        }
    )

    assert dimensions == AttentionDimensions(4, 2, 64)


@pytest.mark.parametrize(
    "properties",
    [
        {},
        {
            "num_attention_heads": 4,
            "num_key_value_heads": 0,
            "head_dim": 64,
        },
        {
            "num_attention_heads": 3,
            "num_key_value_heads": 2,
            "head_dim": 64,
        },
    ],
)
def test_attention_dimensions_reject_missing_or_invalid_values(
    properties: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        AttentionDimensions.from_properties(properties)


def test_moe_dimensions_require_explicit_positive_values() -> None:
    dimensions = MoeDimensions.from_properties(
        {
            "hidden_size": 256,
            "moe_intermediate_size": 128,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_shared_experts": 1,
        }
    )

    assert dimensions == MoeDimensions(256, 128, 4, 2, 1)


@pytest.mark.parametrize(
    "field",
    [
        "hidden_size",
        "moe_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "num_shared_experts",
    ],
)
def test_moe_dimensions_reject_missing_values(field: str) -> None:
    properties = {
        "hidden_size": 256,
        "moe_intermediate_size": 128,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
    }
    del properties[field]

    with pytest.raises(ValueError, match=field):
        MoeDimensions.from_properties(properties)


def test_normalization_properties_are_optional_and_typed() -> None:
    assert NormalizationProperties.from_properties(
        {}
    ) == NormalizationProperties(None)
    assert NormalizationProperties.from_properties(
        {"rms_norm_epsilon": 1e-6}
    ) == NormalizationProperties(1e-6)


@pytest.mark.parametrize(
    "epsilon",
    [True, 0, -1e-6, "1e-6"],
)
def test_normalization_properties_reject_invalid_epsilon(
    epsilon: object,
) -> None:
    with pytest.raises(ValueError, match="rms_norm_epsilon"):
        NormalizationProperties.from_properties(
            {"rms_norm_epsilon": epsilon}
        )
