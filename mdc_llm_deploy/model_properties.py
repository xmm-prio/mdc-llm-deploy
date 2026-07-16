"""Typed readers for model semantics stored in graph properties."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _positive_int(
    properties: Mapping[str, Any],
    name: str,
) -> int:
    value = properties.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"Model property {name!r} must be a positive integer"
        )
    return value


def _nonnegative_int(
    properties: Mapping[str, Any],
    name: str,
) -> int:
    value = properties.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(
            f"Model property {name!r} must be a non-negative integer"
        )
    return value


@dataclass(frozen=True, slots=True)
class AttentionDimensions:
    """Validated grouped-query attention dimensions."""

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @classmethod
    def from_properties(
        cls,
        properties: Mapping[str, Any],
    ) -> AttentionDimensions:
        """Parse dimensions without applying architecture defaults."""
        result = cls(
            num_attention_heads=_positive_int(
                properties,
                "num_attention_heads",
            ),
            num_key_value_heads=_positive_int(
                properties,
                "num_key_value_heads",
            ),
            head_dim=_positive_int(properties, "head_dim"),
        )
        if (
            result.num_attention_heads
            % result.num_key_value_heads
        ):
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads"
            )
        return result


@dataclass(frozen=True, slots=True)
class MoeDimensions:
    """Validated model dimensions required by Tiny MoE lowering."""

    hidden_size: int
    intermediate_size: int
    routed_expert_count: int
    routed_top_k: int
    shared_expert_count: int

    @classmethod
    def from_properties(
        cls,
        properties: Mapping[str, Any],
    ) -> MoeDimensions:
        """Parse MoE dimensions without architecture defaults."""
        return cls(
            hidden_size=_positive_int(
                properties,
                "hidden_size",
            ),
            intermediate_size=_positive_int(
                properties,
                "moe_intermediate_size",
            ),
            routed_expert_count=_positive_int(
                properties,
                "num_experts",
            ),
            routed_top_k=_positive_int(
                properties,
                "num_experts_per_tok",
            ),
            shared_expert_count=_nonnegative_int(
                properties,
                "num_shared_experts",
            ),
        )


@dataclass(frozen=True, slots=True)
class NormalizationProperties:
    """Validated model normalization semantics."""

    rms_norm_epsilon: float | None

    @classmethod
    def from_properties(
        cls,
        properties: Mapping[str, Any],
    ) -> NormalizationProperties:
        """Parse optional normalization properties."""
        raw_epsilon = properties.get("rms_norm_epsilon")
        if raw_epsilon is None:
            return cls(rms_norm_epsilon=None)
        if (
            isinstance(raw_epsilon, bool)
            or not isinstance(raw_epsilon, (int, float))
            or raw_epsilon <= 0
        ):
            raise ValueError(
                "Model property 'rms_norm_epsilon' must be a positive number"
            )
        return cls(rms_norm_epsilon=float(raw_epsilon))


__all__ = [
    "AttentionDimensions",
    "MoeDimensions",
    "NormalizationProperties",
]
