"""MinMax quantization configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from ...lifecycle.config import QuantizationConfig

WeightGranularity: TypeAlias = Literal["per_tensor", "per_channel"]
ActivationGranularity: TypeAlias = Literal["per_tensor", "per_token"]


@dataclass(frozen=True, slots=True, kw_only=True)
class MinMaxConfig(QuantizationConfig):
    """Configure eager INT8 MinMax fake quantization."""

    weight: bool = True
    activation: bool = False
    weight_granularity: WeightGranularity = "per_tensor"
    activation_granularity: ActivationGranularity = "per_tensor"
    weight_symmetric: bool = True
    activation_symmetric: bool = True

    def __post_init__(self) -> None:
        if not self.weight and not self.activation:
            raise ValueError("weight and activation quantization cannot both be disabled")
        if self.weight_granularity not in ("per_tensor", "per_channel"):
            raise ValueError(f"unsupported weight granularity: {self.weight_granularity!r}")
        if self.activation_granularity not in ("per_tensor", "per_token"):
            raise ValueError(
                f"unsupported activation granularity: {self.activation_granularity!r}"
            )


__all__ = ["ActivationGranularity", "MinMaxConfig", "WeightGranularity"]
