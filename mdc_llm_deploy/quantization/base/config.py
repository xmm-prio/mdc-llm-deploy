"""Algorithm-independent quantization configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .selector import TargetSelector


@dataclass(frozen=True, slots=True, kw_only=True)
class QuantizationConfig:
    """Base configuration shared by quantization algorithms."""

    targets: TargetSelector = field(default_factory=TargetSelector)


__all__ = ["QuantizationConfig"]
