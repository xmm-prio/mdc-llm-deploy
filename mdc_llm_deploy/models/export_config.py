"""Configuration contract for export-specialized models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MaskMode = Literal["causal", "none"]


@dataclass(frozen=True, slots=True)
class ExportModelConfig:
    """Freeze sequence-dependent export semantics during model construction."""

    sequence_length: int
    mask_mode: MaskMode = "causal"

    def __post_init__(self) -> None:
        if type(self.sequence_length) is not int or self.sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")
        if self.mask_mode not in {"causal", "none"}:
            raise ValueError("mask_mode must be 'causal' or 'none'")


__all__ = ["ExportModelConfig", "MaskMode"]
