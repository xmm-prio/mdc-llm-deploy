"""Shared value types and integer-domain contracts for PTQ."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class QuantizedTensor:
    """Quantized integer values and affine parameters."""

    values: Tensor
    dequantized: Tensor
    scale: Tensor
    zero_point: Tensor
    bits: int
    symmetric: bool


def integer_range(bits: int) -> tuple[int, int]:
    """Return signed integer range for a supported bit width."""
    if bits not in {4, 8}:
        raise ValueError("bits must be 4 or 8")
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
