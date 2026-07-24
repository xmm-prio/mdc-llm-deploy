"""Streaming float32 MinMax statistics and INT8 parameter calculation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class QuantizationParameters:
    """Frozen INT8 quantization parameters."""

    scale: Tensor
    zero_point: Tensor | None


class MinMaxObserver:
    """Collect finite tensor ranges in float32 without retaining samples."""

    def __init__(self, *, axis: int | None = None) -> None:
        self._axis = axis
        self._minimum: Tensor | None = None
        self._maximum: Tensor | None = None
        self._rank: int | None = None
        self._axis_size: int | None = None

    @property
    def observed(self) -> bool:
        """Return whether at least one tensor was observed."""
        return self._minimum is not None

    def observe(self, value: Tensor) -> None:
        """Update streaming ranges from one floating-point tensor."""
        if not value.is_floating_point():
            raise TypeError("observed tensor must use a floating-point dtype")
        if value.numel() == 0:
            raise ValueError("observed tensor must not be empty")
        float_value = value.detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(float_value).all()):
            raise ValueError("observed tensor must contain only finite values")

        rank = float_value.ndim
        if self._axis is None:
            minimum = float_value.amin()
            maximum = float_value.amax()
        else:
            axis = self._axis % rank
            axis_size = float_value.shape[axis]
            if self._rank is not None and rank != self._rank:
                raise ValueError(
                    f"observed tensor rank changed from {self._rank} to {rank}"
                )
            if self._axis_size is not None and axis_size != self._axis_size:
                raise ValueError(
                    f"observed axis length changed from {self._axis_size} to {axis_size}"
                )
            reduced_dimensions = tuple(index for index in range(rank) if index != axis)
            minimum = float_value.amin(dim=reduced_dimensions, keepdim=True)
            maximum = float_value.amax(dim=reduced_dimensions, keepdim=True)
            self._rank = rank
            self._axis_size = axis_size

        self._minimum = minimum if self._minimum is None else torch.minimum(self._minimum, minimum)
        self._maximum = maximum if self._maximum is None else torch.maximum(self._maximum, maximum)

    def calculate_qparams(self, *, symmetric: bool) -> QuantizationParameters:
        """Freeze finite positive scale and optional INT8 zero-point."""
        if self._minimum is None or self._maximum is None:
            raise RuntimeError("observer has not collected any values")
        minimum = self._minimum
        maximum = self._maximum
        if symmetric:
            largest = torch.maximum(minimum.abs(), maximum.abs())
            scale = torch.where(largest == 0, torch.ones_like(largest), largest / 127.0)
            zero_point = None
        else:
            zero_range = maximum == minimum
            adjusted_minimum = torch.minimum(minimum, torch.zeros_like(minimum))
            adjusted_maximum = torch.maximum(maximum, torch.zeros_like(maximum))
            scale = torch.where(
                zero_range,
                torch.ones_like(minimum),
                (adjusted_maximum - adjusted_minimum) / 255.0,
            )
            projected = torch.round(-128.0 - adjusted_minimum / scale).clamp(-128, 127)
            zero_point = torch.where(zero_range, torch.zeros_like(projected), projected).to(
                dtype=torch.int8
            )
        _validate_scale(scale)
        return QuantizationParameters(scale=scale, zero_point=zero_point)


def _validate_scale(scale: Tensor) -> None:
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("quantization scale must be finite and strictly positive")


__all__ = ["MinMaxObserver", "QuantizationParameters"]
