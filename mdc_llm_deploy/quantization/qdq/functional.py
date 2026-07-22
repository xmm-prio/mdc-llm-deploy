"""Exporter-independent QDQ fake-quantization API."""

from __future__ import annotations

import torch
from torch import Tensor

from ._registration import register_qdq_operator


def qdq(
    inputs: Tensor,
    scale: Tensor,
    zero_point: Tensor | None = None,
    *,
    axis: int | None = None,
) -> Tensor:
    """Apply frozen INT8 quantize-dequantize semantics."""
    if not inputs.is_floating_point():
        raise TypeError("QDQ input must use a floating-point dtype")
    if not scale.is_floating_point():
        raise TypeError("QDQ scale must use a floating-point dtype")
    if zero_point is not None and zero_point.dtype is not torch.int8:
        raise TypeError("QDQ zero-point must use torch.int8")
    if axis is not None and not -inputs.ndim <= axis < inputs.ndim:
        raise ValueError(f"QDQ axis {axis} is invalid for rank-{inputs.ndim} input")
    return register_qdq_operator()(inputs, scale, zero_point, axis)


__all__ = ["qdq"]
