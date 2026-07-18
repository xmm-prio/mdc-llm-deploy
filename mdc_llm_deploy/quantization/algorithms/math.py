"""Independent affine MinMax PTQ arithmetic."""

from __future__ import annotations

import torch
from torch import Tensor

from ..planning.types import (
    QuantizedTensor as QuantizedTensor,
)
from ..planning.types import (
    integer_range as integer_range,
)
from .dequant_scale import (
    decode_dequant_scale as decode_dequant_scale,
)
from .dequant_scale import (
    encode_dequant_scale as encode_dequant_scale,
)
from .gptq import (
    GPTQ_FALLBACK_CHOLESKY_FAILED as GPTQ_FALLBACK_CHOLESKY_FAILED,
)
from .gptq import (
    GPTQ_FALLBACK_NON_FINITE_HESSIAN as GPTQ_FALLBACK_NON_FINITE_HESSIAN,
)
from .gptq import (
    GptqFallbackError as GptqFallbackError,
)
from .gptq import (
    gptq_weight_quantize as gptq_weight_quantize,
)


def _qparams_from_extrema(
    minimum: Tensor,
    maximum: Tensor,
    *,
    bits: int,
    symmetric: bool,
) -> tuple[Tensor, Tensor]:
    """Calculate qparams from FP32 minimum and maximum tensors."""
    if not (
        torch.isfinite(minimum).all()
        and torch.isfinite(maximum).all()
    ):
        raise ValueError("tensor contains NaN or Inf")
    qmin, qmax = integer_range(bits)
    if symmetric:
        bound = torch.maximum(minimum.abs(), maximum.abs())
        scale = bound / qmax
        scale = torch.where(
            bound == 0,
            torch.ones_like(scale),
            scale,
        )
        zero_point = torch.zeros_like(
            scale,
            dtype=torch.int32,
        )
    else:
        minimum = torch.minimum(
            minimum,
            torch.zeros_like(minimum),
        )
        maximum = torch.maximum(
            maximum,
            torch.zeros_like(maximum),
        )
        span = maximum - minimum
        scale = span / (qmax - qmin)
        scale = torch.where(
            span == 0,
            torch.ones_like(scale),
            scale,
        )
        zero_point = torch.round(qmin - minimum / scale)
        zero_point = zero_point.clamp(
            qmin,
            qmax,
        ).to(torch.int32)
    return scale.float(), zero_point


def calculate_qparams(
    tensor: Tensor,
    *,
    bits: int,
    symmetric: bool,
    axis: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Calculate FP32 MinMax scale and int32 zero point."""
    if not tensor.is_floating_point():
        raise TypeError("tensor must use a floating dtype")
    source = tensor.detach().float()
    if not torch.isfinite(source).all():
        raise ValueError("tensor contains NaN or Inf")
    if axis is None:
        minimum = source.amin()
        maximum = source.amax()
    else:
        normalized_axis = axis % source.ndim
        reduction = tuple(
            index
            for index in range(source.ndim)
            if index != normalized_axis
        )
        minimum = source.amin(dim=reduction, keepdim=True)
        maximum = source.amax(dim=reduction, keepdim=True)
    return _qparams_from_extrema(
        minimum,
        maximum,
        bits=bits,
        symmetric=symmetric,
    )


def quantize(
    tensor: Tensor,
    *,
    bits: int,
    symmetric: bool,
    axis: int | None = None,
) -> QuantizedTensor:
    """Apply ties-to-even fake quantization."""
    scale, zero_point = calculate_qparams(
        tensor,
        bits=bits,
        symmetric=symmetric,
        axis=axis,
    )
    qmin, qmax = integer_range(bits)
    values = (
        torch.round(tensor.float() / scale)
        + zero_point.float()
    )
    values = values.clamp(qmin, qmax).to(torch.int8)
    dequantized = (
        (values.to(torch.int32) - zero_point).float()
        * scale
    ).to(tensor.dtype)
    return QuantizedTensor(
        values=values,
        dequantized=dequantized,
        scale=scale,
        zero_point=zero_point,
        bits=bits,
        symmetric=symmetric,
    )
