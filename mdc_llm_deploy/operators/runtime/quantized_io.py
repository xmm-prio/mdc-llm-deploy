"""Ascend quantization and dequantization operator runtime."""
# mypy: disable-error-code="no-any-return,no-untyped-call"

from __future__ import annotations

import torch
from torch import Tensor

from .validation import (
    broadcastable,
    can_read_values,
    check_dtype,
    check_finite,
    check_no_autograd,
    check_rank,
    check_same_device,
    check_same_dtype,
)

_FLOAT_DTYPES = {torch.float16, torch.float32, torch.bfloat16}


def _axis_parameter(parameter: Tensor, x: Tensor, axis: int) -> Tensor:
    if parameter.ndim == 0:
        return parameter
    if parameter.numel() == 1:
        return parameter.reshape((1,) * x.ndim)
    if parameter.ndim == 1 and parameter.shape[0] == x.shape[axis]:
        shape = [1] * x.ndim
        shape[axis] = parameter.shape[0]
        return parameter.reshape(shape)
    return parameter


def ascend_quant_v2_reference(
    x: Tensor,
    scale: Tensor,
    offset: Tensor | None = None,
    axis: int = -1,
    dtype: int = 2,
) -> Tensor:
    del dtype
    normalized_axis = axis % x.ndim
    multiplier = _axis_parameter(scale, x, normalized_axis).float()
    zero_point: Tensor | float = (
        0.0
        if offset is None
        else _axis_parameter(offset, x, normalized_axis).float()
    )
    return (
        torch.round(x.float() * multiplier + zero_point)
        .clamp(-128, 127)
        .to(torch.int8)
    )


def ascend_quant_v2_meta(
    x: Tensor,
    scale: Tensor,
    offset: Tensor | None = None,
    axis: int = -1,
    dtype: int = 2,
) -> Tensor:
    del scale, offset, axis, dtype
    return torch.empty_like(x, dtype=torch.int8)


def _check_axis_parameter(
    name: str,
    parameter: Tensor,
    x: Tensor,
    axis: int,
) -> None:
    shaped = _axis_parameter(parameter, x, axis)
    if not broadcastable(shaped.shape, x.shape):
        raise ValueError(f"{name} does not match quantization axis")
    if parameter.numel() not in {1, x.shape[axis]}:
        raise ValueError(f"{name} does not match quantization axis")


def ascend_quant_v2(
    x: Tensor,
    scale: Tensor,
    offset: Tensor | None = None,
    *,
    axis: int = -1,
    dtype: int = 2,
) -> Tensor:
    """Apply multiplication-scale INT8 quantization with ties-to-even."""
    check_no_autograd(x, scale, offset)
    check_same_device("AscendQuantV2", x, scale, offset)
    check_same_dtype("AscendQuantV2", x, scale, offset)
    check_dtype("AscendQuantV2", x, _FLOAT_DTYPES)
    check_rank("AscendQuantV2 x", x, 1, 8)
    check_finite("AscendQuantV2", x, scale, offset)
    if dtype != 2:
        raise ValueError("0.1.0 only supports GE dtype=2 INT8")
    if axis < -x.ndim or axis >= x.ndim:
        raise ValueError("axis is outside input rank")
    normalized_axis = axis % x.ndim
    _check_axis_parameter("scale", scale, x, normalized_axis)
    if offset is not None:
        _check_axis_parameter("offset", offset, x, normalized_axis)
    if can_read_values(scale) and bool((scale <= 0).any()):
        raise ValueError("scale must be positive")
    return torch.ops.mdc_llm_deploy.ascend_quant_v2.default(
        x,
        scale,
        offset,
        axis,
        dtype,
    )


def _decode_dequant_scale(encoded: Tensor) -> Tensor:
    raw = encoded.to(torch.int64)
    if can_read_values(encoded) and bool((raw >> 32).ne(0).any()):
        raise ValueError("encoded scale high 32 bits must be zero")
    low_bits = (raw & 0xFFFFFFFF).to(torch.int32).contiguous()
    decoded = low_bits.view(torch.float32)
    if can_read_values(encoded) and not bool(torch.isfinite(decoded).all()):
        raise ValueError("encoded scale decodes to NaN or Inf")
    return decoded


def ascend_dequant_reference(
    x: Tensor,
    deq_scale: Tensor,
    sqrt_mode: bool = False,
    relu_flag: bool = False,
    dtype: int = 0,
) -> Tensor:
    scale = _decode_dequant_scale(deq_scale)
    if sqrt_mode:
        scale = torch.sqrt(scale) * torch.sqrt(scale)
    output = x.float() * scale
    if relu_flag:
        output = output.relu()
    return output.to(torch.float32 if dtype == 0 else torch.float16)


def ascend_dequant_meta(
    x: Tensor,
    deq_scale: Tensor,
    sqrt_mode: bool = False,
    relu_flag: bool = False,
    dtype: int = 0,
) -> Tensor:
    del deq_scale, sqrt_mode, relu_flag
    return torch.empty_like(
        x,
        dtype=torch.float32 if dtype == 0 else torch.float16,
    )


def ascend_dequant(
    x: Tensor,
    deq_scale: Tensor,
    *,
    sqrt_mode: bool = False,
    relu_flag: bool = False,
    dtype: int = 0,
) -> Tensor:
    """Decode restricted uint64 FP32 bits and dequantize INT32 input."""
    check_no_autograd(x, deq_scale)
    check_same_device("AscendDequant", x, deq_scale)
    if x.dtype != torch.int32:
        raise TypeError("AscendDequant x must use int32")
    if deq_scale.dtype != torch.uint64:
        raise TypeError("AscendDequant scale must use uint64")
    check_rank("AscendDequant x", x, 1, 8)
    if dtype not in {0, 1}:
        raise ValueError("dtype must be 0 or 1")
    if (
        deq_scale.ndim > 1
        or deq_scale.numel() not in {1, x.shape[-1]}
    ):
        raise ValueError(
            "deq_scale must be scalar or match output channels"
        )
    if can_read_values(deq_scale):
        _decode_dequant_scale(deq_scale)
    return torch.ops.mdc_llm_deploy.ascend_dequant.default(
        x,
        deq_scale,
        sqrt_mode,
        relu_flag,
        dtype,
    )
