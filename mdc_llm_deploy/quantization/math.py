"""Independent PTQ arithmetic shared by MinMax and GPTQ."""

from __future__ import annotations

from dataclasses import dataclass

import torch
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
    qmin, qmax = integer_range(bits)
    if axis is None:
        minimum = source.amin()
        maximum = source.amax()
    else:
        normalized_axis = axis % source.ndim
        reduction = tuple(index for index in range(source.ndim) if index != normalized_axis)
        minimum = source.amin(dim=reduction, keepdim=True)
        maximum = source.amax(dim=reduction, keepdim=True)
    if symmetric:
        bound = torch.maximum(minimum.abs(), maximum.abs())
        scale = bound / qmax
        scale = torch.where(bound == 0, torch.ones_like(scale), scale)
        zero_point = torch.zeros_like(scale, dtype=torch.int32)
    else:
        minimum = torch.minimum(minimum, torch.zeros_like(minimum))
        maximum = torch.maximum(maximum, torch.zeros_like(maximum))
        span = maximum - minimum
        scale = span / (qmax - qmin)
        scale = torch.where(span == 0, torch.ones_like(scale), scale)
        zero_point = torch.round(qmin - minimum / scale)
        zero_point = zero_point.clamp(qmin, qmax).to(torch.int32)
    return scale.float(), zero_point


def quantize(
    tensor: Tensor,
    *,
    bits: int,
    symmetric: bool,
    axis: int | None = None,
) -> QuantizedTensor:
    """Apply ties-to-even fake quantization."""
    scale, zero_point = calculate_qparams(
        tensor, bits=bits, symmetric=symmetric, axis=axis
    )
    qmin, qmax = integer_range(bits)
    values = torch.round(tensor.float() / scale) + zero_point.float()
    values = values.clamp(qmin, qmax).to(torch.int8)
    dequantized = (
        (values.to(torch.int32) - zero_point).float() * scale
    ).to(tensor.dtype)
    return QuantizedTensor(
        values=values,
        dequantized=dequantized,
        scale=scale,
        zero_point=zero_point,
        bits=bits,
        symmetric=symmetric,
    )


def encode_dequant_scale(scale: Tensor) -> Tensor:
    """Encode full FP32 bits in the low half of uint64 values."""
    source = scale.detach().contiguous().float()
    if not torch.isfinite(source).all():
        raise ValueError("scale contains NaN or Inf")
    if (source <= 0).any():
        raise ValueError("scale must be positive")
    low_bits = source.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return low_bits.to(torch.uint64)


def decode_dequant_scale(encoded: Tensor) -> Tensor:
    """Decode restricted AscendDequant uint64 scale representation."""
    if encoded.dtype != torch.uint64:
        raise TypeError("encoded scale must use uint64")
    raw = encoded.to(torch.int64)
    if ((raw >> 32) != 0).any():
        raise ValueError("encoded scale high 32 bits must be zero")
    decoded = (raw & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    if not torch.isfinite(decoded).all():
        raise ValueError("encoded scale decodes to NaN or Inf")
    return decoded


def gptq_weight_quantize(
    weight: Tensor,
    activations: Tensor,
    *,
    bits: int,
    percdamp: float = 0.01,
    actorder: bool = True,
    block_size: int = 128,
    per_channel: bool = True,
) -> QuantizedTensor:
    """Quantize a matrix with Hessian-aware sequential error compensation."""
    if weight.ndim != 2 or activations.ndim != 2:
        raise ValueError("weight and activations must be matrices")
    if activations.shape[1] != weight.shape[1]:
        raise ValueError("activation feature size must match weight input size")
    if block_size <= 0 or percdamp < 0:
        raise ValueError("block_size must be positive and percdamp non-negative")
    source = weight.detach().float()
    samples = activations.detach().float()
    if not torch.isfinite(source).all() or not torch.isfinite(samples).all():
        raise ValueError("GPTQ inputs contain NaN or Inf")
    hessian = 2.0 / max(samples.shape[0], 1) * samples.transpose(0, 1) @ samples
    diagonal = torch.diagonal(hessian)
    damp = percdamp * diagonal.mean()
    hessian = hessian + torch.eye(
        hessian.shape[0], device=hessian.device, dtype=hessian.dtype
    ) * damp
    chol = torch.linalg.cholesky(hessian)
    inverse = torch.cholesky_inverse(chol)
    order = (
        torch.argsort(diagonal, descending=True)
        if actorder
        else torch.arange(weight.shape[1], device=weight.device)
    )
    inverse_order = torch.argsort(order)
    work = source[:, order].clone()
    quantized = torch.empty_like(work, dtype=torch.int8)
    parameter_shape = (source.shape[0], 1) if per_channel else (1, 1)
    scales = torch.empty(parameter_shape, dtype=torch.float32, device=source.device)
    zero_points = torch.zeros(parameter_shape, dtype=torch.int32, device=source.device)
    _, qmax = integer_range(bits)
    bounds = (
        work.abs().amax(dim=1, keepdim=True)
        if per_channel
        else work.abs().amax().reshape(1, 1)
    )
    scales.copy_(torch.where(bounds == 0, torch.ones_like(bounds), bounds / qmax))
    qmin, qmax = integer_range(bits)
    for start in range(0, work.shape[1], block_size):
        end = min(start + block_size, work.shape[1])
        for column in range(start, end):
            current = work[:, column]
            q = torch.round(current / scales[:, 0]).clamp(qmin, qmax)
            quantized[:, column] = q.to(torch.int8)
            restored = q * scales[:, 0]
            pivot = inverse[order[column], order[column]].clamp_min(1e-12)
            error = (current - restored) / pivot
            if column + 1 < work.shape[1]:
                coupling = inverse[order[column], order[column + 1 :]]
                work[:, column + 1 :] -= error.unsqueeze(1) * coupling.unsqueeze(0)
    quantized = quantized[:, inverse_order]
    restored = quantized.float() * scales
    return QuantizedTensor(
        values=quantized,
        dequantized=restored.to(weight.dtype),
        scale=scales,
        zero_point=zero_points,
        bits=bits,
        symmetric=True,
    )
