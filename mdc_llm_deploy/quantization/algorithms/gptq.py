"""Hessian-aware GPTQ weight quantization."""

from __future__ import annotations

import torch
from torch import Tensor

from ..planning.types import QuantizedTensor, integer_range

GPTQ_FALLBACK_NON_FINITE_HESSIAN = "non_finite_hessian"
GPTQ_FALLBACK_CHOLESKY_FAILED = "cholesky_failed"


class GptqFallbackError(RuntimeError):
    """Signal a PRD-approved GPTQ fallback condition."""

    def __init__(self, reason: str) -> None:
        if reason not in {
            GPTQ_FALLBACK_NON_FINITE_HESSIAN,
            GPTQ_FALLBACK_CHOLESKY_FAILED,
        }:
            raise ValueError(
                f"Unsupported GPTQ fallback reason: {reason}"
            )
        super().__init__(reason)
        self.reason = reason


def _clip_scales(
    source: Tensor,
    *,
    bits: int,
    per_channel: bool,
) -> Tensor:
    """Select scales by deterministic 20-point clipping error search."""
    _, qmax = integer_range(bits)
    base_bounds = (
        source.abs().amax(dim=1, keepdim=True)
        if per_channel
        else source.abs().amax().reshape(1, 1)
    )
    ratios = torch.tensor(
        [0.5 + index * 0.5 / 19 for index in range(20)],
        dtype=torch.float32,
        device=source.device,
    )
    best_error = torch.full_like(base_bounds, torch.inf)
    best_scale = torch.ones_like(base_bounds)
    qmin, qmax = integer_range(bits)
    for ratio in ratios:
        bounds = base_bounds * ratio
        scales = torch.where(
            bounds == 0,
            torch.ones_like(bounds),
            bounds / qmax,
        )
        restored = (
            torch.round(source / scales)
            .clamp(qmin, qmax)
            .mul(scales)
        )
        squared_error = (source - restored).square()
        error = (
            squared_error.mean(dim=1, keepdim=True)
            if per_channel
            else squared_error.mean().reshape(1, 1)
        )
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_scale = torch.where(improved, scales, best_scale)
    return best_scale


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
    """Quantize a matrix with Hessian-aware error compensation."""
    if weight.ndim != 2 or activations.ndim != 2:
        raise ValueError(
            "weight and activations must be matrices"
        )
    if activations.shape[1] != weight.shape[1]:
        raise ValueError(
            "activation feature size must match weight input size"
        )
    if block_size <= 0 or percdamp < 0:
        raise ValueError(
            "block_size must be positive and percdamp non-negative"
        )
    source = weight.detach().float()
    samples = activations.detach().float()
    if (
        not torch.isfinite(source).all()
        or not torch.isfinite(samples).all()
    ):
        raise ValueError("GPTQ inputs contain NaN or Inf")
    hessian = (
        2.0
        / max(samples.shape[0], 1)
        * samples.transpose(0, 1)
        @ samples
    )
    diagonal = torch.diagonal(hessian)
    damp = percdamp * diagonal.mean()
    hessian = hessian + torch.eye(
        hessian.shape[0],
        device=hessian.device,
        dtype=hessian.dtype,
    ) * damp
    if not torch.isfinite(hessian).all():
        raise GptqFallbackError(
            GPTQ_FALLBACK_NON_FINITE_HESSIAN
        )
    chol, cholesky_info = torch.linalg.cholesky_ex(
        hessian,
        check_errors=False,
    )
    if (cholesky_info != 0).any():
        raise GptqFallbackError(
            GPTQ_FALLBACK_CHOLESKY_FAILED
        )
    inverse = torch.cholesky_inverse(chol)
    order = (
        torch.argsort(diagonal, descending=True)
        if actorder
        else torch.arange(
            weight.shape[1],
            device=weight.device,
        )
    )
    inverse_order = torch.argsort(order)
    work = source[:, order].clone()
    quantized = torch.empty_like(work, dtype=torch.int8)
    parameter_shape = (
        (source.shape[0], 1)
        if per_channel
        else (1, 1)
    )
    scales = _clip_scales(
        source,
        bits=bits,
        per_channel=per_channel,
    )
    if scales.shape != parameter_shape:
        raise RuntimeError(
            "GPTQ clipping produced an invalid scale shape"
        )
    zero_points = torch.zeros(
        parameter_shape,
        dtype=torch.int32,
        device=source.device,
    )
    qmin, qmax = integer_range(bits)
    for start in range(0, work.shape[1], block_size):
        end = min(start + block_size, work.shape[1])
        for column in range(start, end):
            current = work[:, column]
            q = torch.round(current / scales[:, 0]).clamp(
                qmin,
                qmax,
            )
            quantized[:, column] = q.to(torch.int8)
            restored = q * scales[:, 0]
            pivot = inverse[
                order[column],
                order[column],
            ].clamp_min(1e-12)
            error = (current - restored) / pivot
            if column + 1 < work.shape[1]:
                coupling = inverse[
                    order[column],
                    order[column + 1 :],
                ]
                work[:, column + 1 :] -= (
                    error.unsqueeze(1)
                    * coupling.unsqueeze(0)
                )
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
