"""CPU and CUDA kernels for RmsNorm."""

from __future__ import annotations

from typing import Any

import torch

from .contract import MAX_TRITON_BLOCK_SIZE, rstd_shape, validate_torch_inputs

_TRITON_RUNTIME: tuple[Any, Any] | None = None
tl: Any = None


def _load_triton_runtime() -> tuple[Any, Any]:
    """Load and define the Triton kernel only when CUDA execution is requested."""
    global _TRITON_RUNTIME
    if _TRITON_RUNTIME is not None:
        return _TRITON_RUNTIME

    try:
        import triton  # type: ignore[import-not-found]
        import triton.language as triton_language  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("RmsNorm CUDA execution requires Triton") from error
    globals()["tl"] = triton_language

    def rms_norm_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        gamma_ptr,
        y_ptr,
        rstd_ptr,
        epsilon,
        normalized_size,
        block_size,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, block_size)
        mask = columns < normalized_size
        offsets = row * normalized_size + columns
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / normalized_size
        rstd = 1.0 / tl.sqrt(mean_square + epsilon)
        tl.store(y_ptr + offsets, x * rstd * gamma, mask=mask)
        tl.store(rstd_ptr + row, rstd)

    rms_norm_kernel.__annotations__.update(
        {
            "normalized_size": triton_language.constexpr,
            "block_size": triton_language.constexpr,
        }
    )
    compiled_kernel = triton.jit(rms_norm_kernel)
    _TRITON_RUNTIME = (triton, compiled_kernel)
    return _TRITON_RUNTIME


def cpu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute RMS normalization with FP32 accumulation on CPU."""
    validate_torch_inputs(x, gamma, epsilon, device_type="cpu", check_values=True)
    normalized_dims = tuple(range(x.ndim - gamma.ndim, x.ndim))
    x_float = x.float()
    rstd = torch.rsqrt(torch.mean(x_float.square(), dim=normalized_dims) + float(epsilon))
    scale_shape = (*rstd.shape, *((1,) * gamma.ndim))
    y = (x_float * rstd.reshape(scale_shape) * gamma.float()).to(x.dtype)
    return y, rstd


def cuda(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute RMS normalization using the Triton CUDA kernel."""
    validate_torch_inputs(x, gamma, epsilon, device_type="cuda", check_values=True)
    if not x.is_contiguous() or not gamma.is_contiguous():
        raise ValueError("RmsNorm CUDA inputs must be contiguous")

    normalized_size = gamma.numel()
    if normalized_size > MAX_TRITON_BLOCK_SIZE:
        raise ValueError(
            f"RmsNorm CUDA normalized size must not exceed {MAX_TRITON_BLOCK_SIZE}"
        )

    triton, kernel = _load_triton_runtime()
    row_count = x.numel() // normalized_size
    y = torch.empty_like(x)
    rstd = torch.empty(rstd_shape(x, gamma), dtype=torch.float32, device=x.device)
    if row_count == 0:
        return y, rstd

    block_size = triton.next_power_of_2(normalized_size)
    num_warps = 8 if block_size >= 2_048 else 4
    kernel[(row_count,)](
        x,
        gamma,
        y,
        rstd,
        normalized_size=normalized_size,
        block_size=block_size,
        num_warps=num_warps,
        epsilon=float(epsilon),
    )
    return y, rstd
