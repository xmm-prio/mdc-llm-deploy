"""Broad Torch execution contract for RmsNorm."""

from __future__ import annotations

import math
from typing import Any

import torch

SUPPORTED_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
MAX_TRITON_BLOCK_SIZE = 65_536


def validate_torch_inputs(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
    *,
    device_type: str | None = None,
    check_values: bool,
) -> None:
    """Validate inputs accepted by broad Torch execution."""
    if not isinstance(x, torch.Tensor) or not isinstance(gamma, torch.Tensor):
        raise TypeError("x and gamma must be tensors")
    if not isinstance(epsilon, (float, int)) or isinstance(epsilon, bool):
        raise TypeError("epsilon must be a real number")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not 1 <= x.ndim <= 8:
        raise ValueError("x must have between 1 and 8 dimensions")
    if not 1 <= gamma.ndim <= x.ndim:
        raise ValueError("gamma must have between 1 and x.ndim dimensions")
    if tuple(x.shape[-gamma.ndim :]) != tuple(gamma.shape):
        raise ValueError("gamma shape must match one or more trailing x dimensions")
    if any(size == 0 for size in gamma.shape):
        raise ValueError("gamma dimensions must be non-empty")
    if x.dtype not in SUPPORTED_DTYPES or gamma.dtype not in SUPPORTED_DTYPES:
        raise TypeError("x and gamma must use float16, bfloat16, or float32")
    if x.dtype != gamma.dtype:
        raise TypeError("x and gamma must have the same dtype")
    if x.device != gamma.device:
        raise ValueError("x and gamma must be on the same device")
    if device_type is not None and x.device.type != device_type:
        raise ValueError(f"x and gamma must be on {device_type}")
    if check_values and (
        not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(gamma).all())
    ):
        raise ValueError("x and gamma must contain only finite values")


def rstd_shape(x: torch.Tensor, gamma: torch.Tensor) -> tuple[Any, ...]:
    """Return the prefix shape retained by reciprocal standard deviation."""
    return tuple(x.shape[: x.ndim - gamma.ndim])
