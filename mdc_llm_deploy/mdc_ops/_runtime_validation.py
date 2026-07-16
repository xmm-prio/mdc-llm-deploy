"""Shared runtime validation primitives for MDC operators."""
# mypy: disable-error-code="no-any-return,no-untyped-call"

from __future__ import annotations

import torch
from torch import Tensor


def is_fake(value: Tensor) -> bool:
    """Return whether a tensor is managed by FakeTensorMode."""
    return type(value).__name__ == "FakeTensor"


def can_read_values(value: Tensor) -> bool:
    """Return whether eager validation may inspect tensor values."""
    return (
        value.device.type != "meta"
        and not is_fake(value)
        and not torch.compiler.is_compiling()
    )


def check_no_autograd(*values: Tensor | None) -> None:
    """Reject autograd inputs unsupported by MDC custom operators."""
    if torch.is_grad_enabled() and any(
        value is not None and value.requires_grad for value in values
    ):
        raise RuntimeError("MDC custom operators do not support autograd")


def check_finite(name: str, *values: Tensor | None) -> None:
    """Require all inspectable floating-point inputs to be finite."""
    for value in values:
        if (
            value is not None
            and value.is_floating_point()
            and can_read_values(value)
            and not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{name} input contains NaN or Inf")


def check_same_device(name: str, *values: Tensor | None) -> None:
    """Require all present inputs to use one device."""
    devices = {value.device for value in values if value is not None}
    if len(devices) != 1:
        raise ValueError(f"{name} inputs must use one device")


def check_same_dtype(name: str, *values: Tensor | None) -> None:
    """Require all present inputs to use one dtype."""
    dtypes = {value.dtype for value in values if value is not None}
    if len(dtypes) != 1:
        raise TypeError(f"{name} inputs must use one dtype")


def check_dtype(
    name: str,
    value: Tensor,
    allowed: set[torch.dtype],
) -> None:
    """Require a tensor dtype to belong to an allowed set."""
    if value.dtype not in allowed:
        allowed_names = ", ".join(sorted(str(dtype) for dtype in allowed))
        raise TypeError(f"{name} dtype must be one of: {allowed_names}")


def check_rank(
    name: str,
    value: Tensor,
    minimum: int,
    maximum: int,
) -> None:
    """Require a tensor rank to lie in an inclusive interval."""
    if not minimum <= value.ndim <= maximum:
        raise ValueError(f"{name} rank must be in [{minimum}, {maximum}]")


def broadcastable(source: torch.Size, target: torch.Size) -> bool:
    """Return whether source broadcasts exactly to target."""
    try:
        return torch.broadcast_shapes(tuple(source), tuple(target)) == tuple(target)
    except RuntimeError:
        return False
