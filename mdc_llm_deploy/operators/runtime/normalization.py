"""RMS normalization and rotary position embedding operators."""
# mypy: disable-error-code="no-any-return,no-untyped-call"

from __future__ import annotations

import math

import torch
from torch import Tensor

from .validation import (
    broadcastable,
    check_dtype,
    check_finite,
    check_no_autograd,
    check_rank,
    check_same_device,
    check_same_dtype,
)

_FLOAT_DTYPES = {torch.float16, torch.float32, torch.bfloat16}


def rms_norm_reference(
    x: Tensor,
    gamma: Tensor,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    reduction = tuple(range(x.ndim - gamma.ndim, x.ndim))
    source = x.float()
    rstd = torch.rsqrt(source.square().mean(dim=reduction) + epsilon)
    expanded = rstd[(...,) + (None,) * gamma.ndim]
    output = (source * expanded * gamma.float()).to(x.dtype)
    return output, rstd


def rms_norm_meta(
    x: Tensor,
    gamma: Tensor,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    del epsilon
    return (
        torch.empty_like(x),
        torch.empty(
            x.shape[: x.ndim - gamma.ndim],
            dtype=torch.float32,
            device=x.device,
        ),
    )


def rms_norm(
    x: Tensor,
    gamma: Tensor,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Execute RmsNorm with FP32 accumulation."""
    check_no_autograd(x, gamma)
    check_same_device("RmsNorm", x, gamma)
    check_same_dtype("RmsNorm", x, gamma)
    check_dtype("RmsNorm", x, _FLOAT_DTYPES)
    check_rank("RmsNorm x", x, 1, 8)
    check_rank("RmsNorm gamma", gamma, 1, x.ndim)
    check_finite("RmsNorm", x, gamma)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if tuple(x.shape[-gamma.ndim :]) != tuple(gamma.shape):
        raise ValueError("gamma shape must match trailing x dimensions")
    return torch.ops.mdc_llm_deploy.rms_norm.default(x, gamma, epsilon)


def _rope_parameter(value: Tensor, target: Tensor, layout: int) -> Tensor:
    if value.ndim != target.ndim - 1:
        return value
    dimension = 1 if layout == 3 else target.ndim - 2
    return value.unsqueeze(dimension)


def _rotate_half(value: Tensor, rotary_mode: str) -> Tensor:
    if rotary_mode == "half":
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)
    if rotary_mode == "interleave":
        even = value[..., ::2]
        odd = value[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)
    first, second, third, fourth = value.chunk(4, dim=-1)
    return torch.cat((-third, -fourth, first, second), dim=-1)


def apply_rotary_pos_emb_reference(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    layout: int = 1,
    rotary_mode: str = "half",
) -> tuple[Tensor, Tensor]:
    query_cos = _rope_parameter(cos, query, layout).float()
    query_sin = _rope_parameter(sin, query, layout).float()
    key_cos = _rope_parameter(cos, key, layout).float()
    key_sin = _rope_parameter(sin, key, layout).float()
    query_float = query.float()
    key_float = key.float()
    return (
        (
            query_float * query_cos
            + _rotate_half(query_float, rotary_mode) * query_sin
        ).to(query.dtype),
        (
            key_float * key_cos
            + _rotate_half(key_float, rotary_mode) * key_sin
        ).to(key.dtype),
    )


def apply_rotary_pos_emb_meta(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    layout: int = 1,
    rotary_mode: str = "half",
) -> tuple[Tensor, Tensor]:
    del cos, sin, layout, rotary_mode
    return torch.empty_like(query), torch.empty_like(key)


def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    *,
    layout: int = 1,
    rotary_mode: str = "half",
) -> tuple[Tensor, Tensor]:
    """Apply a supported rotary position embedding layout."""
    check_no_autograd(query, key, cos, sin)
    check_same_device("ApplyRotaryPosEmb", query, key, cos, sin)
    check_same_dtype("ApplyRotaryPosEmb", query, key, cos, sin)
    check_dtype("ApplyRotaryPosEmb", query, _FLOAT_DTYPES)
    check_finite("ApplyRotaryPosEmb", query, key, cos, sin)
    if layout not in {1, 2, 3, 4}:
        raise ValueError("layout must be 1, 2, 3, or 4")
    expected_rank = 3 if layout == 4 else 4
    if query.ndim != expected_rank or key.ndim != expected_rank:
        raise ValueError("query and key rank do not match layout")
    if rotary_mode not in {"half", "interleave", "quarter"}:
        raise ValueError("unsupported rotary_mode")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] % 2:
        raise ValueError("query and key head dim must match and be even")
    if rotary_mode == "quarter" and query.shape[-1] % 4:
        raise ValueError("quarter rotation requires head dim divisible by 4")
    token_dimensions = (
        (0, 1)
        if layout in {1, 2}
        else ((0, 2) if layout == 3 else (0,))
    )
    if any(
        query.shape[index] != key.shape[index]
        for index in token_dimensions
    ):
        raise ValueError("query and key token dimensions must match")
    if tuple(cos.shape) != tuple(sin.shape):
        raise ValueError("cos and sin shapes must match")
    query_cos = _rope_parameter(cos, query, layout)
    key_cos = _rope_parameter(cos, key, layout)
    if not broadcastable(
        query_cos.shape,
        query.shape,
    ) or not broadcastable(key_cos.shape, key.shape):
        raise ValueError(
            "cos and sin are not broadcastable for the declared layout"
        )
    return torch.ops.mdc_llm_deploy.apply_rotary_pos_emb.default(
        query,
        key,
        cos,
        sin,
        layout,
        rotary_mode,
    )
