"""Broad Torch contract for ApplyRotaryPosEmb."""

from __future__ import annotations

from collections.abc import Sequence

import torch

LAYOUT_RANK = {1: 4, 2: 4, 3: 4, 4: 3}
HEAD_AXIS = {1: 2, 2: 2, 3: 1, 4: 1}
ROTARY_MODES = frozenset({"half", "interleave", "quarter"})
TORCH_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
QUALIFIED_NAME = "mdc_llm_deploy::apply_rotary_pos_emb"
TORCH_SCHEMA = (
    "(Tensor query, Tensor key, Tensor cos, Tensor sin, "
    "int layout=1, str rotary_mode='half') -> (Tensor, Tensor)"
)


def validate_torch_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int,
    rotary_mode: str,
    *,
    check_values: bool,
) -> int:
    """Validate broad eager/Fake/CUDA inputs and return rotary dimension."""
    if layout not in LAYOUT_RANK:
        raise ValueError("layout must be one of 1=BSND, 2=SBND, 3=BNSD, or 4=TND")
    if rotary_mode not in ROTARY_MODES:
        raise ValueError("rotary_mode must be 'half', 'interleave', or 'quarter'")

    tensors = (query, key, cos, sin)
    if any(tensor.dtype not in TORCH_DTYPES for tensor in tensors):
        raise TypeError("query, key, cos, and sin must have a floating-point RoPE dtype")
    if any(tensor.dtype != query.dtype for tensor in tensors[1:]):
        raise TypeError("query, key, cos, and sin must have the same dtype")
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("query, key, cos, and sin must be on the same device")

    rank = LAYOUT_RANK[layout]
    if any(tensor.ndim != rank for tensor in tensors):
        raise ValueError(f"layout {layout} requires every input to have rank {rank}")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape")

    head_axis = HEAD_AXIS[layout]
    for axis in range(rank):
        if axis != head_axis and query.shape[axis] != key.shape[axis]:
            raise ValueError("query and key may differ only in head count")
    if cos.shape[head_axis] != 1:
        raise ValueError("cos and sin head dimension must be 1")
    for axis in range(rank - 1):
        if axis == head_axis:
            continue
        if cos.shape[axis] not in (1, query.shape[axis]):
            raise ValueError("cos and sin must broadcast over non-head dimensions")

    head_dim = query.shape[-1]
    if key.shape[-1] != head_dim:
        raise ValueError("query and key must have the same head dimension")
    rotary_dim = cos.shape[-1]
    if rotary_dim <= 0 or rotary_dim > head_dim:
        raise ValueError("rotary dimension must satisfy 0 < R <= D")
    divisor = 4 if rotary_mode == "quarter" else 2
    if rotary_dim % divisor:
        raise ValueError(f"{rotary_mode} rotary dimension must be divisible by {divisor}")

    if check_values:
        _validate_finite(tensors)
    return rotary_dim


def _validate_finite(tensors: Sequence[torch.Tensor]) -> None:
    for name, tensor in zip(("query", "key", "cos", "sin"), tensors, strict=True):
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} must contain only finite values")
