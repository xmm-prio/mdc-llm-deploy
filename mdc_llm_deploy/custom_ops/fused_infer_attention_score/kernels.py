"""CPU and CUDA kernels for broad Torch fused-attention execution."""

from __future__ import annotations

import torch

from .contract import (
    MASK_DTYPES,
    MAX_TOKENS,
    from_bnsd,
    optional_tensors,
    to_bnsd,
    validate_metadata,
    validate_optional_scope,
)


def _sequence_lengths(
    lengths: torch.Tensor | None,
    *,
    batch: int,
    maximum: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if lengths is None:
        return torch.full((batch,), maximum, dtype=torch.int64, device=device)
    if lengths.device != device:
        raise ValueError(f"{name} must be on the same device as query")
    if lengths.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype int64")
    if lengths.numel() not in {1, batch}:
        raise ValueError(f"{name} must contain one value or one value per batch")
    normalized = lengths.reshape(-1)
    if normalized.numel() == 1:
        normalized = normalized.expand(batch)
    if bool(torch.any(normalized < 0).item()) or bool(torch.any(normalized > maximum).item()):
        raise ValueError(f"{name} values must be in [0, {maximum}]")
    return normalized.contiguous()


def _expanded_mask(
    mask: torch.Tensor | None,
    shape: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.device != device:
        raise ValueError("atten_mask must be on the same device as query")
    if mask.dtype not in MASK_DTYPES:
        raise TypeError("atten_mask must have bool, int8, or uint8 dtype")
    try:
        return torch.broadcast_to(mask, shape)
    except RuntimeError as error:
        raise ValueError(f"atten_mask cannot broadcast to {shape}") from error


def _visibility(
    mask: torch.Tensor | None,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    shape: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, query_length, key_length = shape
    query_positions = torch.arange(query_length, device=query_lengths.device)
    key_positions = torch.arange(key_length, device=query_lengths.device)
    active_queries = query_positions[None, :] < query_lengths[:, None]
    visible = key_positions[None, :] < key_lengths[:, None]
    visible = visible[:, None, None, :].expand(shape)
    if mask is not None:
        visible = visible & ~mask.to(torch.bool)
    active_rows = active_queries[:, None, :, None].expand(batch, shape[1], query_length, 1)
    fully_masked = active_rows.squeeze(-1) & ~visible.any(dim=-1)
    if bool(fully_masked.any().item()):
        raise ValueError("Every active query row must contain at least one visible key")
    return visible, active_rows


def attention_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pse_shift: torch.Tensor | None = None,
    atten_mask: torch.Tensor | None = None,
    actual_seq_lengths: torch.Tensor | None = None,
    actual_seq_lengths_kv: torch.Tensor | None = None,
    dequant_scale1: torch.Tensor | None = None,
    quant_scale1: torch.Tensor | None = None,
    dequant_scale2: torch.Tensor | None = None,
    quant_scale2: torch.Tensor | None = None,
    quant_offset2: torch.Tensor | None = None,
    antiquant_scale: torch.Tensor | None = None,
    antiquant_offset: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    query_padding_size: torch.Tensor | None = None,
    kv_padding_size: torch.Tensor | None = None,
    key_antiquant_scale: torch.Tensor | None = None,
    key_antiquant_offset: torch.Tensor | None = None,
    value_antiquant_scale: torch.Tensor | None = None,
    value_antiquant_offset: torch.Tensor | None = None,
    key_shared_prefix: torch.Tensor | None = None,
    value_shared_prefix: torch.Tensor | None = None,
    actual_shared_prefix_len: torch.Tensor | None = None,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    key_rope_antiquant_scale: torch.Tensor | None = None,
    dequant_scale_query: torch.Tensor | None = None,
    learnable_sink: torch.Tensor | None = None,
    num_heads: int = 1,
    scale: float = 1.0,
    pre_tokens: int = MAX_TOKENS,
    next_tokens: int = MAX_TOKENS,
    input_layout: str = "BNSD",
    num_key_value_heads: int = 0,
    sparse_mode: int = 0,
    inner_precise: int = 0,
    block_size: int = 0,
    antiquant_mode: int = 0,
    softmax_lse_flag: bool = False,
    key_antiquant_mode: int = 0,
    value_antiquant_mode: int = 0,
    query_quant_mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute attention on CPU or CUDA with FP32 accumulation."""
    optional = optional_tensors(
        pse_shift,
        atten_mask,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        dequant_scale1,
        quant_scale1,
        dequant_scale2,
        quant_scale2,
        quant_offset2,
        antiquant_scale,
        antiquant_offset,
        block_table,
        query_padding_size,
        kv_padding_size,
        key_antiquant_scale,
        key_antiquant_offset,
        value_antiquant_scale,
        value_antiquant_offset,
        key_shared_prefix,
        value_shared_prefix,
        actual_shared_prefix_len,
        query_rope,
        key_rope,
        key_rope_antiquant_scale,
        dequant_scale_query,
        learnable_sink,
    )
    validate_optional_scope(optional)
    batch, query_heads, query_length, key_length, _ = validate_metadata(
        query,
        key,
        value,
        num_heads=num_heads,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
        sparse_mode=sparse_mode,
        inner_precise=inner_precise,
        block_size=block_size,
        antiquant_mode=antiquant_mode,
        key_antiquant_mode=key_antiquant_mode,
        value_antiquant_mode=value_antiquant_mode,
        query_quant_mode=query_quant_mode,
    )
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in (query, key, value)):
        raise ValueError("query, key, and value must contain only finite values")

    query_bnsd = to_bnsd(query, input_layout)
    key_bnsd = to_bnsd(key, input_layout)
    value_bnsd = to_bnsd(value, input_layout)
    query_lengths = _sequence_lengths(
        actual_seq_lengths,
        batch=batch,
        maximum=query_length,
        device=query.device,
        name="actual_seq_lengths",
    )
    key_lengths = _sequence_lengths(
        actual_seq_lengths_kv,
        batch=batch,
        maximum=key_length,
        device=query.device,
        name="actual_seq_lengths_kv",
    )
    mask = _expanded_mask(
        atten_mask,
        (batch, query_heads, query_length, key_length),
        query.device,
    )
    visible, active_rows = _visibility(
        mask,
        query_lengths,
        key_lengths,
        (batch, query_heads, query_length, key_length),
    )
    repeats = query_heads // key_bnsd.shape[1]
    key_expanded = key_bnsd.repeat_interleave(repeats, dim=1)
    value_expanded = value_bnsd.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(query_bnsd.float(), key_expanded.float().transpose(-1, -2))
    scores = scores.mul(scale).masked_fill(~visible, -torch.inf)
    scores = torch.where(active_rows, scores, torch.zeros_like(scores))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, value_expanded.float())
    output = torch.where(active_rows, output, torch.zeros_like(output))
    output = from_bnsd(output.to(query.dtype), input_layout)
    if softmax_lse_flag:
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        lse = torch.where(active_rows, lse, torch.full_like(lse, torch.inf))
    else:
        lse = torch.zeros(1, dtype=torch.float32, device=query.device)
    return output, lse
