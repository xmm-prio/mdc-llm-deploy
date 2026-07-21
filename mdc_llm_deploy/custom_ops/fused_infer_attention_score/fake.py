"""FakeTensor implementation for broad Torch fused attention."""

from __future__ import annotations

import torch

from .contract import (
    MAX_TOKENS,
    optional_tensors,
    validate_metadata,
    validate_optional_scope,
)


def fake_attention(
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
    """Infer output metadata without reading tensor values."""
    validate_optional_scope(
        optional_tensors(
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
    )
    batch, query_heads, query_length, _, _ = validate_metadata(
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
    output = torch.empty_like(query)
    lse_shape = (batch, query_heads, query_length, 1) if softmax_lse_flag else (1,)
    lse = torch.empty(lse_shape, dtype=torch.float32, device=query.device)
    return output, lse
