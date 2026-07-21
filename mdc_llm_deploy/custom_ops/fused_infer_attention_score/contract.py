"""Torch contract and shared validation for fused inference attention."""

from __future__ import annotations

import math
from typing import Final

import torch

MAX_TOKENS: Final = 2_147_483_647
FLOAT_DTYPES: Final = (torch.float16, torch.bfloat16, torch.float32)
MASK_DTYPES: Final = (torch.bool, torch.int8, torch.uint8)
QUALIFIED_NAME: Final = "mdc_llm_deploy::fused_infer_attention_score"
PLUGIN_NAME: Final = "fused_infer_attention_score"
TORCH_SCHEMA: Final = (
    "(Tensor query, Tensor key, Tensor value, Tensor? pse_shift=None, "
    "Tensor? atten_mask=None, Tensor? actual_seq_lengths=None, "
    "Tensor? actual_seq_lengths_kv=None, Tensor? dequant_scale1=None, "
    "Tensor? quant_scale1=None, Tensor? dequant_scale2=None, "
    "Tensor? quant_scale2=None, Tensor? quant_offset2=None, "
    "Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, "
    "Tensor? block_table=None, Tensor? query_padding_size=None, "
    "Tensor? kv_padding_size=None, Tensor? key_antiquant_scale=None, "
    "Tensor? key_antiquant_offset=None, Tensor? value_antiquant_scale=None, "
    "Tensor? value_antiquant_offset=None, Tensor? key_shared_prefix=None, "
    "Tensor? value_shared_prefix=None, Tensor? actual_shared_prefix_len=None, "
    "Tensor? query_rope=None, Tensor? key_rope=None, "
    "Tensor? key_rope_antiquant_scale=None, Tensor? dequant_scale_query=None, "
    "Tensor? learnable_sink=None, int num_heads=1, float scale=1.0, "
    "int pre_tokens=2147483647, int next_tokens=2147483647, "
    "str input_layout='BNSD', int num_key_value_heads=0, int sparse_mode=0, "
    "int inner_precise=0, int block_size=0, int antiquant_mode=0, "
    "bool softmax_lse_flag=False, int key_antiquant_mode=0, "
    "int value_antiquant_mode=0, int query_quant_mode=0) -> (Tensor, Tensor)"
)
TORCH_INPUT_SLOTS: Final = (
    "query",
    "key",
    "value",
    "pse_shift",
    "atten_mask",
    "actual_seq_lengths",
    "actual_seq_lengths_kv",
    "dequant_scale1",
    "quant_scale1",
    "dequant_scale2",
    "quant_scale2",
    "quant_offset2",
    "antiquant_scale",
    "antiquant_offset",
    "block_table",
    "query_padding_size",
    "kv_padding_size",
    "key_antiquant_scale",
    "key_antiquant_offset",
    "value_antiquant_scale",
    "value_antiquant_offset",
    "key_shared_prefix",
    "value_shared_prefix",
    "actual_shared_prefix_len",
    "query_rope",
    "key_rope",
    "key_rope_antiquant_scale",
    "dequant_scale_query",
    "learnable_sink",
)


def optional_tensors(
    pse_shift: torch.Tensor | None,
    atten_mask: torch.Tensor | None,
    actual_seq_lengths: torch.Tensor | None,
    actual_seq_lengths_kv: torch.Tensor | None,
    dequant_scale1: torch.Tensor | None,
    quant_scale1: torch.Tensor | None,
    dequant_scale2: torch.Tensor | None,
    quant_scale2: torch.Tensor | None,
    quant_offset2: torch.Tensor | None,
    antiquant_scale: torch.Tensor | None,
    antiquant_offset: torch.Tensor | None,
    block_table: torch.Tensor | None,
    query_padding_size: torch.Tensor | None,
    kv_padding_size: torch.Tensor | None,
    key_antiquant_scale: torch.Tensor | None,
    key_antiquant_offset: torch.Tensor | None,
    value_antiquant_scale: torch.Tensor | None,
    value_antiquant_offset: torch.Tensor | None,
    key_shared_prefix: torch.Tensor | None,
    value_shared_prefix: torch.Tensor | None,
    actual_shared_prefix_len: torch.Tensor | None,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    key_rope_antiquant_scale: torch.Tensor | None,
    dequant_scale_query: torch.Tensor | None,
    learnable_sink: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
    """Collect optional inputs in Torch schema order."""
    return (
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


def validate_optional_scope(optional: tuple[torch.Tensor | None, ...]) -> None:
    """Reject Torch features not implemented by the broad local kernel."""
    supported_indices = {1, 2, 3}
    unsupported = [
        TORCH_INPUT_SLOTS[index + 3]
        for index, value in enumerate(optional)
        if index not in supported_indices and value is not None
    ]
    if unsupported:
        raise NotImplementedError(
            f"Unsupported Torch attention inputs: {', '.join(unsupported)}"
        )


def validate_attributes(
    *,
    num_heads: int,
    scale: float,
    pre_tokens: int,
    next_tokens: int,
    input_layout: str,
    num_key_value_heads: int,
    sparse_mode: int,
    inner_precise: int,
    block_size: int,
    antiquant_mode: int,
    key_antiquant_mode: int,
    value_antiquant_mode: int,
    query_quant_mode: int,
) -> None:
    """Validate attributes supported by broad Torch execution."""
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if num_key_value_heads < 0:
        raise ValueError("num_key_value_heads must be non-negative")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if input_layout not in {"BNSD", "BSND"}:
        raise NotImplementedError("Torch execution supports only BNSD and BSND layouts")
    unsupported = {
        "pre_tokens": pre_tokens != MAX_TOKENS,
        "next_tokens": next_tokens != MAX_TOKENS,
        "sparse_mode": sparse_mode != 0,
        "inner_precise": inner_precise != 0,
        "block_size": block_size != 0,
        "antiquant_mode": antiquant_mode != 0,
        "key_antiquant_mode": key_antiquant_mode != 0,
        "value_antiquant_mode": value_antiquant_mode != 0,
        "query_quant_mode": query_quant_mode != 0,
    }
    names = [name for name, enabled in unsupported.items() if enabled]
    if names:
        raise NotImplementedError(f"Unsupported Torch attention attributes: {', '.join(names)}")


def to_bnsd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    """Normalize a supported layout to BNSD."""
    return tensor if layout == "BNSD" else tensor.permute(0, 2, 1, 3)


def from_bnsd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    """Restore a BNSD tensor to the requested layout."""
    return tensor if layout == "BNSD" else tensor.permute(0, 2, 1, 3)


def validate_metadata(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    scale: float,
    pre_tokens: int,
    next_tokens: int,
    input_layout: str,
    num_key_value_heads: int,
    sparse_mode: int,
    inner_precise: int,
    block_size: int,
    antiquant_mode: int,
    key_antiquant_mode: int,
    value_antiquant_mode: int,
    query_quant_mode: int,
) -> tuple[int, int, int, int, int]:
    """Validate broad Torch tensor metadata."""
    validate_attributes(
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
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be rank-4 tensors")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if query.dtype not in FLOAT_DTYPES or key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("query, key, and value must share a supported floating-point dtype")

    query_bnsd = to_bnsd(query, input_layout)
    key_bnsd = to_bnsd(key, input_layout)
    value_bnsd = to_bnsd(value, input_layout)
    batch, query_heads, query_length, head_dim = query_bnsd.shape
    key_batch, key_heads, key_length, key_dim = key_bnsd.shape
    if value_bnsd.shape != key_bnsd.shape:
        raise ValueError("key and value shapes must match in the supported Qwen3 scope")
    if key_batch != batch or key_dim != head_dim:
        raise ValueError("query, key, and value batch/head dimensions are incompatible")
    if query_heads != num_heads:
        raise ValueError("num_heads must match the query head axis")
    effective_kv_heads = key_heads if num_key_value_heads == 0 else num_key_value_heads
    if effective_kv_heads != key_heads:
        raise ValueError("num_key_value_heads must match the key/value head axis")
    if query_heads % key_heads:
        raise ValueError("num_heads must be divisible by num_key_value_heads for GQA")
    if query_length == 0 or key_length == 0 or head_dim == 0:
        raise ValueError("sequence lengths and head dimension must be non-zero")
    return batch, query_heads, query_length, key_length, head_dim
