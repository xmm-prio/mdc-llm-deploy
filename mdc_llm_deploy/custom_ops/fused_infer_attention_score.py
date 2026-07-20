"""Fused floating-point attention for Qwen3 inference and ONNX export."""

from __future__ import annotations

import importlib
import math
from typing import Any, ClassVar

import torch
from torch.onnx.symbolic_helper import parse_args

from .base import CustomOp

_MAX_TOKENS = 2_147_483_647
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MASK_DTYPES = (torch.bool, torch.int8, torch.uint8)
tl: Any = None


def _optional_tensors(
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


def _validate_attributes(
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
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if num_key_value_heads < 0:
        raise ValueError("num_key_value_heads must be non-negative")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if input_layout not in {"BNSD", "BSND"}:
        raise NotImplementedError("Torch execution supports only BNSD and BSND layouts")
    unsupported = {
        "pre_tokens": pre_tokens != _MAX_TOKENS,
        "next_tokens": next_tokens != _MAX_TOKENS,
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


def _to_bnsd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    return tensor if layout == "BNSD" else tensor.permute(0, 2, 1, 3)


def _from_bnsd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    return tensor if layout == "BNSD" else tensor.permute(0, 2, 1, 3)


def _validate_metadata(
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
    _validate_attributes(
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
    if query.dtype not in _FLOAT_DTYPES or key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("query, key, and value must share a supported floating-point dtype")

    query_bnsd = _to_bnsd(query, input_layout)
    key_bnsd = _to_bnsd(key, input_layout)
    value_bnsd = _to_bnsd(value, input_layout)
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


def _validate_optional_scope(optional: tuple[torch.Tensor | None, ...]) -> None:
    supported_indices = {1, 2, 3}
    unsupported = [
        FusedInferAttentionScore.torch_input_slots[index + 3]
        for index, value in enumerate(optional)
        if index not in supported_indices and value is not None
    ]
    if unsupported:
        raise NotImplementedError(
            f"Unsupported Torch attention inputs: {', '.join(unsupported)}"
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
    if mask.dtype not in _MASK_DTYPES:
        raise TypeError("atten_mask must have bool, int8, or uint8 dtype")
    try:
        return torch.broadcast_to(mask, shape)
    except RuntimeError as error:
        raise ValueError(f"atten_mask cannot broadcast to {shape}") from error


def _validate_finite(*tensors: torch.Tensor) -> None:
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise ValueError("query, key, and value must contain only finite values")


def _symbolic_is_none(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value.node().mustBeNone())
    except (AttributeError, RuntimeError):
        return False


def _symbolic_dtype(value: Any, name: str) -> str:
    try:
        dtype = value.type().scalarType()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(
            f"ONNX FusedInferAttentionScore requires tensor metadata for {name}"
        ) from error
    if dtype is None:
        raise RuntimeError(f"ONNX FusedInferAttentionScore requires known {name} dtype")
    return str(dtype)


def _symbolic_shape(value: Any, name: str) -> tuple[int | None, ...]:
    try:
        sizes = value.type().sizes()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(
            f"ONNX FusedInferAttentionScore requires tensor metadata for {name}"
        ) from error
    if sizes is None:
        raise RuntimeError(f"ONNX FusedInferAttentionScore requires known {name} shape")
    return tuple(int(size) if size is not None else None for size in sizes)


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


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor | None,
    actual_seq_lengths: torch.Tensor | None,
    actual_seq_lengths_kv: torch.Tensor | None,
    *,
    scale: float,
    input_layout: str,
    softmax_lse_flag: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_bnsd = _to_bnsd(query, input_layout)
    key_bnsd = _to_bnsd(key, input_layout)
    value_bnsd = _to_bnsd(value, input_layout)
    batch, query_heads, query_length, _ = query_bnsd.shape
    key_length = key_bnsd.shape[2]
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
    scores = scores * scale
    scores = scores.masked_fill(~visible, -torch.inf)
    scores = torch.where(active_rows, scores, torch.zeros_like(scores))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, value_expanded.float())
    output = torch.where(active_rows, output, torch.zeros_like(output))
    output = _from_bnsd(output.to(query.dtype), input_layout)
    if softmax_lse_flag:
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        lse = torch.where(active_rows, lse, torch.full_like(lse, torch.inf))
    else:
        lse = torch.zeros(1, dtype=torch.float32, device=query.device)
    return output, lse


def _get_triton_kernel() -> Any:
    kernel = FusedInferAttentionScore._triton_kernel
    if kernel is not None:
        return kernel
    try:
        triton = importlib.import_module("triton")
        triton_language = importlib.import_module("triton.language")
    except ImportError as error:
        raise RuntimeError("Triton is required for CUDA FusedInferAttentionScore") from error
    globals()["tl"] = triton_language

    def online_attention(  # type: ignore[no-untyped-def]
        query_ptr,
        key_ptr,
        value_ptr,
        mask_ptr,
        query_lengths_ptr,
        key_lengths_ptr,
        output_ptr,
        lse_ptr,
        batch,
        query_heads,
        query_length,
        key_length,
        key_heads,
        head_dim,
        scale,
        q_stride_b,
        q_stride_n,
        q_stride_s,
        q_stride_d,
        k_stride_b,
        k_stride_n,
        k_stride_s,
        k_stride_d,
        v_stride_b,
        v_stride_n,
        v_stride_s,
        v_stride_d,
        o_stride_b,
        o_stride_n,
        o_stride_s,
        o_stride_d,
        has_mask,
        has_query_lengths,
        has_key_lengths,
        write_lse,
        block_k,
        block_d,
    ):
        row = tl.program_id(0)
        query_index = row % query_length
        head_index = (row // query_length) % query_heads
        batch_index = row // (query_length * query_heads)
        key_head_index = head_index // (query_heads // key_heads)
        offsets_d = tl.arange(0, block_d)
        dim_mask = offsets_d < head_dim
        query_offsets = (
            batch_index * q_stride_b
            + head_index * q_stride_n
            + query_index * q_stride_s
            + offsets_d * q_stride_d
        )
        query_values = tl.load(query_ptr + query_offsets, mask=dim_mask, other=0.0).to(
            tl.float32
        )
        query_limit = (
            tl.load(query_lengths_ptr + batch_index)
            if has_query_lengths
            else query_length
        )
        key_limit = (
            tl.load(key_lengths_ptr + batch_index) if has_key_lengths else key_length
        )
        query_is_valid = query_index < query_limit
        safe_key_limit = tl.maximum(key_limit, 1)
        maximum = -float("inf")
        denominator = 0.0
        accumulator = tl.zeros((block_d,), tl.float32)
        for start_key in tl.range(0, key_length, block_k):
            offsets_k = start_key + tl.arange(0, block_k)
            key_offsets = (
                batch_index * k_stride_b
                + key_head_index * k_stride_n
                + offsets_k[:, None] * k_stride_s
                + offsets_d[None, :] * k_stride_d
            )
            key_values = tl.load(
                key_ptr + key_offsets,
                mask=(offsets_k[:, None] < key_length) & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(key_values * query_values[None, :], axis=1) * scale
            visible = offsets_k < safe_key_limit
            if has_mask:
                mask_offset = (
                    ((batch_index * query_heads + head_index) * query_length + query_index)
                    * key_length
                    + offsets_k
                )
                visible = visible & (tl.load(mask_ptr + mask_offset, mask=offsets_k < key_length, other=1) == 0)
            scores = tl.where(visible, scores, -float("inf"))
            block_maximum = tl.max(scores, axis=0)
            next_maximum = tl.maximum(maximum, block_maximum)
            correction = tl.exp(maximum - next_maximum)
            probabilities = tl.exp(scores - next_maximum)
            value_offsets = (
                batch_index * v_stride_b
                + key_head_index * v_stride_n
                + offsets_k[:, None] * v_stride_s
                + offsets_d[None, :] * v_stride_d
            )
            value_values = tl.load(
                value_ptr + value_offsets,
                mask=(offsets_k[:, None] < key_length) & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * correction + tl.sum(
                probabilities[:, None] * value_values, axis=0
            )
            denominator = denominator * correction + tl.sum(probabilities, axis=0)
            maximum = next_maximum
        result = accumulator / denominator
        result = tl.where(query_is_valid, result, 0.0)
        output_offsets = (
            batch_index * o_stride_b
            + head_index * o_stride_n
            + query_index * o_stride_s
            + offsets_d * o_stride_d
        )
        tl.store(output_ptr + output_offsets, result, mask=dim_mask)
        if write_lse:
            row_lse = maximum + tl.log(denominator)
            row_lse = tl.where(query_is_valid, row_lse, float("inf"))
            tl.store(lse_ptr + row, row_lse)

    online_attention.__annotations__.update(
        {
            "has_mask": tl.constexpr,
            "has_query_lengths": tl.constexpr,
            "has_key_lengths": tl.constexpr,
            "write_lse": tl.constexpr,
            "block_k": tl.constexpr,
            "block_d": tl.constexpr,
        }
    )
    compiled_kernel = triton.jit(online_attention)
    FusedInferAttentionScore._triton_kernel = compiled_kernel
    return compiled_kernel


class FusedInferAttentionScore(CustomOp):
    """Implement broad Torch attention and narrow MC62 decode ONNX lowering."""

    qualified_name = "mdc_llm_deploy::fused_infer_attention_score"
    schema = (
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
        "str input_layout='BSH', int num_key_value_heads=0, int sparse_mode=0, "
        "int inner_precise=0, int block_size=0, int antiquant_mode=0, "
        "bool softmax_lse_flag=False, int key_antiquant_mode=0, "
        "int value_antiquant_mode=0, int query_quant_mode=0) -> (Tensor, Tensor)"
    )
    torch_input_slots: ClassVar[tuple[str, ...]] = (
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
    onnx_input_slots: ClassVar[tuple[str, ...]] = (
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
    )
    onnx_attribute_defaults: ClassVar[dict[str, object]] = {
        "num_heads": 1,
        "scale": 1.0,
        "input_layout": "BNSD",
        "num_key_value_heads": 0,
    }
    _triton_kernel: ClassVar[Any | None] = None

    @staticmethod
    def cpu(
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
        pre_tokens: int = _MAX_TOKENS,
        next_tokens: int = _MAX_TOKENS,
        input_layout: str = "BSH",
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
        optional = _optional_tensors(
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
        _validate_optional_scope(optional)
        _validate_metadata(
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
        _validate_finite(query, key, value)
        return _reference_attention(
            query,
            key,
            value,
            atten_mask,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            scale=scale,
            input_layout=input_layout,
            softmax_lse_flag=softmax_lse_flag,
        )

    @staticmethod
    def cuda(
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
        pre_tokens: int = _MAX_TOKENS,
        next_tokens: int = _MAX_TOKENS,
        input_layout: str = "BSH",
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
        optional = _optional_tensors(
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
        _validate_optional_scope(optional)
        batch, query_heads, query_length, key_length, head_dim = _validate_metadata(
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
        if query.device.type != "cuda":
            raise ValueError("CUDA kernel requires CUDA tensors")
        _validate_finite(query, key, value)
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
        _visibility(
            mask,
            query_lengths,
            key_lengths,
            (batch, query_heads, query_length, key_length),
        )
        query_bnsd = _to_bnsd(query, input_layout)
        key_bnsd = _to_bnsd(key, input_layout)
        value_bnsd = _to_bnsd(value, input_layout)
        output_bnsd = torch.empty_like(query_bnsd)
        lse = (
            torch.empty(
                batch,
                query_heads,
                query_length,
                1,
                dtype=torch.float32,
                device=query.device,
            )
            if softmax_lse_flag
            else torch.zeros(1, dtype=torch.float32, device=query.device)
        )
        mask_storage = query if mask is None else mask.contiguous()
        kernel = _get_triton_kernel()
        triton = importlib.import_module("triton")
        block_d = triton.next_power_of_2(head_dim)
        if block_d > 512:
            raise NotImplementedError("CUDA attention supports head dimensions up to 512")
        kernel[(batch * query_heads * query_length,)](
            query_bnsd,
            key_bnsd,
            value_bnsd,
            mask_storage,
            query_lengths,
            key_lengths,
            output_bnsd,
            lse,
            batch,
            query_heads,
            query_length,
            key_length,
            key_bnsd.shape[1],
            head_dim,
            scale,
            *query_bnsd.stride(),
            *key_bnsd.stride(),
            *value_bnsd.stride(),
            *output_bnsd.stride(),
            has_mask=mask is not None,
            has_query_lengths=actual_seq_lengths is not None,
            has_key_lengths=actual_seq_lengths_kv is not None,
            write_lse=softmax_lse_flag,
            block_k=32,
            block_d=block_d,
        )
        return _from_bnsd(output_bnsd, input_layout), lse

    @staticmethod
    def fake(
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
        pre_tokens: int = _MAX_TOKENS,
        next_tokens: int = _MAX_TOKENS,
        input_layout: str = "BSH",
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
        optional = _optional_tensors(
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
        _validate_optional_scope(optional)
        batch, query_heads, query_length, _, _ = _validate_metadata(
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
        lse = (
            torch.empty(
                batch,
                query_heads,
                query_length,
                1,
                dtype=torch.float32,
                device=query.device,
            )
            if softmax_lse_flag
            else torch.empty(1, dtype=torch.float32, device=query.device)
        )
        return output, lse

    @staticmethod
    @parse_args(
        "v", "v", "v", "v", "v", "v", "v", "v", "v", "v",
        "v", "v", "v", "v", "v", "v", "v", "v", "v", "v",
        "v", "v", "v", "v", "v", "v", "v", "v", "v",
        "i", "f", "i", "i", "s", "i", "i", "i", "i", "i", "b", "i", "i", "i",
    )
    def onnx(
        graph: Any,
        query: Any,
        key: Any,
        value: Any,
        pse_shift: Any,
        atten_mask: Any,
        actual_seq_lengths: Any,
        actual_seq_lengths_kv: Any,
        dequant_scale1: Any,
        quant_scale1: Any,
        dequant_scale2: Any,
        quant_scale2: Any,
        quant_offset2: Any,
        antiquant_scale: Any,
        antiquant_offset: Any,
        block_table: Any,
        query_padding_size: Any,
        kv_padding_size: Any,
        key_antiquant_scale: Any,
        key_antiquant_offset: Any,
        value_antiquant_scale: Any,
        value_antiquant_offset: Any,
        key_shared_prefix: Any,
        value_shared_prefix: Any,
        actual_shared_prefix_len: Any,
        query_rope: Any,
        key_rope: Any,
        key_rope_antiquant_scale: Any,
        dequant_scale_query: Any,
        learnable_sink: Any,
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
        softmax_lse_flag: bool,
        key_antiquant_mode: int,
        value_antiquant_mode: int,
        query_quant_mode: int,
    ) -> Any:
        if input_layout != "BNSD":
            raise RuntimeError("MC62 ONNX FusedInferAttentionScore supports only BNSD layout")
        _validate_attributes(
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
        if softmax_lse_flag:
            raise RuntimeError("MC62 ONNX FusedInferAttentionScore does not support softmax LSE")
        torch_inputs = (
            query,
            key,
            value,
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
        missing = [
            FusedInferAttentionScore.torch_input_slots[index]
            for index, tensor in enumerate(torch_inputs[:3])
            if _symbolic_is_none(tensor)
        ]
        if missing:
            raise RuntimeError(
                f"MC62 ONNX FusedInferAttentionScore requires Q/K/V: {', '.join(missing)}"
            )
        unsupported = [
            FusedInferAttentionScore.torch_input_slots[index]
            for index, tensor in enumerate(torch_inputs[3:], start=3)
            if not _symbolic_is_none(tensor)
        ]
        if unsupported:
            raise RuntimeError(
                f"MC62 float decode ONNX does not support optional inputs: {', '.join(unsupported)}"
            )
        query_dtype = _symbolic_dtype(query, "query")
        key_dtype = _symbolic_dtype(key, "key")
        value_dtype = _symbolic_dtype(value, "value")
        if query_dtype not in {"Half", "BFloat16"}:
            raise RuntimeError(
                "MC62 float decode ONNX FusedInferAttentionScore supports only FLOAT16 and BFLOAT16 Q/K/V"
            )
        if key_dtype != query_dtype or value_dtype != query_dtype:
            raise RuntimeError("ONNX FusedInferAttentionScore Q/K/V dtypes must match")
        query_shape = _symbolic_shape(query, "query")
        if len(query_shape) != 4:
            raise RuntimeError("MC62 float decode ONNX requires rank-4 BNSD query")
        if query_shape[2] != 1:
            raise RuntimeError(
                "MC62 float decode ONNX requires query sequence length S=1; "
                "float prefill must use small ops or fully-int8 FIA"
            )
        attention_out = graph.op(
            "FusedInferAttentionScore",
            query,
            key,
            value,
            num_heads_i=num_heads,
            scale_f=scale,
            input_layout_s=input_layout,
            num_key_value_heads_i=num_key_value_heads,
        )
        softmax_lse = graph.op(
            "Constant",
            value_t=torch.zeros(1, dtype=torch.float32),
        )
        return attention_out, softmax_lse
