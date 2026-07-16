"""Grouped-query attention validation, reference kernel, and meta kernel."""
# mypy: disable-error-code="no-any-return"

from __future__ import annotations

import math

import torch
from torch import Tensor

from .validation import (
    broadcastable,
    can_read_values,
    check_dtype,
    check_finite,
    check_no_autograd,
    check_same_device,
)

_FLOAT_DTYPES = {torch.float16, torch.float32, torch.bfloat16}
_ATTENTION_DTYPES = _FLOAT_DTYPES | {torch.int8}
_MASK_DTYPES = {torch.bool, torch.int8, torch.uint8}


def _dequantize_input(
    tensor: Tensor,
    scale: Tensor | None,
    offset: Tensor | None,
) -> Tensor:
    if tensor.is_floating_point():
        return tensor.float()
    assert scale is not None
    zero_point: Tensor | float = 0.0 if offset is None else offset.float()
    return (tensor.float() - zero_point) * scale.float()


def fused_infer_attention_score_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    atten_mask: Tensor | None = None,
    scale: float = 1.0,
    num_heads: int | None = None,
    num_key_value_heads: int | None = None,
    key_antiquant_scale: Tensor | None = None,
    key_antiquant_offset: Tensor | None = None,
    value_antiquant_scale: Tensor | None = None,
    value_antiquant_offset: Tensor | None = None,
    dequant_scale_query: Tensor | None = None,
    quant_scale1: Tensor | None = None,
    softmax_lse_flag: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute grouped-query attention with PyTorch reference operations."""
    del num_heads, num_key_value_heads
    query_float = _dequantize_input(query, dequant_scale_query, None)
    key_float = _dequantize_input(
        key, key_antiquant_scale, key_antiquant_offset
    )
    value_float = _dequantize_input(
        value, value_antiquant_scale, value_antiquant_offset
    )
    groups = query.shape[1] // key.shape[1]
    key_float = key_float.repeat_interleave(groups, dim=1)
    value_float = value_float.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query_float, key_float.transpose(-2, -1)) * scale
    if atten_mask is not None:
        mask = torch.broadcast_to(atten_mask.to(torch.bool), scores.shape)
        scores = scores.masked_fill(mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    if quant_scale1 is not None:
        quantized = torch.round(
            probabilities * quant_scale1.float()
        ).clamp(-128, 127)
        probabilities = quantized / quant_scale1.float()
    output_dtype = query.dtype if query.is_floating_point() else torch.float16
    output = torch.matmul(probabilities, value_float).to(output_dtype)
    lse = (
        torch.logsumexp(scores, dim=-1, keepdim=True)
        if softmax_lse_flag
        else torch.zeros(1, dtype=torch.float32, device=query.device)
    )
    return output, lse


def fused_infer_attention_score_meta(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    atten_mask: Tensor | None = None,
    scale: float = 1.0,
    num_heads: int | None = None,
    num_key_value_heads: int | None = None,
    key_antiquant_scale: Tensor | None = None,
    key_antiquant_offset: Tensor | None = None,
    value_antiquant_scale: Tensor | None = None,
    value_antiquant_offset: Tensor | None = None,
    dequant_scale_query: Tensor | None = None,
    quant_scale1: Tensor | None = None,
    softmax_lse_flag: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return output metadata for grouped-query attention."""
    del (
        key,
        atten_mask,
        scale,
        num_heads,
        num_key_value_heads,
        key_antiquant_scale,
        key_antiquant_offset,
        value_antiquant_scale,
        value_antiquant_offset,
        dequant_scale_query,
        quant_scale1,
    )
    output_shape = (*query.shape[:-1], value.shape[-1])
    output_dtype = query.dtype if query.is_floating_point() else torch.float16
    output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
    lse_shape = (*query.shape[:-1], 1) if softmax_lse_flag else (1,)
    return output, torch.empty(
        lse_shape,
        dtype=torch.float32,
        device=query.device,
    )


def _check_quantization_parameter(
    name: str,
    parameter: Tensor | None,
    tensor: Tensor,
    *,
    required: bool,
    positive: bool = True,
) -> None:
    if required and parameter is None:
        raise ValueError(f"Quantized {name} requires scale")
    if parameter is None:
        return
    if parameter.device != tensor.device:
        raise ValueError(
            f"{name} quantization parameter must use input device"
        )
    if parameter.dtype not in _FLOAT_DTYPES:
        raise TypeError(
            f"{name} quantization parameter must be floating point"
        )
    check_finite("FusedInferAttentionScore", parameter)
    if not broadcastable(parameter.shape, tensor.shape):
        raise ValueError(
            f"{name} quantization parameter is not broadcastable"
        )
    if positive and can_read_values(parameter) and bool((parameter <= 0).any()):
        raise ValueError(f"{name} quantization scale must be positive")


def fused_infer_attention_score(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    atten_mask: Tensor | None = None,
    scale: float = 1.0,
    num_heads: int | None = None,
    num_key_value_heads: int | None = None,
    key_antiquant_scale: Tensor | None = None,
    key_antiquant_offset: Tensor | None = None,
    value_antiquant_scale: Tensor | None = None,
    value_antiquant_offset: Tensor | None = None,
    dequant_scale_query: Tensor | None = None,
    quant_scale1: Tensor | None = None,
    softmax_lse_flag: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute BNSD grouped-query attention."""
    optional = (
        atten_mask,
        key_antiquant_scale,
        key_antiquant_offset,
        value_antiquant_scale,
        value_antiquant_offset,
        dequant_scale_query,
        quant_scale1,
    )
    check_no_autograd(query, key, value, *optional)
    check_same_device("FusedInferAttentionScore", query, key, value)
    check_dtype("FusedInferAttentionScore query", query, _ATTENTION_DTYPES)
    check_dtype("FusedInferAttentionScore key", key, _ATTENTION_DTYPES)
    check_dtype("FusedInferAttentionScore value", value, _ATTENTION_DTYPES)
    check_finite("FusedInferAttentionScore", query, key, value)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Attention scale must be finite and positive")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Attention inputs must use BNSD rank 4")
    if tuple(key.shape) != tuple(value.shape) or query.shape[0] != key.shape[0]:
        raise ValueError("K and V shapes must match, including batch")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match")
    q_heads = query.shape[1]
    kv_heads = key.shape[1]
    if (
        num_heads not in {None, q_heads}
        or num_key_value_heads not in {None, kv_heads}
    ):
        raise ValueError("Head attributes do not match tensors")
    if q_heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if query.is_floating_point():
        if dequant_scale_query is not None:
            raise ValueError("Floating query must not provide dequant scale")
    else:
        _check_quantization_parameter(
            "query",
            dequant_scale_query,
            query,
            required=True,
        )
    if key.dtype != value.dtype:
        raise TypeError("K and V must use one dtype")
    _check_quantization_parameter(
        "key",
        key_antiquant_scale,
        key,
        required=not key.is_floating_point(),
    )
    _check_quantization_parameter(
        "value",
        value_antiquant_scale,
        value,
        required=not value.is_floating_point(),
    )
    for name, offset, tensor, tensor_scale in (
        ("key", key_antiquant_offset, key, key_antiquant_scale),
        ("value", value_antiquant_offset, value, value_antiquant_scale),
    ):
        if tensor.is_floating_point() and (
            tensor_scale is not None or offset is not None
        ):
            raise ValueError(
                f"Floating {name} must not provide antiquant parameters"
            )
        if offset is not None:
            _check_quantization_parameter(
                name,
                offset,
                tensor,
                required=False,
                positive=False,
            )
    if atten_mask is not None:
        if atten_mask.device != query.device:
            raise ValueError("Attention mask must use query device")
        check_dtype("Attention mask", atten_mask, _MASK_DTYPES)
        scores_shape = (
            query.shape[0],
            q_heads,
            query.shape[2],
            key.shape[2],
        )
        if not broadcastable(atten_mask.shape, torch.Size(scores_shape)):
            raise ValueError("Attention mask is not broadcastable")
        if can_read_values(atten_mask):
            mask = torch.broadcast_to(atten_mask.to(torch.bool), scores_shape)
            if bool(mask.all(dim=-1).any()):
                raise ValueError("Attention mask contains a fully masked row")
    if quant_scale1 is not None:
        if (
            quant_scale1.device != query.device
            or quant_scale1.dtype != torch.float32
        ):
            raise TypeError("quant_scale1 must use query device and float32")
        if quant_scale1.numel() != 1:
            raise ValueError("quant_scale1 must be per-tensor")
        check_finite("FusedInferAttentionScore", quant_scale1)
        if can_read_values(quant_scale1) and bool((quant_scale1 <= 0).any()):
            raise ValueError("quant_scale1 must be positive")
    return torch.ops.mdc_llm_deploy.fused_infer_attention_score.default(
        query,
        key,
        value,
        atten_mask,
        scale,
        num_heads,
        num_key_value_heads,
        key_antiquant_scale,
        key_antiquant_offset,
        value_antiquant_scale,
        value_antiquant_offset,
        dequant_scale_query,
        quant_scale1,
        softmax_lse_flag,
    )
