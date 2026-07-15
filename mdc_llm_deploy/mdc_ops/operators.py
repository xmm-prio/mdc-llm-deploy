"""MDC operator validation, reference kernels, and device dispatch."""
# mypy: disable-error-code="no-any-return,no-untyped-call"

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from .schema import OPERATOR_SCHEMAS, TORCH_NAMESPACE

type Kernel = Callable[..., Tensor | tuple[Tensor, Tensor]]
_FLOAT_DTYPES = {torch.float16, torch.float32, torch.bfloat16}
_ATTENTION_DTYPES = _FLOAT_DTYPES | {torch.int8}
_MASK_DTYPES = {torch.bool, torch.int8, torch.uint8}
_MOE_EXPERT_COUNT = 5
_MOE_SCALE_COUNT = 21


def _is_fake(value: Tensor) -> bool:
    return type(value).__name__ == "FakeTensor"


def _can_read_values(value: Tensor) -> bool:
    return (
        value.device.type != "meta"
        and not _is_fake(value)
        and not torch.compiler.is_compiling()
    )


def _check_no_autograd(*values: Tensor | None) -> None:
    if torch.is_grad_enabled() and any(
        value is not None and value.requires_grad for value in values
    ):
        raise RuntimeError("MDC custom operators do not support autograd")


def _check_finite(name: str, *values: Tensor | None) -> None:
    for value in values:
        if (
            value is not None
            and value.is_floating_point()
            and _can_read_values(value)
            and not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{name} input contains NaN or Inf")


def _check_same_device(name: str, *values: Tensor | None) -> None:
    devices = {value.device for value in values if value is not None}
    if len(devices) != 1:
        raise ValueError(f"{name} inputs must use one device")


def _check_same_dtype(name: str, *values: Tensor | None) -> None:
    dtypes = {value.dtype for value in values if value is not None}
    if len(dtypes) != 1:
        raise TypeError(f"{name} inputs must use one dtype")


def _check_dtype(name: str, value: Tensor, allowed: set[torch.dtype]) -> None:
    if value.dtype not in allowed:
        allowed_names = ", ".join(sorted(str(dtype) for dtype in allowed))
        raise TypeError(f"{name} dtype must be one of: {allowed_names}")


def _check_rank(name: str, value: Tensor, minimum: int, maximum: int) -> None:
    if not minimum <= value.ndim <= maximum:
        raise ValueError(f"{name} rank must be in [{minimum}, {maximum}]")


def _broadcastable(source: torch.Size, target: torch.Size) -> bool:
    try:
        return torch.broadcast_shapes(tuple(source), tuple(target)) == tuple(target)
    except RuntimeError:
        return False


def _rms_norm_reference(
    x: Tensor, gamma: Tensor, epsilon: float = 1e-6
) -> tuple[Tensor, Tensor]:
    reduction = tuple(range(x.ndim - gamma.ndim, x.ndim))
    source = x.float()
    rstd = torch.rsqrt(source.square().mean(dim=reduction) + epsilon)
    expanded = rstd[(...,) + (None,) * gamma.ndim]
    output = (source * expanded * gamma.float()).to(x.dtype)
    return output, rstd


def _rms_norm_meta(
    x: Tensor, gamma: Tensor, epsilon: float = 1e-6
) -> tuple[Tensor, Tensor]:
    del epsilon
    return (
        torch.empty_like(x),
        torch.empty(x.shape[: x.ndim - gamma.ndim], dtype=torch.float32, device=x.device),
    )


def rms_norm(x: Tensor, gamma: Tensor, epsilon: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Execute RmsNorm with FP32 accumulation."""
    _check_no_autograd(x, gamma)
    _check_same_device("RmsNorm", x, gamma)
    _check_same_dtype("RmsNorm", x, gamma)
    _check_dtype("RmsNorm", x, _FLOAT_DTYPES)
    _check_rank("RmsNorm x", x, 1, 8)
    _check_rank("RmsNorm gamma", gamma, 1, x.ndim)
    _check_finite("RmsNorm", x, gamma)
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


def _apply_rotary_pos_emb_reference(
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
        (query_float * query_cos + _rotate_half(query_float, rotary_mode) * query_sin).to(
            query.dtype
        ),
        (key_float * key_cos + _rotate_half(key_float, rotary_mode) * key_sin).to(
            key.dtype
        ),
    )


def _apply_rotary_pos_emb_meta(
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
    _check_no_autograd(query, key, cos, sin)
    _check_same_device("ApplyRotaryPosEmb", query, key, cos, sin)
    _check_same_dtype("ApplyRotaryPosEmb", query, key, cos, sin)
    _check_dtype("ApplyRotaryPosEmb", query, _FLOAT_DTYPES)
    _check_finite("ApplyRotaryPosEmb", query, key, cos, sin)
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
    token_dimensions = (0, 1) if layout in {1, 2} else ((0, 2) if layout == 3 else (0,))
    if any(query.shape[index] != key.shape[index] for index in token_dimensions):
        raise ValueError("query and key token dimensions must match")
    if tuple(cos.shape) != tuple(sin.shape):
        raise ValueError("cos and sin shapes must match")
    query_cos = _rope_parameter(cos, query, layout)
    key_cos = _rope_parameter(cos, key, layout)
    if not _broadcastable(query_cos.shape, query.shape) or not _broadcastable(
        key_cos.shape, key.shape
    ):
        raise ValueError("cos and sin are not broadcastable for the declared layout")
    return torch.ops.mdc_llm_deploy.apply_rotary_pos_emb.default(
        query, key, cos, sin, layout, rotary_mode
    )


def _dequantize_attention_input(
    tensor: Tensor,
    scale: Tensor | None,
    offset: Tensor | None,
) -> Tensor:
    if tensor.is_floating_point():
        return tensor.float()
    assert scale is not None
    zero_point: Tensor | float = 0.0 if offset is None else offset.float()
    return (tensor.float() - zero_point) * scale.float()


def _fused_infer_attention_score_reference(
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
    del num_heads, num_key_value_heads
    query_float = _dequantize_attention_input(query, dequant_scale_query, None)
    key_float = _dequantize_attention_input(
        key, key_antiquant_scale, key_antiquant_offset
    )
    value_float = _dequantize_attention_input(
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
        quantized = torch.round(probabilities * quant_scale1.float()).clamp(-128, 127)
        probabilities = quantized / quant_scale1.float()
    output_dtype = query.dtype if query.is_floating_point() else torch.float16
    output = torch.matmul(probabilities, value_float).to(output_dtype)
    lse = (
        torch.logsumexp(scores, dim=-1, keepdim=True)
        if softmax_lse_flag
        else torch.zeros(1, dtype=torch.float32, device=query.device)
    )
    return output, lse


def _fused_infer_attention_score_meta(
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
    return output, torch.empty(lse_shape, dtype=torch.float32, device=query.device)


def _check_attention_parameter(
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
        raise ValueError(f"{name} quantization parameter must use input device")
    if parameter.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} quantization parameter must be floating point")
    _check_finite("FusedInferAttentionScore", parameter)
    if not _broadcastable(parameter.shape, tensor.shape):
        raise ValueError(f"{name} quantization parameter is not broadcastable")
    if positive and _can_read_values(parameter) and bool((parameter <= 0).any()):
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
    _check_no_autograd(query, key, value, *optional)
    _check_same_device("FusedInferAttentionScore", query, key, value)
    _check_dtype("FusedInferAttentionScore query", query, _ATTENTION_DTYPES)
    _check_dtype("FusedInferAttentionScore key", key, _ATTENTION_DTYPES)
    _check_dtype("FusedInferAttentionScore value", value, _ATTENTION_DTYPES)
    _check_finite("FusedInferAttentionScore", query, key, value)
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
    if num_heads not in {None, q_heads} or num_key_value_heads not in {None, kv_heads}:
        raise ValueError("Head attributes do not match tensors")
    if q_heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if query.is_floating_point():
        if dequant_scale_query is not None:
            raise ValueError("Floating query must not provide dequant scale")
    else:
        _check_attention_parameter(
            "query", dequant_scale_query, query, required=True
        )
    if key.dtype != value.dtype:
        raise TypeError("K and V must use one dtype")
    _check_attention_parameter(
        "key", key_antiquant_scale, key, required=not key.is_floating_point()
    )
    _check_attention_parameter(
        "value", value_antiquant_scale, value, required=not value.is_floating_point()
    )
    for name, offset, tensor, tensor_scale in (
        ("key", key_antiquant_offset, key, key_antiquant_scale),
        ("value", value_antiquant_offset, value, value_antiquant_scale),
    ):
        if tensor.is_floating_point() and (tensor_scale is not None or offset is not None):
            raise ValueError(f"Floating {name} must not provide antiquant parameters")
        if offset is not None:
            _check_attention_parameter(
                name, offset, tensor, required=False, positive=False
            )
    if atten_mask is not None:
        if atten_mask.device != query.device:
            raise ValueError("Attention mask must use query device")
        _check_dtype("Attention mask", atten_mask, _MASK_DTYPES)
        scores_shape = (query.shape[0], q_heads, query.shape[2], key.shape[2])
        if not _broadcastable(atten_mask.shape, torch.Size(scores_shape)):
            raise ValueError("Attention mask is not broadcastable")
        if _can_read_values(atten_mask):
            mask = torch.broadcast_to(atten_mask.to(torch.bool), scores_shape)
            if bool(mask.all(dim=-1).any()):
                raise ValueError("Attention mask contains a fully masked row")
    if quant_scale1 is not None:
        if quant_scale1.device != query.device or quant_scale1.dtype != torch.float32:
            raise TypeError("quant_scale1 must use query device and float32")
        if quant_scale1.numel() != 1:
            raise ValueError("quant_scale1 must be per-tensor")
        _check_finite("FusedInferAttentionScore", quant_scale1)
        if _can_read_values(quant_scale1) and bool((quant_scale1 <= 0).any()):
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


def _axis_parameter(parameter: Tensor, x: Tensor, axis: int) -> Tensor:
    if parameter.ndim == 0:
        return parameter
    if parameter.numel() == 1:
        return parameter.reshape((1,) * x.ndim)
    if parameter.ndim == 1 and parameter.shape[0] == x.shape[axis]:
        shape = [1] * x.ndim
        shape[axis] = parameter.shape[0]
        return parameter.reshape(shape)
    return parameter


def _ascend_quant_v2_reference(
    x: Tensor,
    scale: Tensor,
    offset: Tensor | None = None,
    axis: int = -1,
    dtype: int = 2,
) -> Tensor:
    del dtype
    normalized_axis = axis % x.ndim
    multiplier = _axis_parameter(scale, x, normalized_axis).float()
    zero_point: Tensor | float = (
        0.0
        if offset is None
        else _axis_parameter(offset, x, normalized_axis).float()
    )
    return (
        torch.round(x.float() * multiplier + zero_point)
        .clamp(-128, 127)
        .to(torch.int8)
    )


def _ascend_quant_v2_meta(
    x: Tensor,
    scale: Tensor,
    offset: Tensor | None = None,
    axis: int = -1,
    dtype: int = 2,
) -> Tensor:
    del scale, offset, axis, dtype
    return torch.empty_like(x, dtype=torch.int8)


def _check_axis_parameter(name: str, parameter: Tensor, x: Tensor, axis: int) -> None:
    shaped = _axis_parameter(parameter, x, axis)
    if not _broadcastable(shaped.shape, x.shape):
        raise ValueError(f"{name} does not match quantization axis")
    if parameter.numel() not in {1, x.shape[axis]}:
        raise ValueError(f"{name} does not match quantization axis")


def ascend_quant_v2(
    x: Tensor,
    scale: Tensor,
    offset: Tensor | None = None,
    *,
    axis: int = -1,
    dtype: int = 2,
) -> Tensor:
    """Apply multiplication-scale INT8 quantization with ties-to-even."""
    _check_no_autograd(x, scale, offset)
    _check_same_device("AscendQuantV2", x, scale, offset)
    _check_same_dtype("AscendQuantV2", x, scale, offset)
    _check_dtype("AscendQuantV2", x, _FLOAT_DTYPES)
    _check_rank("AscendQuantV2 x", x, 1, 8)
    _check_finite("AscendQuantV2", x, scale, offset)
    if dtype != 2:
        raise ValueError("0.1.0 only supports GE dtype=2 INT8")
    if axis < -x.ndim or axis >= x.ndim:
        raise ValueError("axis is outside input rank")
    normalized_axis = axis % x.ndim
    _check_axis_parameter("scale", scale, x, normalized_axis)
    if offset is not None:
        _check_axis_parameter("offset", offset, x, normalized_axis)
    if _can_read_values(scale) and bool((scale <= 0).any()):
        raise ValueError("scale must be positive")
    return torch.ops.mdc_llm_deploy.ascend_quant_v2.default(
        x, scale, offset, axis, dtype
    )


def _decode_dequant_scale(encoded: Tensor) -> Tensor:
    raw = encoded.to(torch.int64)
    if _can_read_values(encoded) and bool((raw >> 32).ne(0).any()):
        raise ValueError("encoded scale high 32 bits must be zero")
    low_bits = (raw & 0xFFFFFFFF).to(torch.int32).contiguous()
    decoded = low_bits.view(torch.float32)
    if _can_read_values(encoded) and not bool(torch.isfinite(decoded).all()):
        raise ValueError("encoded scale decodes to NaN or Inf")
    return decoded


def _ascend_dequant_reference(
    x: Tensor,
    deq_scale: Tensor,
    sqrt_mode: bool = False,
    relu_flag: bool = False,
    dtype: int = 0,
) -> Tensor:
    scale = _decode_dequant_scale(deq_scale)
    if sqrt_mode:
        scale = torch.sqrt(scale) * torch.sqrt(scale)
    output = x.float() * scale
    if relu_flag:
        output = output.relu()
    return output.to(torch.float32 if dtype == 0 else torch.float16)


def _ascend_dequant_meta(
    x: Tensor,
    deq_scale: Tensor,
    sqrt_mode: bool = False,
    relu_flag: bool = False,
    dtype: int = 0,
) -> Tensor:
    del deq_scale, sqrt_mode, relu_flag
    return torch.empty_like(x, dtype=torch.float32 if dtype == 0 else torch.float16)


def ascend_dequant(
    x: Tensor,
    deq_scale: Tensor,
    *,
    sqrt_mode: bool = False,
    relu_flag: bool = False,
    dtype: int = 0,
) -> Tensor:
    """Decode restricted uint64 FP32 bits and dequantize INT32 input."""
    _check_no_autograd(x, deq_scale)
    _check_same_device("AscendDequant", x, deq_scale)
    if x.dtype != torch.int32:
        raise TypeError("AscendDequant x must use int32")
    if deq_scale.dtype != torch.uint64:
        raise TypeError("AscendDequant scale must use uint64")
    _check_rank("AscendDequant x", x, 1, 8)
    if dtype not in {0, 1}:
        raise ValueError("dtype must be 0 or 1")
    if deq_scale.ndim > 1 or deq_scale.numel() not in {1, x.shape[-1]}:
        raise ValueError("deq_scale must be scalar or match output channels")
    if _can_read_values(deq_scale):
        _decode_dequant_scale(deq_scale)
    return torch.ops.mdc_llm_deploy.ascend_dequant.default(
        x, deq_scale, sqrt_mode, relu_flag, dtype
    )


def _moe_expert_reference(
    x: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    expert_weights: Tensor,
    quant_scales: Tensor,
    quant_offsets: Tensor | None = None,
) -> Tensor:
    offsets = (
        torch.zeros_like(quant_scales, dtype=torch.int32)
        if quant_offsets is None
        else quant_offsets
    )
    hidden = (x.float() - offsets[0].float()) * quant_scales[0]
    hidden_size = x.shape[1]
    intermediate_size = expert_weights.numel() // (
        _MOE_EXPERT_COUNT * 3 * hidden_size
    )
    cursor = 0
    output = torch.zeros_like(hidden)
    for expert_id in range(_MOE_EXPERT_COUNT):
        base = 1 + expert_id * 4
        matrices = []
        for rows, columns, scale_index in (
            (intermediate_size, hidden_size, base),
            (intermediate_size, hidden_size, base + 1),
            (hidden_size, intermediate_size, base + 3),
        ):
            length = rows * columns
            packed = expert_weights[cursor : cursor + length].view(rows, columns)
            matrix = (
                packed.float() - offsets[scale_index].float()
            ) * quant_scales[scale_index]
            matrices.append(matrix)
            cursor += length
        gate, up, down = matrices
        intermediate = functional.silu(hidden @ gate.t()) * (hidden @ up.t())
        activation_scale = quant_scales[base + 2]
        activation_offset = offsets[base + 2].float()
        quantized = torch.round(
            intermediate / activation_scale + activation_offset
        ).clamp(-128, 127)
        intermediate = (quantized - activation_offset) * activation_scale
        expert_output = intermediate @ down.t()
        weight = (
            (topk_ids == expert_id).to(topk_weight.dtype) * topk_weight
        ).sum(dim=1, keepdim=True)
        output += expert_output * weight.float()
    return output.to(torch.float16)


def _moe_expert_meta(
    x: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    expert_weights: Tensor,
    quant_scales: Tensor,
    quant_offsets: Tensor | None = None,
) -> Tensor:
    del topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets
    return torch.empty_like(x, dtype=torch.float16)


def moe_expert(
    x: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    expert_weights: Tensor,
    quant_scales: Tensor,
    quant_offsets: Tensor | None = None,
) -> Tensor:
    """Execute packed five-expert Tiny MoE with 21 ordered parameters."""
    values = (x, topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets)
    _check_no_autograd(*values)
    _check_same_device("MoeExpert", *values)
    if x.dtype != torch.int8 or expert_weights.dtype != torch.int8:
        raise TypeError("MoeExpert activations and weights must use int8")
    if topk_ids.dtype != torch.int16 or topk_weight.dtype != torch.float16:
        raise TypeError("MoeExpert routing tensors must use int16 and float16")
    if quant_scales.dtype != torch.float32:
        raise TypeError("MoeExpert quant_scales must use float32")
    if quant_offsets is not None and quant_offsets.dtype != torch.int32:
        raise TypeError("MoeExpert quant_offsets must use int32")
    _check_finite("MoeExpert", topk_weight, quant_scales)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("MoeExpert x must be a non-empty rank-2 tensor")
    expected_route_shape = (x.shape[0], 3)
    if tuple(topk_ids.shape) != expected_route_shape or tuple(
        topk_weight.shape
    ) != expected_route_shape:
        raise ValueError("MoeExpert routing shape must be [tokenNum, 3]")
    if expert_weights.ndim != 1:
        raise ValueError("MoeExpert expert_weights must be packed rank 1")
    if tuple(quant_scales.shape) != (_MOE_SCALE_COUNT,):
        raise ValueError("MoeExpert requires exactly 21 ordered scales")
    if quant_offsets is not None and tuple(quant_offsets.shape) != (_MOE_SCALE_COUNT,):
        raise ValueError("MoeExpert offsets must match the 21-scale order")
    denominator = _MOE_EXPERT_COUNT * 3 * x.shape[1]
    if expert_weights.numel() == 0 or expert_weights.numel() % denominator:
        raise ValueError("Packed expert weight length is invalid")
    if _can_read_values(quant_scales) and bool((quant_scales <= 0).any()):
        raise ValueError("MoeExpert quant_scales must be positive")
    if _can_read_values(topk_ids):
        routed_ids = topk_ids[:, :2]
        if bool((routed_ids < 0).any()) or bool(
            (routed_ids >= _MOE_EXPERT_COUNT - 1).any()
        ):
            raise ValueError("MoeExpert routed id is outside [0, 4)")
        if bool((routed_ids[:, 0] == routed_ids[:, 1]).any()):
            raise ValueError("MoeExpert routed ids must be unique per token")
        if not bool((topk_ids[:, 2] == _MOE_EXPERT_COUNT - 1).all()):
            raise ValueError("MoeExpert shared id 4 must appear last")
        routed_weights = topk_weight[:, :2].float()
        if bool((topk_weight < 0).any()) or not torch.allclose(
            routed_weights.sum(dim=1),
            torch.ones(x.shape[0], device=x.device),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError(
                "MoeExpert routed weights must be non-negative and sum to one"
            )
        if not bool((topk_weight[:, 2] == 1).all()):
            raise ValueError("MoeExpert shared weight must equal one")
    return torch.ops.mdc_llm_deploy.moe_expert.default(
        x,
        topk_ids,
        topk_weight,
        expert_weights,
        quant_scales,
        quant_offsets,
    )


_REFERENCE_KERNELS: dict[str, Kernel] = {
    "rms_norm": _rms_norm_reference,
    "apply_rotary_pos_emb": _apply_rotary_pos_emb_reference,
    "fused_infer_attention_score": _fused_infer_attention_score_reference,
    "ascend_quant_v2": _ascend_quant_v2_reference,
    "ascend_dequant": _ascend_dequant_reference,
    "moe_expert": _moe_expert_reference,
}

_META_KERNELS: dict[str, Kernel] = {
    "rms_norm": _rms_norm_meta,
    "apply_rotary_pos_emb": _apply_rotary_pos_emb_meta,
    "fused_infer_attention_score": _fused_infer_attention_score_meta,
    "ascend_quant_v2": _ascend_quant_v2_meta,
    "ascend_dequant": _ascend_dequant_meta,
    "moe_expert": _moe_expert_meta,
}


def _npu_is_available() -> bool:
    backend = getattr(torch, "npu", None)
    return bool(backend is not None and backend.is_available())


def _register_kernels(
    library: torch.library.Library,
    dispatch_key: str,
    kernels: dict[str, Kernel],
) -> None:
    for name, kernel in kernels.items():
        library.impl(name, kernel, dispatch_key)


_DEFINITION_LIBRARY = torch.library.Library(TORCH_NAMESPACE, "DEF")
for _schema in OPERATOR_SCHEMAS.values():
    _DEFINITION_LIBRARY.define(_schema.torch_schema)

_IMPLEMENTATION_LIBRARY = torch.library.Library(TORCH_NAMESPACE, "IMPL")
_register_kernels(_IMPLEMENTATION_LIBRARY, "CPU", _REFERENCE_KERNELS)
_register_kernels(_IMPLEMENTATION_LIBRARY, "Meta", _META_KERNELS)

_REGISTERED_DEVICE_DISPATCHES = ["CPU", "Meta"]
if torch.cuda.is_available():
    _register_kernels(_IMPLEMENTATION_LIBRARY, "CUDA", _REFERENCE_KERNELS)
    _REGISTERED_DEVICE_DISPATCHES.append("CUDA")
if _npu_is_available():
    _register_kernels(_IMPLEMENTATION_LIBRARY, "PrivateUse1", _REFERENCE_KERNELS)
    _REGISTERED_DEVICE_DISPATCHES.append("PrivateUse1")

REGISTERED_DEVICE_DISPATCHES = tuple(_REGISTERED_DEVICE_DISPATCHES)


def registered_device_dispatches() -> tuple[str, ...]:
    """Return dispatches registered in the current runtime."""
    return REGISTERED_DEVICE_DISPATCHES


def operator_schemas() -> Iterable[Any]:
    """Return immutable operator schema values."""
    return OPERATOR_SCHEMAS.values()
