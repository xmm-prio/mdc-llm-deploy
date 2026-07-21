"""Broad Torch contracts for MoeExpert execution."""

from __future__ import annotations

import torch


def validate_torch_contract(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_weights: torch.Tensor,
    quant_scales: torch.Tensor | None,
    quant_offsets: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    """Validate floating-token and conventional INT8-weight execution."""
    tensors = [x, topk_ids, topk_weight, expert_weights]
    tensors.extend(
        tensor for tensor in (quant_scales, quant_offsets) if tensor is not None
    )
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("MoeExpert inputs must be on the same device")
    if x.ndim != 2 or not x.dtype.is_floating_point:
        raise TypeError("x must be a floating-point rank-2 tensor")
    if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be an INT32 or INT64 rank-2 tensor")
    if topk_weight.ndim != 2 or topk_weight.dtype != x.dtype:
        raise TypeError("topk_weight must be rank-2 and have the same dtype as x")
    if topk_ids.shape != topk_weight.shape or topk_ids.shape[0] != x.shape[0]:
        raise ValueError("routing tensors must have shape [token_count, top_k]")
    if topk_ids.shape[1] <= 0:
        raise ValueError("top_k must be positive")
    if expert_weights.ndim != 2 or expert_weights.shape[0] <= 0:
        raise ValueError("expert_weights must have shape [expert_count, packed_width]")

    token_count, hidden_size = x.shape
    expert_count, packed_width = expert_weights.shape
    divisor = 3 * hidden_size
    if hidden_size <= 0 or packed_width <= 0 or packed_width % divisor:
        raise ValueError(
            "expert_weights packed width must equal 3 * hidden_size * intermediate_size"
        )
    intermediate_size = packed_width // divisor
    expected_quant_shape = (expert_count, 2 * intermediate_size + hidden_size)
    if expert_weights.dtype == torch.int8:
        if quant_scales is None:
            raise ValueError("INT8 expert_weights require quant_scales")
        if quant_scales.shape != expected_quant_shape or not quant_scales.dtype.is_floating_point:
            raise ValueError(
                "quant_scales must be floating-point with shape "
                "[expert_count, 2 * intermediate_size + hidden_size]"
            )
        if quant_offsets is not None and (
            quant_offsets.shape != expected_quant_shape
            or not quant_offsets.dtype.is_floating_point
        ):
            raise ValueError(
                "quant_offsets must match quant_scales shape and be floating-point"
            )
    elif expert_weights.dtype.is_floating_point:
        if expert_weights.dtype != x.dtype:
            raise TypeError("floating expert_weights must have the same dtype as x")
        if quant_scales is not None or quant_offsets is not None:
            raise ValueError("floating expert_weights must not use quantization parameters")
    else:
        raise TypeError("expert_weights must be floating-point or INT8")
    return (
        int(token_count),
        int(hidden_size),
        int(topk_ids.shape[1]),
        int(expert_count),
        int(intermediate_size),
    )


def validate_mdc_contract(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_weights: torch.Tensor,
    quant_scales: torch.Tensor | None,
    quant_offsets: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    """Validate fully quantized MDC execution without narrowing Torch schema."""
    tensors = [x, topk_ids, topk_weight, expert_weights]
    if quant_scales is not None:
        tensors.append(quant_scales)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("MoeExpert inputs must be on the same device")
    if x.ndim != 2 or x.dtype != torch.int8:
        raise TypeError("MDC x must be an INT8 rank-2 tensor")
    if topk_ids.ndim != 2 or topk_ids.dtype != torch.int16:
        raise TypeError("MDC topk_ids must be an INT16 rank-2 tensor")
    if topk_weight.shape != topk_ids.shape or topk_weight.dtype != torch.float16:
        raise TypeError("MDC topk_weight must be FLOAT16 and match topk_ids shape")
    if topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] <= 0:
        raise ValueError("MDC routing shape must be [token_count, positive top_k]")
    if expert_weights.ndim != 2 or expert_weights.dtype != torch.int8:
        raise TypeError("MDC expert_weights must be an INT8 rank-2 tensor")
    if quant_scales is None:
        raise ValueError("MDC quant_scales is required")
    if quant_scales.ndim != 1 or quant_scales.dtype != torch.float32:
        raise TypeError("MDC quant_scales must be a FLOAT32 rank-1 tensor")
    if quant_offsets is not None:
        raise ValueError("MDC quant_offsets is unsupported and must be omitted")

    scale_count = quant_scales.shape[0]
    if scale_count < 5 or (scale_count - 1) % 4:
        raise ValueError("MDC quant_scales length must equal 1 + 4 * expert_count")
    expert_count = (scale_count - 1) // 4
    token_count, hidden_size = x.shape
    packed_rows, weight_hidden_size = expert_weights.shape
    if weight_hidden_size != hidden_size or packed_rows % (3 * expert_count):
        raise ValueError("MDC expert_weights shape must be [3 * E * I, H]")
    intermediate_size = packed_rows // (3 * expert_count)
    if hidden_size <= 0 or hidden_size % 256:
        raise ValueError("MDC hidden_size must be a positive multiple of 256")
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("MDC intermediate_size must be a positive multiple of 128")
    return (
        int(token_count),
        int(hidden_size),
        int(topk_ids.shape[1]),
        int(expert_count),
        int(intermediate_size),
    )


def validate_routing(
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_count: int,
) -> None:
    """Validate routing values read by eager kernels."""
    if bool(torch.any((topk_ids < 0) | (topk_ids >= expert_count)).item()):
        raise ValueError("topk_ids contains an out-of-range expert id")
    if topk_ids.shape[1] > 1:
        sorted_ids = torch.sort(topk_ids, dim=1).values
        if bool(torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]).item()):
            raise ValueError("topk_ids must not repeat an expert for one token")
    weights = topk_weight.float()
    if not bool(torch.all(torch.isfinite(weights)).item()):
        raise ValueError("topk_weight must contain only finite values")
    if bool(torch.any(weights < 0).item()):
        raise ValueError("topk_weight must be non-negative")
    if not bool(
        torch.allclose(
            weights.sum(dim=1),
            torch.ones(weights.shape[0], device=weights.device),
            rtol=1e-4,
            atol=1e-5,
        )
    ):
        raise ValueError("each topk_weight row must sum to one")
