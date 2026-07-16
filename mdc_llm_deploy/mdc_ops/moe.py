"""Tiny MoE validation, reference kernel, and meta kernel."""
# mypy: disable-error-code="no-any-return"

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

from ..moe_layout import DEFAULT_MOE_LAYOUT, Projection
from ._runtime_validation import (
    can_read_values,
    check_finite,
    check_no_autograd,
    check_same_device,
)


def moe_expert_reference(
    x: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    expert_weights: Tensor,
    quant_scales: Tensor,
    quant_offsets: Tensor | None = None,
) -> Tensor:
    """Execute the fixed Tiny MoE ABI with PyTorch reference operations."""
    offsets = (
        torch.zeros_like(quant_scales, dtype=torch.int32)
        if quant_offsets is None
        else quant_offsets
    )
    hidden = (x.float() - offsets[0].float()) * quant_scales[0]
    hidden_size = x.shape[1]
    intermediate_size = expert_weights.numel() // (
        DEFAULT_MOE_LAYOUT.packed_projection_count * hidden_size
    )
    output = torch.zeros_like(hidden)
    segments = DEFAULT_MOE_LAYOUT.weight_segments(
        hidden_size,
        intermediate_size,
    )
    gate_projection, up_projection = (
        DEFAULT_MOE_LAYOUT.input_activation_projections
    )
    for expert_id in range(DEFAULT_MOE_LAYOUT.expert_count):
        matrices: dict[Projection, Tensor] = {}
        for segment in (
            item for item in segments if item.expert_id == expert_id
        ):
            scale_index = DEFAULT_MOE_LAYOUT.scale_index(
                expert_id,
                DEFAULT_MOE_LAYOUT.quant_slot_for_projection(
                    segment.projection
                ),
            )
            packed = expert_weights[
                segment.offset : segment.offset + segment.length
            ].view(segment.rows, segment.columns)
            matrix = (
                packed.float() - offsets[scale_index].float()
            ) * quant_scales[scale_index]
            matrices[segment.projection] = matrix
        gate = matrices[gate_projection]
        up = matrices[up_projection]
        down = matrices[DEFAULT_MOE_LAYOUT.output_projection]
        intermediate = functional.silu(hidden @ gate.t()) * (hidden @ up.t())
        activation_index = DEFAULT_MOE_LAYOUT.scale_index(
            expert_id,
            "intermediate",
        )
        activation_scale = quant_scales[activation_index]
        activation_offset = offsets[activation_index].float()
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


def moe_expert_meta(
    x: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    expert_weights: Tensor,
    quant_scales: Tensor,
    quant_offsets: Tensor | None = None,
) -> Tensor:
    """Return output metadata for the fixed Tiny MoE ABI."""
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
    """Execute packed Tiny MoE using the fixed deployment ABI."""
    values = (x, topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets)
    check_no_autograd(*values)
    check_same_device("MoeExpert", *values)
    if x.dtype != torch.int8 or expert_weights.dtype != torch.int8:
        raise TypeError("MoeExpert activations and weights must use int8")
    if topk_ids.dtype != torch.int16 or topk_weight.dtype != torch.float16:
        raise TypeError("MoeExpert routing tensors must use int16 and float16")
    if quant_scales.dtype != torch.float32:
        raise TypeError("MoeExpert quant_scales must use float32")
    if quant_offsets is not None and quant_offsets.dtype != torch.int32:
        raise TypeError("MoeExpert quant_offsets must use int32")
    check_finite("MoeExpert", topk_weight, quant_scales)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("MoeExpert x must be a non-empty rank-2 tensor")
    expected_route_shape = (x.shape[0], DEFAULT_MOE_LAYOUT.route_width)
    if tuple(topk_ids.shape) != expected_route_shape or tuple(
        topk_weight.shape
    ) != expected_route_shape:
        raise ValueError(
            f"MoeExpert routing shape must be [tokenNum, {DEFAULT_MOE_LAYOUT.route_width}]"
        )
    if expert_weights.ndim != 1:
        raise ValueError("MoeExpert expert_weights must be packed rank 1")
    if tuple(quant_scales.shape) != (DEFAULT_MOE_LAYOUT.quant_parameter_count,):
        raise ValueError(
            "MoeExpert requires exactly "
            f"{DEFAULT_MOE_LAYOUT.quant_parameter_count} ordered scales"
        )
    if quant_offsets is not None and tuple(quant_offsets.shape) != (
        DEFAULT_MOE_LAYOUT.quant_parameter_count,
    ):
        raise ValueError(
            "MoeExpert offsets must match the "
            f"{DEFAULT_MOE_LAYOUT.quant_parameter_count}-scale order"
        )
    denominator = DEFAULT_MOE_LAYOUT.packed_projection_count * x.shape[1]
    if expert_weights.numel() == 0 or expert_weights.numel() % denominator:
        raise ValueError("Packed expert weight length is invalid")
    if can_read_values(quant_scales) and bool((quant_scales <= 0).any()):
        raise ValueError("MoeExpert quant_scales must be positive")
    if can_read_values(topk_ids):
        routed_ids = topk_ids[:, : DEFAULT_MOE_LAYOUT.routed_top_k]
        if bool((routed_ids < 0).any()) or bool(
            (routed_ids >= DEFAULT_MOE_LAYOUT.routed_expert_count).any()
        ):
            raise ValueError(
                "MoeExpert routed id is outside "
                f"[0, {DEFAULT_MOE_LAYOUT.routed_expert_count})"
            )
        sorted_ids = torch.sort(routed_ids, dim=1).values
        if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
            raise ValueError("MoeExpert routed ids must be unique per token")
        if not bool(
            (
                topk_ids[:, DEFAULT_MOE_LAYOUT.routed_top_k]
                == DEFAULT_MOE_LAYOUT.shared_expert_id
            ).all()
        ):
            raise ValueError(
                f"MoeExpert shared id {DEFAULT_MOE_LAYOUT.shared_expert_id} "
                "must appear last"
            )
        routed_weights = topk_weight[
            :, : DEFAULT_MOE_LAYOUT.routed_top_k
        ].float()
        if bool((topk_weight < 0).any()) or not torch.allclose(
            routed_weights.sum(dim=1),
            torch.ones(x.shape[0], device=x.device),
            atol=1e-3,
            rtol=1e-3,
        ):
            raise ValueError(
                "MoeExpert routed weights must be non-negative and sum to one"
            )
        if not bool(
            (topk_weight[:, DEFAULT_MOE_LAYOUT.routed_top_k] == 1).all()
        ):
            raise ValueError("MoeExpert shared weight must equal one")
    return torch.ops.mdc_llm_deploy.moe_expert.default(
        x,
        topk_ids,
        topk_weight,
        expert_weights,
        quant_scales,
        quant_offsets,
    )
