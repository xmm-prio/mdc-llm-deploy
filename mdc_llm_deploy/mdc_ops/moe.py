"""Generic expert-major MoE validation and reference kernels."""
# mypy: disable-error-code="no-any-return"

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

from ._runtime_validation import (
    can_read_values,
    check_finite,
    check_no_autograd,
    check_same_device,
)

_PROJECTION_COUNT = 3


def _projection_matrices(
    packed: Tensor,
    hidden_size: int,
    scales: Tensor | None,
    offsets: Tensor | None,
    expert_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    packed_width = packed.numel()
    denominator = _PROJECTION_COUNT * hidden_size
    if packed_width == 0 or packed_width % denominator:
        raise ValueError("Packed expert width is invalid")
    intermediate_size = packed_width // denominator
    lengths = hidden_size * intermediate_size
    matrices: list[Tensor] = []
    for projection_id, shape in enumerate(
        (
            (intermediate_size, hidden_size),
            (intermediate_size, hidden_size),
            (hidden_size, intermediate_size),
        )
    ):
        start = projection_id * lengths
        matrix = packed[start : start + lengths].reshape(shape)
        if matrix.dtype == torch.int8:
            if scales is None:
                raise ValueError("INT8 expert weights require quant_scales")
            parameter_index = expert_id * _PROJECTION_COUNT + projection_id
            offset = (
                torch.zeros((), dtype=torch.int32, device=matrix.device)
                if offsets is None
                else offsets.reshape(-1)[parameter_index]
            )
            matrix = (
                matrix.float() - offset.float()
            ) * scales.reshape(-1)[parameter_index].float()
        else:
            matrix = matrix.float()
        matrices.append(matrix)
    return matrices[0], matrices[1], matrices[2]


def moe_expert_reference(
    x: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    expert_weights: Tensor,
    quant_scales: Tensor | None = None,
    quant_offsets: Tensor | None = None,
) -> Tensor:
    """Execute expert-major packed weights with PyTorch operations."""
    hidden = x.float()
    output = torch.zeros_like(hidden)
    for expert_id in range(expert_weights.shape[0]):
        gate, up, down = _projection_matrices(
            expert_weights[expert_id],
            x.shape[-1],
            quant_scales,
            quant_offsets,
            expert_id,
        )
        expert_output = functional.silu(hidden @ gate.t()) * (hidden @ up.t())
        expert_output = expert_output @ down.t()
        route_weight = (
            (topk_ids == expert_id).to(topk_weight.dtype) * topk_weight
        ).sum(dim=-1, keepdim=True)
        output = output + expert_output * route_weight.float()
    return output.to(x.dtype)


def moe_expert_meta(
    x: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    expert_weights: Tensor,
    quant_scales: Tensor | None = None,
    quant_offsets: Tensor | None = None,
) -> Tensor:
    """Return metadata matching the activation input."""
    del topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets
    return torch.empty_like(x)


def _validate_quantization(
    expert_weights: Tensor,
    quant_scales: Tensor | None,
    quant_offsets: Tensor | None,
) -> None:
    expert_count = expert_weights.shape[0]
    expected = expert_count * _PROJECTION_COUNT
    if expert_weights.dtype == torch.int8:
        if quant_scales is None:
            raise ValueError("INT8 expert weights require quant_scales")
        if quant_scales.dtype not in {torch.float16, torch.float32}:
            raise TypeError("MoeExpert quant_scales must be floating point")
        if quant_scales.numel() != expected:
            raise ValueError(
                f"MoeExpert quant_scales must contain {expected} values"
            )
        if can_read_values(quant_scales) and bool((quant_scales <= 0).any()):
            raise ValueError("MoeExpert quant_scales must be positive")
        if quant_offsets is not None:
            if quant_offsets.dtype != torch.int32:
                raise TypeError("MoeExpert quant_offsets must use int32")
            if quant_offsets.numel() != expected:
                raise ValueError(
                    f"MoeExpert quant_offsets must contain {expected} values"
                )
    elif not expert_weights.dtype.is_floating_point:
        raise TypeError("MoeExpert weights must be floating point or int8")
    elif quant_scales is not None or quant_offsets is not None:
        raise ValueError("Floating-point expert weights do not use quant parameters")


def moe_expert(
    x: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    expert_weights: Tensor,
    quant_scales: Tensor | None = None,
    quant_offsets: Tensor | None = None,
) -> Tensor:
    """Execute a validated inference-only expert-major MoE."""
    values = (
        x,
        topk_ids,
        topk_weight,
        expert_weights,
        quant_scales,
        quant_offsets,
    )
    check_no_autograd(*values)
    check_same_device("MoeExpert", *values)
    if not x.dtype.is_floating_point:
        raise TypeError("MoeExpert activations must be floating point")
    if topk_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("MoeExpert topk_ids must use int32 or int64")
    if topk_weight.dtype != x.dtype:
        raise TypeError("MoeExpert topk_weight must match activation dtype")
    check_finite("MoeExpert", x, topk_weight, quant_scales)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("MoeExpert x must be a non-empty rank-2 tensor")
    if expert_weights.ndim != 2 or expert_weights.shape[0] == 0:
        raise ValueError("expert_weights must be non-empty expert-major rank 2")
    route_shape = tuple(topk_ids.shape)
    if (
        len(route_shape) != 2
        or route_shape[0] != x.shape[0]
        or route_shape[1] == 0
        or tuple(topk_weight.shape) != route_shape
    ):
        raise ValueError("Routing tensors must have shape [token_count, top_k]")
    expert_count = expert_weights.shape[0]
    if route_shape[1] > expert_count:
        raise ValueError("Routing top_k cannot exceed expert_count")
    packed_denominator = _PROJECTION_COUNT * x.shape[1]
    if expert_weights.shape[1] % packed_denominator:
        raise ValueError("Packed expert width is invalid")
    _validate_quantization(expert_weights, quant_scales, quant_offsets)
    if can_read_values(topk_ids):
        if bool((topk_ids < 0).any()) or bool((topk_ids >= expert_count).any()):
            raise ValueError("MoeExpert routing id is outside expert range")
        sorted_ids = torch.sort(topk_ids, dim=-1).values
        if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
            raise ValueError("MoeExpert routing ids must be unique per token")
    if can_read_values(topk_weight) and (
        bool((topk_weight < 0).any())
        or not torch.allclose(
            topk_weight.float().sum(dim=-1),
            torch.ones(x.shape[0], device=x.device),
            atol=1e-3,
            rtol=1e-3,
        )
    ):
        raise ValueError("Routing weights must be non-negative and sum to one")
    return torch.ops.mdc_llm_deploy.moe_expert.default(
        x,
        topk_ids,
        topk_weight,
        expert_weights,
        quant_scales,
        quant_offsets,
    )
