"""Eager and FakeTensor kernels for broad MoeExpert execution."""

from __future__ import annotations

import torch

from .contract import (
    validate_mdc_contract,
    validate_routing,
    validate_torch_contract,
)


def _unpack_weights(
    expert_weights: torch.Tensor,
    quant_scales: torch.Tensor | None,
    quant_offsets: torch.Tensor | None,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = expert_weights.float()
    gate_end = hidden_size * intermediate_size
    up_end = 2 * gate_end
    gate = weights[:, :gate_end].reshape(-1, intermediate_size, hidden_size)
    up = weights[:, gate_end:up_end].reshape(-1, intermediate_size, hidden_size)
    down = weights[:, up_end:].reshape(-1, hidden_size, intermediate_size)
    if expert_weights.dtype != torch.int8:
        return gate, up, down

    assert quant_scales is not None
    scales = quant_scales.float()
    offsets = (
        torch.zeros_like(scales) if quant_offsets is None else quant_offsets.float()
    )
    gate_scale, up_scale, down_scale = torch.split(
        scales, [intermediate_size, intermediate_size, hidden_size], dim=1
    )
    gate_offset, up_offset, down_offset = torch.split(
        offsets, [intermediate_size, intermediate_size, hidden_size], dim=1
    )
    return (
        (gate - gate_offset.unsqueeze(-1)) * gate_scale.unsqueeze(-1),
        (up - up_offset.unsqueeze(-1)) * up_scale.unsqueeze(-1),
        (down - down_offset.unsqueeze(-1)) * down_scale.unsqueeze(-1),
    )


def _execute_general(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_weights: torch.Tensor,
    quant_scales: torch.Tensor | None,
    quant_offsets: torch.Tensor | None,
    *,
    device_type: str,
) -> torch.Tensor:
    token_count, hidden_size, top_k, expert_count, intermediate_size = (
        validate_torch_contract(
            x, topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets
        )
    )
    if x.device.type != device_type:
        raise ValueError(f"MoeExpert kernel requires {device_type} tensors")
    validate_routing(topk_ids, topk_weight, expert_count)
    gate, up, down = _unpack_weights(
        expert_weights, quant_scales, quant_offsets, hidden_size, intermediate_size
    )

    x_fp32 = x.float()
    output = torch.zeros(
        (token_count, hidden_size), dtype=torch.float32, device=x.device
    )
    for route in range(top_k):
        expert_ids = topk_ids[:, route].long()
        gate_output = torch.bmm(
            gate.index_select(0, expert_ids), x_fp32.unsqueeze(-1)
        ).squeeze(-1)
        up_output = torch.bmm(
            up.index_select(0, expert_ids), x_fp32.unsqueeze(-1)
        ).squeeze(-1)
        activated = torch.nn.functional.silu(gate_output) * up_output
        expert_output = torch.bmm(
            down.index_select(0, expert_ids), activated.unsqueeze(-1)
        ).squeeze(-1)
        output.add_(expert_output * topk_weight[:, route].float().unsqueeze(-1))
    return output.to(x.dtype)


def _execute_mdc(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_weights: torch.Tensor,
    quant_scales: torch.Tensor | None,
    quant_offsets: torch.Tensor | None,
    *,
    device_type: str,
) -> torch.Tensor:
    token_count, hidden_size, top_k, expert_count, intermediate_size = (
        validate_mdc_contract(
            x, topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets
        )
    )
    if x.device.type != device_type:
        raise ValueError(f"MoeExpert kernel requires {device_type} tensors")
    assert quant_scales is not None
    validate_routing(topk_ids, topk_weight, expert_count)
    scales = quant_scales.float()
    if not bool(torch.all(torch.isfinite(scales)).item()):
        raise ValueError("MDC quant_scales must contain only finite values")
    if bool(torch.any(scales <= 0).item()):
        raise ValueError("MDC quant_scales must be positive")

    gate, up, down = expert_weights.reshape(
        expert_count, 3, intermediate_size, hidden_size
    ).unbind(dim=1)
    token_scale = scales[0]
    expert_scales = scales[1:].reshape(expert_count, 4)
    x_fp32 = x.float()
    output = torch.zeros(
        (token_count, hidden_size), dtype=torch.float32, device=x.device
    )
    for route in range(top_k):
        ids = topk_ids[:, route].long()
        selected_scales = expert_scales.index_select(0, ids)
        gate_output = torch.bmm(
            gate.index_select(0, ids).float(), x_fp32.unsqueeze(-1)
        ).squeeze(-1)
        up_output = torch.bmm(
            up.index_select(0, ids).float(), x_fp32.unsqueeze(-1)
        ).squeeze(-1)
        gate_output *= (token_scale * selected_scales[:, 0]).unsqueeze(-1)
        up_output *= (token_scale * selected_scales[:, 1]).unsqueeze(-1)
        activated = torch.nn.functional.silu(gate_output) * up_output
        activated = torch.clamp(
            torch.round(activated / selected_scales[:, 2].unsqueeze(-1)),
            -128,
            127,
        )
        expert_output = torch.bmm(
            activated.unsqueeze(1), down.index_select(0, ids).float()
        ).squeeze(1)
        expert_output *= (
            selected_scales[:, 2] * selected_scales[:, 3]
        ).unsqueeze(-1)
        output.add_(expert_output * topk_weight[:, route].float().unsqueeze(-1))
    return output.to(torch.float16)


def cpu(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_weights: torch.Tensor,
    quant_scales: torch.Tensor | None = None,
    quant_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute MoeExpert on CPU."""
    if x.dtype == torch.int8:
        return _execute_mdc(
            x,
            topk_ids,
            topk_weight,
            expert_weights,
            quant_scales,
            quant_offsets,
            device_type="cpu",
        )
    return _execute_general(
        x,
        topk_ids,
        topk_weight,
        expert_weights,
        quant_scales,
        quant_offsets,
        device_type="cpu",
    )


def cuda(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_weights: torch.Tensor,
    quant_scales: torch.Tensor | None = None,
    quant_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute broad MoeExpert semantics with CUDA tensor operations."""
    if x.dtype == torch.int8:
        return _execute_mdc(
            x,
            topk_ids,
            topk_weight,
            expert_weights,
            quant_scales,
            quant_offsets,
            device_type="cuda",
        )
    return _execute_general(
        x,
        topk_ids,
        topk_weight,
        expert_weights,
        quant_scales,
        quant_offsets,
        device_type="cuda",
    )


def fake(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_weights: torch.Tensor,
    quant_scales: torch.Tensor | None = None,
    quant_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Infer output metadata without reading tensor values."""
    if x.dtype == torch.int8:
        validate_mdc_contract(
            x, topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets
        )
        return torch.empty(x.shape, dtype=torch.float16, device=x.device)
    validate_torch_contract(
        x, topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets
    )
    return torch.empty_like(x)
