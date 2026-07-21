"""Deterministic fully quantized MDC MoeExpert hardware case."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from mdc_llm_deploy.custom_ops.moe_expert import moe_expert

from .common import CaseDefinition, generate_case, seeded_generator

_TOKEN_COUNT = 1
_HIDDEN_SIZE = 256
_INTERMEDIATE_SIZE = 256
_EXPERT_COUNT = 4
_TOP_K = 2


def _mdc_golden(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_weights: torch.Tensor,
    quant_scales: torch.Tensor,
) -> torch.Tensor:
    matrices = expert_weights.reshape(
        _EXPERT_COUNT, 3, _INTERMEDIATE_SIZE, _HIDDEN_SIZE
    )
    gate, up, down = matrices.unbind(dim=1)
    token_scale = quant_scales[0].float()
    expert_scales = quant_scales[1:].reshape(_EXPERT_COUNT, 4).float()
    output = torch.zeros((_TOKEN_COUNT, _HIDDEN_SIZE), dtype=torch.float32)
    for route in range(_TOP_K):
        ids = topk_ids[:, route].long()
        scales = expert_scales.index_select(0, ids)
        gate_output = torch.bmm(
            gate.index_select(0, ids).float(), x.float().unsqueeze(-1)
        ).squeeze(-1)
        up_output = torch.bmm(
            up.index_select(0, ids).float(), x.float().unsqueeze(-1)
        ).squeeze(-1)
        gate_output *= (token_scale * scales[:, 0]).unsqueeze(-1)
        up_output *= (token_scale * scales[:, 1]).unsqueeze(-1)
        activated = torch.nn.functional.silu(gate_output) * up_output
        activated = torch.clamp(
            torch.round(activated / scales[:, 2].unsqueeze(-1)), -128, 127
        )
        expert_output = torch.bmm(
            activated.unsqueeze(1), down.index_select(0, ids).float()
        ).squeeze(1)
        expert_output *= (scales[:, 2] * scales[:, 3]).unsqueeze(-1)
        output += expert_output * topk_weight[:, route].float().unsqueeze(-1)
    return output.to(torch.float16)


class _GoldenModel(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
        quant_scales: torch.Tensor,
    ) -> torch.Tensor:
        return _mdc_golden(x, topk_ids, topk_weight, expert_weights, quant_scales)


class _CustomModel(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
        quant_scales: torch.Tensor,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            moe_expert(
                x,
                topk_ids,
                topk_weight,
                expert_weights,
                quant_scales,
            ),
        )


def case_definition() -> CaseDefinition:
    """Build the real five-input fully quantized MDC ABI case."""
    generator = seeded_generator(113)
    expert_scales = torch.tensor(
        [
            [0.020, 0.018, 0.0050, 0.021],
            [0.017, 0.023, 0.0045, 0.019],
            [0.022, 0.016, 0.0055, 0.024],
            [0.019, 0.021, 0.0040, 0.018],
        ],
        dtype=torch.float32,
    )
    return CaseDefinition(
        name="moe_expert_int8",
        golden_model=_GoldenModel(),
        custom_model=_CustomModel(),
        inputs={
            "x": torch.randint(
                -8,
                8,
                (_TOKEN_COUNT, _HIDDEN_SIZE),
                generator=generator,
                dtype=torch.int8,
            ),
            "topk_ids": torch.tensor([[1, 3]], dtype=torch.int16),
            "topk_weight": torch.tensor([[0.375, 0.625]], dtype=torch.float16),
            "expert_weights": torch.randint(
                -8,
                8,
                (3 * _EXPERT_COUNT * _INTERMEDIATE_SIZE, _HIDDEN_SIZE),
                generator=generator,
                dtype=torch.int8,
            ),
            "quant_scales": torch.cat(
                [torch.tensor([0.025], dtype=torch.float32), expert_scales.flatten()]
            ),
        },
        output_names=("out",),
        description=(
            "MDC 全量化 ABI: INT8 token/权重、INT16 top-k、FP16 routing、"
            "1+4E FP32 scales; 无 quant_offsets 输入。"
        ),
        operator_names=("moe_expert",),
    )


def generate(output_root: Path) -> Path:
    """Generate the real MDC MoeExpert artifact."""
    return generate_case(case_definition(), output_root)
