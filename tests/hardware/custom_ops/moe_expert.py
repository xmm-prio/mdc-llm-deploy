"""Deterministic floating-point and INT8 MoeExpert hardware cases."""

from __future__ import annotations

from pathlib import Path

import torch

from mdc_llm_deploy.custom_ops.moe_expert import MoeExpert
from mdc_llm_deploy.custom_ops.registry import register_custom_op

from .common import CaseDefinition, generate_case, seeded_generator

_HIDDEN_SIZE = 8
_INTERMEDIATE_SIZE = 16
_EXPERT_COUNT = 3


def _moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_weights: torch.Tensor,
    quant_scales: torch.Tensor | None = None,
    quant_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = expert_weights.float()
    gate_end = _HIDDEN_SIZE * _INTERMEDIATE_SIZE
    up_end = 2 * gate_end
    gate = weights[:, :gate_end].reshape(-1, _INTERMEDIATE_SIZE, _HIDDEN_SIZE)
    up = weights[:, gate_end:up_end].reshape(-1, _INTERMEDIATE_SIZE, _HIDDEN_SIZE)
    down = weights[:, up_end:].reshape(-1, _HIDDEN_SIZE, _INTERMEDIATE_SIZE)
    if quant_scales is not None:
        offsets = torch.zeros_like(quant_scales) if quant_offsets is None else quant_offsets
        gate_scale, up_scale, down_scale = torch.split(
            quant_scales, [_INTERMEDIATE_SIZE, _INTERMEDIATE_SIZE, _HIDDEN_SIZE], dim=1
        )
        gate_offset, up_offset, down_offset = torch.split(
            offsets, [_INTERMEDIATE_SIZE, _INTERMEDIATE_SIZE, _HIDDEN_SIZE], dim=1
        )
        gate = (gate - gate_offset.unsqueeze(-1)) * gate_scale.unsqueeze(-1)
        up = (up - up_offset.unsqueeze(-1)) * up_scale.unsqueeze(-1)
        down = (down - down_offset.unsqueeze(-1)) * down_scale.unsqueeze(-1)

    output = torch.zeros_like(x)
    for route in range(topk_ids.shape[1]):
        ids = topk_ids[:, route].long()
        gate_output = torch.bmm(gate.index_select(0, ids), x.unsqueeze(-1)).squeeze(-1)
        up_output = torch.bmm(up.index_select(0, ids), x.unsqueeze(-1)).squeeze(-1)
        activated = torch.nn.functional.silu(gate_output) * up_output
        expert_output = torch.bmm(
            down.index_select(0, ids), activated.unsqueeze(-1)
        ).squeeze(-1)
        output = output + expert_output * topk_weight[:, route].unsqueeze(-1)
    return output


class _GoldenFloatModel(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _moe(x, topk_ids, topk_weight, expert_weights)


class _GoldenInt8Model(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
        quant_scales: torch.Tensor,
        quant_offsets: torch.Tensor,
    ) -> torch.Tensor:
        return _moe(
            x, topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets
        )


class _CustomModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._operator = register_custom_op(MoeExpert).definition

    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._operator(x, topk_ids, topk_weight, expert_weights)


class _CustomInt8Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._operator = register_custom_op(MoeExpert).definition

    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
        quant_scales: torch.Tensor,
        quant_offsets: torch.Tensor,
    ) -> torch.Tensor:
        return self._operator(
            x,
            topk_ids,
            topk_weight,
            expert_weights,
            quant_scales,
            quant_offsets,
        )


def _routing_inputs() -> dict[str, torch.Tensor]:
    return {
        "topk_ids": torch.tensor(
            [[0, 1], [1, 2], [2, 0], [1, 0]], dtype=torch.int64
        ),
        "topk_weight": torch.tensor(
            [[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.9, 0.1]],
            dtype=torch.float32,
        ),
    }


def case_definitions() -> tuple[CaseDefinition, CaseDefinition]:
    """Build floating-point and asymmetric INT8 packed-weight cases."""
    float_generator = seeded_generator(109)
    int8_generator = seeded_generator(113)
    packed_width = 3 * _HIDDEN_SIZE * _INTERMEDIATE_SIZE
    scale_width = 2 * _INTERMEDIATE_SIZE + _HIDDEN_SIZE
    routing = _routing_inputs()
    float_case = CaseDefinition(
        name="moe_expert_float",
        golden_model=_GoldenFloatModel(),
        custom_model=_CustomModel(),
        inputs={
            "x": torch.randn(4, _HIDDEN_SIZE, generator=float_generator),
            **routing,
            "expert_weights": torch.randn(
                _EXPERT_COUNT, packed_width, generator=float_generator
            ),
        },
        output_names=("out",),
        description="浮点 packed 权重、top-2 routing 的确定性 MoeExpert 用例。",
    )
    int8_case = CaseDefinition(
        name="moe_expert_int8",
        golden_model=_GoldenInt8Model(),
        custom_model=_CustomInt8Model(),
        inputs={
            "x": torch.randn(4, _HIDDEN_SIZE, generator=int8_generator),
            **routing,
            "expert_weights": torch.randint(
                -8,
                8,
                (_EXPERT_COUNT, packed_width),
                generator=int8_generator,
                dtype=torch.int8,
            ),
            "quant_scales": torch.rand(
                _EXPERT_COUNT, scale_width, generator=int8_generator
            )
            * 0.05
            + 0.01,
            "quant_offsets": torch.randint(
                -2,
                3,
                (_EXPERT_COUNT, scale_width),
                generator=int8_generator,
            ).float(),
        },
        output_names=("out",),
        description="INT8 packed 权重、非对称 per-channel 反量化的确定性 MoeExpert 用例。",
    )
    return float_case, int8_case


def generate(output_root: Path) -> tuple[Path, Path]:
    """Generate both MoeExpert artifacts."""
    return tuple(
        generate_case(definition, output_root) for definition in case_definitions()
    )
