"""Deterministic ApplyRotaryPosEmb hardware case."""

from __future__ import annotations

from pathlib import Path

import torch

from mdc_llm_deploy.custom_ops.apply_rotary_pos_emb import apply_rotary_pos_emb

from .common import CaseDefinition, generate_case, seeded_generator


class _GoldenModel(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_first, query_second = query.chunk(2, dim=-1)
        key_first, key_second = key.chunk(2, dim=-1)
        query_rotated = torch.cat((-query_second, query_first), dim=-1)
        key_rotated = torch.cat((-key_second, key_first), dim=-1)
        return query * cos + query_rotated * sin, key * cos + key_rotated * sin


class _CustomModel(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_pos_emb(query, key, cos, sin, 1, "half")


def case_definition() -> CaseDefinition:
    """Build BSND, half-rotation case with GQA head counts."""
    generator = seeded_generator(101)
    angles = torch.randn(1, 3, 1, 8, generator=generator, dtype=torch.float32)
    return CaseDefinition(
        name="apply_rotary_pos_emb",
        golden_model=_GoldenModel(),
        custom_model=_CustomModel(),
        inputs={
            "query": torch.randn(1, 3, 4, 8, generator=generator),
            "key": torch.randn(1, 3, 2, 8, generator=generator),
            "cos": angles.cos(),
            "sin": angles.sin(),
        },
        output_names=("query_out", "key_out"),
        description="BSND、half 模式、GQA head 数的确定性 RoPE 用例。",
        operator_names=("apply_rotary_pos_emb",),
    )


def generate(output_root: Path) -> Path:
    """Generate ApplyRotaryPosEmb artifacts."""
    return generate_case(case_definition(), output_root)
