"""Deterministic RmsNorm hardware case."""

from __future__ import annotations

from pathlib import Path

import torch

from mdc_llm_deploy.custom_ops.rms_norm import rms_norm

from .common import CaseDefinition, generate_case, seeded_generator

_EPSILON = 1e-5


class _GoldenModel(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rstd = torch.rsqrt(torch.mean(x.square(), dim=-1) + _EPSILON)
        return x * rstd.unsqueeze(-1) * gamma, rstd


class _CustomModel(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rms_norm(x, gamma, _EPSILON)


def case_definition() -> CaseDefinition:
    """Build one trailing-dimension normalization case."""
    generator = seeded_generator(103)
    return CaseDefinition(
        name="rms_norm",
        golden_model=_GoldenModel(),
        custom_model=_CustomModel(),
        inputs={
            "x": torch.randn(2, 3, 16, generator=generator),
            "gamma": torch.randn(16, generator=generator),
        },
        output_names=("y", "rstd"),
        description="FLOAT32、末维归一化的确定性 RMSNorm 用例。",
        operator_names=("rms_norm",),
    )


def generate(output_root: Path) -> Path:
    """Generate RmsNorm artifacts."""
    return generate_case(case_definition(), output_root)
