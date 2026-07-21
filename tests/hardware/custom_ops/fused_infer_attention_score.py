"""Deterministic MC62 floating decode attention hardware case."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from mdc_llm_deploy.custom_ops.fused_infer_attention_score import (
    fused_infer_attention_score,
)

from .common import CaseDefinition, generate_case, seeded_generator

_HEADS = 8
_KV_SEQUENCE = 16
_HEAD_DIM = 64
_SCALE = 1.0 / math.sqrt(_HEAD_DIM)


class _GoldenModel(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * _SCALE
        output = torch.matmul(torch.softmax(scores, dim=-1), value.float())
        return output.to(query.dtype)


class _CustomModel(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        attention_out, _ = fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=_HEADS,
            scale=_SCALE,
            input_layout="BNSD",
            num_key_value_heads=_HEADS,
        )
        return attention_out


def case_definition() -> CaseDefinition:
    """Build the reference-compatible MC62 float decode MHA case."""
    generator = seeded_generator(107)
    return CaseDefinition(
        name="fused_infer_attention_score",
        golden_model=_GoldenModel(),
        custom_model=_CustomModel(),
        inputs={
            "query": torch.randn(
                1, _HEADS, 1, _HEAD_DIM, generator=generator, dtype=torch.float16
            ),
            "key": torch.randn(
                1,
                _HEADS,
                _KV_SEQUENCE,
                _HEAD_DIM,
                generator=generator,
                dtype=torch.float16,
            ),
            "value": torch.randn(
                1,
                _HEADS,
                _KV_SEQUENCE,
                _HEAD_DIM,
                generator=generator,
                dtype=torch.float16,
            ),
        },
        output_names=("attention_out",),
        description=(
            "MC62 BNSD FP16 Decode MHA: B=1、heads=8、query seq=1、"
            "KV seq=16、head dim=64, 无可选输入。"
        ),
        operator_names=("fused_infer_attention_score",),
    )


def generate(output_root: Path) -> Path:
    """Generate FusedInferAttentionScore artifacts."""
    return generate_case(case_definition(), output_root)
