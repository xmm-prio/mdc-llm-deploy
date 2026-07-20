"""Deterministic floating GQA attention hardware case."""

from __future__ import annotations

from pathlib import Path

import torch

from mdc_llm_deploy.custom_ops.fused_infer_attention_score import (
    FusedInferAttentionScore,
)
from mdc_llm_deploy.custom_ops.registry import register_custom_op

from .common import CaseDefinition, generate_case, seeded_generator

_SCALE = 0.5


class _GoldenModel(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        atten_mask: torch.Tensor,
        actual_seq_lengths: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = key.repeat_interleave(2, dim=1)
        value = value.repeat_interleave(2, dim=1)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * _SCALE
        query_positions = torch.arange(query.shape[2])
        key_positions = torch.arange(key.shape[2])
        active = query_positions.unsqueeze(0) < actual_seq_lengths.unsqueeze(1)
        visible = key_positions.unsqueeze(0) < actual_seq_lengths_kv.unsqueeze(1)
        visible = visible[:, None, None, :] & ~atten_mask.to(torch.bool)
        scores = scores.masked_fill(~visible, -torch.inf)
        scores = torch.where(active[:, None, :, None], scores, torch.zeros_like(scores))
        output = torch.matmul(torch.softmax(scores, dim=-1), value.float())
        output = torch.where(active[:, None, :, None], output, torch.zeros_like(output))
        return output.to(query.dtype), torch.zeros(1, dtype=torch.float32)


class _CustomModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._operator = register_custom_op(FusedInferAttentionScore).definition

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        atten_mask: torch.Tensor,
        actual_seq_lengths: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._operator(
            query,
            key,
            value,
            None,
            atten_mask,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            num_heads=4,
            scale=_SCALE,
            input_layout="BNSD",
            num_key_value_heads=2,
        )


def case_definition() -> CaseDefinition:
    """Build legal masked GQA case with per-batch effective lengths."""
    generator = seeded_generator(107)
    mask = torch.zeros(2, 1, 3, 5, dtype=torch.bool)
    mask[:, :, :, -1] = True
    return CaseDefinition(
        name="fused_infer_attention_score",
        golden_model=_GoldenModel(),
        custom_model=_CustomModel(),
        inputs={
            "query": torch.randn(
                2, 4, 3, 8, generator=generator, dtype=torch.float16
            ),
            "key": torch.randn(
                2, 2, 5, 8, generator=generator, dtype=torch.float16
            ),
            "value": torch.randn(
                2, 2, 5, 8, generator=generator, dtype=torch.float16
            ),
            "atten_mask": mask,
            "actual_seq_lengths": torch.tensor([3, 2], dtype=torch.int64),
            "actual_seq_lengths_kv": torch.tensor([5, 4], dtype=torch.int64),
        },
        output_names=("attention_out", "softmax_lse"),
        description="BNSD FP16 GQA、广播 mask、有效序列长度的确定性 Attention 用例。",
    )


def generate(output_root: Path) -> Path:
    """Generate FusedInferAttentionScore artifacts."""
    return generate_case(case_definition(), output_root)
