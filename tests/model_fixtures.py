"""Small real Qwen3 model constructors used by tests."""

from __future__ import annotations

from typing import Literal

import torch

from mdc_llm_deploy.models import (
    ExportModelConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)


def dense_config(*, layers: int = 1) -> Qwen3Config:
    """Return a compact Qwen3 Dense configuration."""
    return Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
    )


def moe_config(
    *,
    layers: int = 1,
    expert_count: int = 4,
    top_k: int = 2,
) -> Qwen3MoeConfig:
    """Return a compact Qwen3-MoE configuration."""
    return Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        moe_intermediate_size=32,
        num_experts=expert_count,
        num_experts_per_tok=top_k,
    )


def dense_model(
    sequence_length: int = 8,
    *,
    layers: int = 1,
    mask_mode: Literal["causal", "none"] = "causal",
) -> Qwen3ForCausalLM:
    """Construct a compact initialized Dense model."""
    return Qwen3ForCausalLM(
        dense_config(layers=layers),
        ExportModelConfig(sequence_length, mask_mode=mask_mode),
        dtype=torch.float32,
    )


def moe_model(
    sequence_length: int = 8,
    *,
    layers: int = 1,
    expert_count: int = 4,
    top_k: int = 2,
) -> Qwen3MoeForCausalLM:
    """Construct a compact initialized MoE model."""
    return Qwen3MoeForCausalLM(
        moe_config(
            layers=layers,
            expert_count=expert_count,
            top_k=top_k,
        ),
        ExportModelConfig(sequence_length),
        dtype=torch.float32,
    )
