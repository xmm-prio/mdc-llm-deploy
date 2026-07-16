"""Grouped-query attention block for Tiny Qwen3 models."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .layers import RotaryEmbedding
from .types import TinyConfig


class TinyAttention(nn.Module):
    """Single-layer grouped-query causal attention."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        kv_size = config.num_key_value_heads * config.head_dim
        self.k_proj = nn.Linear(
            config.hidden_size,
            kv_size,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            kv_size,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.rotary = RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute attention and expose BNSD cache boundaries."""
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch,
            sequence,
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        key = self.k_proj(hidden_states).view(
            batch,
            sequence,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            batch,
            sequence,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        query, key = self.rotary(query, key, position_ids)
        query = query.transpose(1, 2)
        key_cache = key.transpose(1, 2)
        value_cache = value.transpose(1, 2)
        groups = (
            self.config.num_attention_heads
            // self.config.num_key_value_heads
        )
        expanded_key = key_cache.repeat_interleave(
            groups,
            dim=1,
        )
        expanded_value = value_cache.repeat_interleave(
            groups,
            dim=1,
        )
        if attention_mask is None:
            mask = torch.ones(
                sequence,
                sequence,
                dtype=torch.bool,
                device=hidden_states.device,
            ).tril()
            attention_mask = mask.view(
                1,
                1,
                sequence,
                sequence,
            )
        scores = torch.matmul(
            query.float(),
            expanded_key.float().transpose(-2, -1),
        )
        scores = scores * (
            1.0 / math.sqrt(self.config.head_dim)
        )
        scores = scores.masked_fill(
            ~attention_mask,
            torch.finfo(scores.dtype).min,
        )
        probabilities = torch.softmax(
            scores,
            dim=-1,
        ).to(hidden_states.dtype)
        output = torch.matmul(probabilities, expanded_value)
        output = output.transpose(1, 2).reshape(
            batch,
            sequence,
            -1,
        )
        return self.o_proj(output), key_cache, value_cache
