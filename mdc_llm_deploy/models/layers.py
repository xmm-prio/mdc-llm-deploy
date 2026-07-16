"""Reusable normalization and rotary layers for Tiny models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .types import TinyConfig


class RmsNorm(nn.Module):
    """RMS normalization with FP32 accumulation."""

    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the last dimension."""
        variance = x.float().square().mean(
            dim=-1,
            keepdim=True,
        )
        normalized = x * torch.rsqrt(
            variance + self.epsilon
        ).to(x.dtype)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    """Qwen-style half-rotation embedding."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(
                    0,
                    config.head_dim,
                    2,
                    dtype=torch.float32,
                )
                / config.head_dim
            )
        )
        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=True,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply rotary embedding to BSND query and key tensors."""
        frequencies = torch.einsum(
            "bs,d->bsd",
            position_ids.float(),
            self.inv_freq.float(),
        )
        embedding = torch.cat(
            (frequencies, frequencies),
            dim=-1,
        )
        cos = embedding.cos().unsqueeze(2).to(query.dtype)
        sin = embedding.sin().unsqueeze(2).to(query.dtype)
        return (
            query * cos + self._rotate_half(query) * sin,
            key * cos + self._rotate_half(key) * sin,
        )

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)
