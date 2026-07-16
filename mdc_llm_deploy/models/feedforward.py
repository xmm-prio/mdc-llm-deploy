"""Dense and routed feed-forward blocks for Tiny models."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .types import TinyConfig


class TinyMlp(nn.Module):
    """Qwen-style gated MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply SiLU-gated projection."""
        return cast(
            Tensor,
            self.down_proj(
                functional.silu(self.gate_proj(x))
                * self.up_proj(x)
            ),
        )


class TinyMoe(nn.Module):
    """Four routed experts plus one unconditional shared expert."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False,
        )
        self.experts = nn.ModuleList(
            [
                TinyMlp(
                    config.hidden_size,
                    config.moe_intermediate_size,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.shared_expert = TinyMlp(
            config.hidden_size,
            config.moe_intermediate_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Route top-2 experts and add shared expert output."""
        router_probabilities = torch.softmax(
            self.router(x).float(),
            dim=-1,
        )
        top_weights, top_ids = torch.topk(
            router_probabilities,
            self.config.num_experts_per_tok,
            dim=-1,
        )
        top_weights = top_weights / top_weights.sum(
            dim=-1,
            keepdim=True,
        )
        routed = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            weight = (
                (top_ids == expert_id).to(top_weights.dtype)
                * top_weights
            ).sum(dim=-1, keepdim=True)
            routed = routed + expert(x) * weight.to(x.dtype)
        return cast(Tensor, routed + self.shared_expert(x))
