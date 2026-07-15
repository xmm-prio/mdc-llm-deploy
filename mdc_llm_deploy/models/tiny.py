"""Deterministic export-friendly Tiny Qwen3 model family."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

INITIALIZATION_SEED = 20260714
PREFILL_BATCH_SIZE = 1
PREFILL_SEQUENCE_LENGTH = 3072
VOCAB_SIZE = 128


@dataclass(frozen=True, slots=True)
class TinyConfig:
    """Frozen Tiny Qwen3-compatible architecture configuration."""

    vocab_size: int = VOCAB_SIZE
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 1
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    max_position_embeddings: int = PREFILL_SEQUENCE_LENGTH
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    use_cache: bool = True
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    embedding_dropout: float = 0.0
    num_experts: int = 4
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 64
    num_shared_experts: int = 1
    model_type: str = "qwen3"
    _attn_implementation: str = "eager"


class TinyOutput(NamedTuple):
    """Static model output."""

    logits: Tensor
    key_cache: Tensor
    value_cache: Tensor


class RmsNorm(nn.Module):
    """RMS normalization with FP32 accumulation."""

    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the last dimension."""
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.epsilon).to(x.dtype)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    """Qwen-style half-rotation embedding."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / config.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, query: Tensor, key: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Apply rotary embedding to BSND query and key tensors."""
        frequencies = torch.einsum(
            "bs,d->bsd", position_ids.float(), self.inv_freq.float()
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
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


class TinyAttention(nn.Module):
    """Single-layer grouped-query causal attention."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        kv_size = config.num_key_value_heads * config.head_dim
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
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
            batch, sequence, self.config.num_attention_heads, self.config.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch, sequence, self.config.num_key_value_heads, self.config.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, sequence, self.config.num_key_value_heads, self.config.head_dim
        )
        query, key = self.rotary(query, key, position_ids)
        query = query.transpose(1, 2)
        key_cache = key.transpose(1, 2)
        value_cache = value.transpose(1, 2)
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        expanded_key = key_cache.repeat_interleave(groups, dim=1)
        expanded_value = value_cache.repeat_interleave(groups, dim=1)
        if attention_mask is None:
            mask = torch.ones(
                sequence,
                sequence,
                dtype=torch.bool,
                device=hidden_states.device,
            ).tril()
            attention_mask = mask.view(1, 1, sequence, sequence)
        scores = torch.matmul(query.float(), expanded_key.float().transpose(-2, -1))
        scores = scores * (1.0 / math.sqrt(self.config.head_dim))
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1).to(hidden_states.dtype)
        output = torch.matmul(probabilities, expanded_value)
        output = output.transpose(1, 2).reshape(batch, sequence, -1)
        return self.o_proj(output), key_cache, value_cache


class TinyMlp(nn.Module):
    """Qwen-style gated MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SiLU-gated projection."""
        return cast(
            Tensor,
            self.down_proj(functional.silu(self.gate_proj(x)) * self.up_proj(x)),
        )


class TinyMoe(nn.Module):
    """Four routed experts plus one unconditional shared expert."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                TinyMlp(config.hidden_size, config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.shared_expert = TinyMlp(
            config.hidden_size, config.moe_intermediate_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """Route top-2 experts and add shared expert output."""
        router_probabilities = torch.softmax(self.router(x).float(), dim=-1)
        top_weights, top_ids = torch.topk(
            router_probabilities, self.config.num_experts_per_tok, dim=-1
        )
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        routed = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            weight = (
                (top_ids == expert_id).to(top_weights.dtype) * top_weights
            ).sum(dim=-1, keepdim=True)
            routed = routed + expert(x) * weight.to(x.dtype)
        return cast(Tensor, routed + self.shared_expert(x))


class _TinyBase(nn.Module):
    model_kind = "dense"

    def __init__(
        self,
        config: TinyConfig | None = None,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch dtype")
        self.config = config or TinyConfig()
        self._parameter_dtype = dtype
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        self.input_norm = RmsNorm(
            self.config.hidden_size, self.config.rms_norm_eps
        )
        self.self_attn = TinyAttention(self.config)
        self.post_attention_norm = RmsNorm(
            self.config.hidden_size, self.config.rms_norm_eps
        )
        self.final_norm = RmsNorm(
            self.config.hidden_size, self.config.rms_norm_eps
        )
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

    def _initialize(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(INITIALIZATION_SEED)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.normal_(
                        mean=0.0,
                        std=self.config.initializer_range,
                        generator=generator,
                    )
                elif isinstance(module, RmsNorm):
                    module.weight.fill_(1.0)
            for parameter in self.parameters():
                parameter.data = parameter.data.to(dtype=self._parameter_dtype)
        self.eval()

    def _body(self, hidden_states: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, input_ids: Tensor) -> TinyOutput:
        """Run one static prefill pass."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.int64:
            raise TypeError("input_ids must use torch.int64")
        if input_ids.shape[0] != PREFILL_BATCH_SIZE:
            raise ValueError("input_ids batch size must be 1")
        if input_ids.shape[1] < 1 or input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError(
                f"input_ids sequence length must be in [1, {self.config.max_position_embeddings}]"
            )
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        attention, key_cache, value_cache = self.self_attn(
            self.input_norm(hidden_states), position_ids
        )
        hidden_states = hidden_states + attention
        hidden_states = hidden_states + self._body(
            self.post_attention_norm(hidden_states)
        )
        logits = self.lm_head(self.final_norm(hidden_states))
        return TinyOutput(logits, key_cache, value_cache)


class TinyQwen3Dense(_TinyBase):
    """Deterministic one-layer Tiny Qwen3 Dense model."""

    model_kind = "dense"

    def __init__(
        self,
        config: TinyConfig | None = None,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        caller_rng_state = torch.random.get_rng_state()
        try:
            super().__init__(config, dtype=dtype)
            self.mlp = TinyMlp(
                self.config.hidden_size, self.config.intermediate_size
            )
            self._initialize()
        finally:
            torch.random.set_rng_state(caller_rng_state)

    def _body(self, hidden_states: Tensor) -> Tensor:
        return cast(Tensor, self.mlp(hidden_states))


class TinyQwen3Moe(_TinyBase):
    """Deterministic one-layer Tiny Qwen3-MoE model."""

    model_kind = "moe"

    def __init__(
        self,
        config: TinyConfig | None = None,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        caller_rng_state = torch.random.get_rng_state()
        try:
            super().__init__(config, dtype=dtype)
            self.moe = TinyMoe(self.config)
            self._initialize()
        finally:
            torch.random.set_rng_state(caller_rng_state)

    def _body(self, hidden_states: Tensor) -> Tensor:
        return cast(Tensor, self.moe(hidden_states))
