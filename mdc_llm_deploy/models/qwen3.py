"""Inference-only Qwen3 Dense and MoE model implementations."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from ..mdc_ops import moe_expert
from .configuration import ExportModelConfig, Qwen3Config, Qwen3MoeConfig


class _TransformersModelAdapter:
    """Expose the top-level device and dtype semantics used by Transformers."""

    @property
    def device(self) -> torch.device:
        """Return the device of the first model parameter."""
        return next(cast(nn.Module, self).parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the first model parameter."""
        return next(cast(nn.Module, self).parameters()).dtype


class Qwen3RMSNorm(nn.Module):
    """Qwen3 RMS normalization with FP32 accumulation."""

    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize the final dimension."""
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.epsilon).to(
            hidden_states.dtype
        )
        return normalized * self.weight


def _rotate_half(value: Tensor) -> Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Qwen3RotaryEmbedding(nn.Module):
    """Apply precomputed Qwen3 half-rotation values."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_mdc_rotary", torch.ones((), dtype=torch.bool))

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Rotate query and key tensors."""
        return (
            query * cos + _rotate_half(query) * sin,
            key * cos + _rotate_half(key) * sin,
        )


class Qwen3Attention(nn.Module):
    """Grouped-query attention using precomputed rotary and mask buffers."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        query_size = config.num_attention_heads * config.head_dim
        key_value_size = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            query_size,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            key_value_size,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            key_value_size,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(query_size, config.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(config.head_dim, config.rms_norm_eps)
        self.rotary = Qwen3RotaryEmbedding()

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute attention and return BNSD key/value caches."""
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
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary(query, key, cos, sin)
        query = query.transpose(1, 2)
        key_cache = key.transpose(1, 2)
        value_cache = value.transpose(1, 2)
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        expanded_key = key_cache.repeat_interleave(groups, dim=1)
        expanded_value = value_cache.repeat_interleave(groups, dim=1)
        scores = torch.matmul(
            query.float(),
            expanded_key.float().transpose(-2, -1),
        ) * (1.0 / math.sqrt(self.config.head_dim))
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1).to(hidden_states.dtype)
        output = torch.matmul(probabilities, expanded_value)
        output = output.transpose(1, 2).reshape(batch, sequence, -1)
        return self.o_proj(output), key_cache, value_cache


class Qwen3MLP(nn.Module):
    """Qwen3 SiLU-gated feed-forward block."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply gated feed-forward projections."""
        return cast(
            Tensor,
            self.down_proj(
                functional.silu(self.gate_proj(hidden_states))
                * self.up_proj(hidden_states)
            ),
        )


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Route tokens into expert-major packed Qwen3 experts."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        packed_width = (
            3 * config.hidden_size * config.moe_intermediate_size
        )
        self.expert_weights = nn.Parameter(
            torch.empty(config.num_experts, packed_width)
        )
        self.quant_scales: nn.Parameter | None
        self.quant_offsets: nn.Parameter | None
        self.register_parameter("quant_scales", None)
        self.register_parameter("quant_offsets", None)

    def set_packed_weights(
        self,
        weights: Tensor,
        *,
        scales: Tensor | None = None,
        offsets: Tensor | None = None,
    ) -> None:
        """Install validated floating-point or INT8 packed expert weights."""
        expected_shape = self.expert_weights.shape
        if weights.shape != expected_shape:
            raise ValueError(
                f"Packed expert weights must have shape {tuple(expected_shape)}"
            )
        parameter_count = self.config.num_experts * 3
        if weights.dtype == torch.int8:
            if scales is None or scales.numel() != parameter_count:
                raise ValueError(
                    f"INT8 expert weights require {parameter_count} scales"
                )
            if offsets is not None and offsets.numel() != parameter_count:
                raise ValueError(
                    f"INT8 expert weights require {parameter_count} offsets"
                )
        elif scales is not None or offsets is not None:
            raise ValueError(
                "Floating-point expert weights do not use quant parameters"
            )
        self.expert_weights = nn.Parameter(
            weights.detach().clone(),
            requires_grad=False,
        )
        self.quant_scales = (
            None
            if scales is None
            else nn.Parameter(
                scales.detach().clone().float(),
                requires_grad=False,
            )
        )
        self.quant_offsets = (
            None
            if offsets is None
            else nn.Parameter(
                offsets.detach().clone().to(torch.int32),
                requires_grad=False,
            )
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply normalized top-k routing and packed experts."""
        original_shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, self.config.hidden_size)
        router_logits = self.gate(flattened).float()
        routing = torch.softmax(router_logits, dim=-1)
        topk_weight, topk_ids = torch.topk(
            routing,
            self.config.num_experts_per_tok,
            dim=-1,
        )
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        output = moe_expert(
            flattened,
            topk_ids,
            topk_weight.to(flattened.dtype),
            self.expert_weights,
            self.quant_scales,
            self.quant_offsets,
        )
        return output.reshape(original_shape)


class Qwen3DecoderLayer(nn.Module):
    """One Qwen3 decoder layer."""

    def __init__(self, config: Qwen3Config, *, moe: bool) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        if moe:
            if not isinstance(config, Qwen3MoeConfig):
                raise TypeError("MoE decoder layer requires Qwen3MoeConfig")
            self.mlp: nn.Module = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply attention and feed-forward residual blocks."""
        residual = hidden_states
        attention, key, value = self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            attention_mask,
        )
        hidden_states = residual + attention
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, key, value


class Qwen3Model(nn.Module):
    """Shared Qwen3 decoder body preserving official parameter FQNs."""

    def __init__(self, config: Qwen3Config, *, moe: bool) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, moe=moe)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)


class _Qwen3ForCausalLM(_TransformersModelAdapter, nn.Module):
    model_kind = "dense"

    def __init__(
        self,
        config: Qwen3Config,
        export_config: ExportModelConfig,
        *,
        dtype: torch.dtype = torch.float16,
        moe: bool,
    ) -> None:
        super().__init__()
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        if export_config.sequence_length > config.max_position_embeddings:
            raise ValueError("Export sequence exceeds max_position_embeddings")
        self.config = config
        self.export_config = export_config
        self.model = Qwen3Model(config, moe=moe)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        positions = torch.arange(export_config.sequence_length, dtype=torch.float32)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / config.head_dim
            )
        )
        frequencies = torch.outer(positions, inv_freq)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer(
            "cos_cache",
            embedding.cos().view(1, export_config.sequence_length, 1, config.head_dim),
            persistent=True,
        )
        self.register_buffer(
            "sin_cache",
            embedding.sin().view(1, export_config.sequence_length, 1, config.head_dim),
            persistent=True,
        )
        mask = (
            torch.ones(
                export_config.sequence_length,
                export_config.sequence_length,
                dtype=torch.bool,
            ).tril().view(
                1,
                1,
                export_config.sequence_length,
                export_config.sequence_length,
            )
            if export_config.mask_mode == "causal"
            else None
        )
        self.register_buffer("causal_mask", mask, persistent=True)
        generator = torch.Generator(device="cpu").manual_seed(20260716)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.normal_(
                        mean=0.0,
                        std=config.initializer_range,
                        generator=generator,
                    )
                elif isinstance(module, Qwen3RMSNorm):
                    module.weight.fill_(1.0)
            for module in self.modules():
                if isinstance(module, Qwen3MoeSparseMoeBlock):
                    module.expert_weights.normal_(
                        mean=0.0,
                        std=config.initializer_range,
                        generator=generator,
                    )
        self.tie_weights()
        self.to(dtype=dtype)
        self.requires_grad_(False)
        self.eval()

    def tie_weights(self) -> None:
        """Restore configured input/output embedding parameter aliases."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: Tensor) -> tuple[Tensor, ...]:
        """Run a fixed-length prefill and return logits plus per-layer KV."""
        if input_ids.dtype != torch.int64 or input_ids.ndim != 2:
            raise TypeError("input_ids must be a rank-2 int64 tensor")
        if input_ids.shape != (1, self.export_config.sequence_length):
            raise ValueError("input_ids shape does not match ExportModelConfig")
        hidden_states = self.model.embed_tokens(input_ids)
        cos = self.cos_cache.to(hidden_states.dtype)
        sin = self.sin_cache.to(hidden_states.dtype)
        outputs: list[Tensor] = []
        for layer in self.model.layers:
            hidden_states, key, value = layer(
                hidden_states,
                cos,
                sin,
                self.causal_mask,
            )
            outputs.extend((key, value))
        logits = self.lm_head(self.model.norm(hidden_states))
        return (logits, *outputs)


class Qwen3ForCausalLM(_Qwen3ForCausalLM):
    """Export-specialized Qwen3 Dense causal language model."""

    model_kind = "dense"

    def __init__(
        self,
        config: Qwen3Config,
        export_config: ExportModelConfig,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__(config, export_config, dtype=dtype, moe=False)


class Qwen3MoeForCausalLM(_Qwen3ForCausalLM):
    """Export-specialized Qwen3-MoE causal language model."""

    model_kind = "moe"

    def __init__(
        self,
        config: Qwen3MoeConfig,
        export_config: ExportModelConfig,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__(config, export_config, dtype=dtype, moe=True)


__all__ = [
    "Qwen3Attention",
    "Qwen3ForCausalLM",
    "Qwen3MLP",
    "Qwen3Model",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeSparseMoeBlock",
    "Qwen3RMSNorm",
    "Qwen3RotaryEmbedding",
]
