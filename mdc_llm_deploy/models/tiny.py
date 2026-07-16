"""Deterministic export-friendly Tiny Qwen3 model family."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from .attention import TinyAttention as TinyAttention
from .feedforward import (
    TinyMlp as TinyMlp,
)
from .feedforward import (
    TinyMoe as TinyMoe,
)
from .layers import (
    RmsNorm as RmsNorm,
)
from .layers import (
    RotaryEmbedding as RotaryEmbedding,
)
from .types import (
    PREFILL_SEQUENCE_LENGTH as PREFILL_SEQUENCE_LENGTH,
)
from .types import (
    VOCAB_SIZE as VOCAB_SIZE,
)
from .types import (
    TinyConfig as TinyConfig,
)
from .types import (
    TinyOutput as TinyOutput,
)

INITIALIZATION_SEED = 20260714
PREFILL_BATCH_SIZE = 1


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
