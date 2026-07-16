"""Pure data contracts for the Tiny Qwen3 model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from torch import Tensor

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
