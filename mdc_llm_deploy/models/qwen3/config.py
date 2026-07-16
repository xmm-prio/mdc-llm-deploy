"""Configuration contracts for export-specialized Qwen3 models."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, TypeVar

ConfigT = TypeVar("ConfigT", bound="Qwen3Config")


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    """Normalized subset of Qwen3 architecture configuration."""

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    use_qk_norm: bool = True
    hidden_act: str = "silu"
    model_type: str = "qwen3"

    def __post_init__(self) -> None:
        integer_fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.hidden_act != "silu":
            raise ValueError("Only silu hidden activation is supported")

    @classmethod
    def from_dict(cls: type[ConfigT], value: dict[str, Any]) -> ConfigT:
        """Normalize a Transformers-compatible configuration mapping."""
        names = {item.name for item in fields(cls)}
        selected = {name: item for name, item in value.items() if name in names}
        return cls(**selected)


@dataclass(frozen=True, slots=True)
class Qwen3MoeConfig(Qwen3Config):
    """Normalized Qwen3-MoE architecture configuration."""

    moe_intermediate_size: int = 768
    num_experts: int = 128
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    model_type: str = "qwen3_moe"

    def __post_init__(self) -> None:
        Qwen3Config.__post_init__(self)
        if type(self.moe_intermediate_size) is not int or self.moe_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size must be a positive integer")
        if type(self.num_experts) is not int or self.num_experts <= 0:
            raise ValueError("num_experts must be a positive integer")
        if (
            type(self.num_experts_per_tok) is not int
            or not 0 < self.num_experts_per_tok <= self.num_experts
        ):
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")


__all__ = [
    "Qwen3Config",
    "Qwen3MoeConfig",
]
