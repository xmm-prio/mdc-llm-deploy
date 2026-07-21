from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import onnx
import torch
from torch.onnx import ONNXProgram
from transformers import (
    PreTrainedModel,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)
from transformers.exporters import OnnxConfig, OnnxExporter

_BATCH_SIZE = 1
_PREFILL_LENGTH = 3
_VOCAB_SIZE = 32
_HIDDEN_SIZE = 32
_NUM_ATTENTION_HEADS = 4
_NUM_KEY_VALUE_HEADS = 2
_MAX_POSITION_EMBEDDINGS = 32


class Qwen3Family(Enum):
    DENSE_4B = "qwen3-4b"
    MOE_30B_A3B = "qwen3-30b-a3b"


class AttentionBackend(Enum):
    EAGER = "eager"
    SDPA = "sdpa"


@dataclass(frozen=True)
class Qwen3ExportCase:
    family: Qwen3Family
    attention_backend: AttentionBackend
    dtype: torch.dtype

    @property
    def id(self) -> str:
        dtype_name = str(self.dtype).removeprefix("torch.")
        return f"{self.family.value}-{self.attention_backend.value}-{dtype_name}"


DTYPES = (torch.float16, torch.bfloat16, torch.float32)
EXPORT_CASES = tuple(
    Qwen3ExportCase(family, attention_backend, dtype)
    for family in Qwen3Family
    for attention_backend in AttentionBackend
    for dtype in DTYPES
)


def onnx_export_config() -> OnnxConfig:
    return OnnxConfig(
        opset_version=18,
        optimize=True,
        dynamic=False,
        external_data=False,
    )


def build_qwen3_model(case: Qwen3ExportCase, *, use_cache: bool) -> PreTrainedModel:
    torch.manual_seed(0)
    common_config: dict[str, Any] = {
        "vocab_size": _VOCAB_SIZE,
        "hidden_size": _HIDDEN_SIZE,
        "num_hidden_layers": 1,
        "num_attention_heads": _NUM_ATTENTION_HEADS,
        "num_key_value_heads": _NUM_KEY_VALUE_HEADS,
        "max_position_embeddings": _MAX_POSITION_EMBEDDINGS,
        "use_cache": use_cache,
        "pad_token_id": 0,
        "eos_token_id": _VOCAB_SIZE - 1,
        "dtype": case.dtype,
    }
    if case.family is Qwen3Family.DENSE_4B:
        config = Qwen3Config(
            **common_config,
            intermediate_size=64,
            head_dim=_HIDDEN_SIZE // _NUM_ATTENTION_HEADS,
        )
        model: PreTrainedModel = Qwen3ForCausalLM(config)
    else:
        config = Qwen3MoeConfig(
            **common_config,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_experts=4,
            num_experts_per_tok=2,
        )
        model = Qwen3MoeForCausalLM(config)

    model.set_attn_implementation(case.attention_backend.value)
    return model.eval().to(dtype=case.dtype)


def prefill_inputs() -> dict[str, torch.Tensor | bool]:
    return {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((_BATCH_SIZE, _PREFILL_LENGTH), dtype=torch.long),
        "position_ids": torch.arange(_PREFILL_LENGTH, dtype=torch.long).unsqueeze(0),
        "use_cache": False,
    }


def generation_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((_BATCH_SIZE, _PREFILL_LENGTH), dtype=torch.long),
    }


def export_static_prefill(case: Qwen3ExportCase) -> onnx.ModelProto:
    model = build_qwen3_model(case, use_cache=False)
    program = OnnxExporter().export(model, prefill_inputs(), onnx_export_config())
    return program.model_proto


def export_static_generation(case: Qwen3ExportCase) -> Mapping[str, onnx.ModelProto]:
    model = build_qwen3_model(case, use_cache=True)
    programs = OnnxExporter().export_for_generation(
        model,
        generation_inputs(),
        onnx_export_config(),
    )
    return {
        component_name: _as_model_proto(program)
        for component_name, program in programs.items()
    }


def _as_model_proto(program: object) -> onnx.ModelProto:
    if not isinstance(program, ONNXProgram):
        raise TypeError(f"Expected ONNXProgram, got {type(program).__name__}")
    return program.model_proto
