"""Automatic Qwen3 export-model construction."""

from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import load_config, load_model_state, resolve_checkpoint
from .export_config import ExportModelConfig
from .qwen3 import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)


class AutoExportModel:
    """Construct the supported export model declared by a checkpoint."""

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        export_config: ExportModelConfig,
        *,
        dtype: torch.dtype = torch.float16,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> Qwen3ForCausalLM | Qwen3MoeForCausalLM:
        """Resolve, construct, and load a Qwen3 checkpoint."""
        directory = resolve_checkpoint(
            source,
            revision=revision,
            local_files_only=local_files_only,
        )
        raw_config = load_config(directory)
        model_type = raw_config.get("model_type")
        architectures = raw_config.get("architectures", ())
        is_moe = model_type in {"qwen3_moe", "qwen3-moe"} or any(
            "Moe" in str(name) for name in architectures
        )
        if is_moe:
            model: Qwen3ForCausalLM | Qwen3MoeForCausalLM = (
                Qwen3MoeForCausalLM(
                    Qwen3MoeConfig.from_dict(raw_config),
                    export_config,
                    dtype=dtype,
                )
            )
        elif model_type == "qwen3":
            model = Qwen3ForCausalLM(
                Qwen3Config.from_dict(raw_config),
                export_config,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported checkpoint model_type: {model_type!r}")
        load_model_state(model, directory)
        return model


__all__ = ["AutoExportModel"]
