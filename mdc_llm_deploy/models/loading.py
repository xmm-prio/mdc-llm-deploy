"""Checkpoint location and safetensors loading for Qwen3 export models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .qwen3 import (
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
)

ExportModel = Qwen3ForCausalLM | Qwen3MoeForCausalLM


def resolve_checkpoint(
    source: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local checkpoint directory or Hugging Face repository."""
    candidate = Path(source)
    if candidate.is_dir():
        return candidate.resolve()
    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=str(source),
        revision=revision,
        allow_patterns=["*.json", "*.safetensors"],
        local_files_only=local_files_only,
    )
    return Path(downloaded)


def load_config(directory: Path) -> dict[str, Any]:
    """Load checkpoint config.json."""
    path = directory / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Checkpoint config.json must contain an object")
    return value


def _checkpoint_files(directory: Path) -> tuple[Path, ...]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("Safetensors index must contain weight_map")
        names = sorted({str(name) for name in weight_map.values()})
        files = tuple(directory / name for name in names)
    else:
        single = directory / "model.safetensors"
        files = (single,) if single.is_file() else tuple(
            sorted(directory.glob("*.safetensors"))
        )
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError("Checkpoint safetensors files are missing")
    return files


def load_safetensors(directory: Path) -> dict[str, Tensor]:
    """Load one or more safetensors shards on CPU."""
    from safetensors import safe_open

    state: dict[str, Tensor] = {}
    for path in _checkpoint_files(directory):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                if key in state:
                    raise ValueError(f"Duplicate checkpoint tensor: {key}")
                state[key] = handle.get_tensor(key)
    return state


def _compatible_tensors(
    model: ExportModel,
    state: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Drop checkpoint tensors whose shapes do not match the model."""
    expected = model.state_dict()
    return {
        name: tensor
        for name, tensor in state.items()
        if name not in expected or tensor.shape == expected[name].shape
    }


def _pack_moe_experts(
    state: dict[str, Tensor],
    model: ExportModel,
) -> None:
    layers = model.model.layers
    for layer_id, layer in enumerate(layers):
        mlp = layer.mlp
        if not isinstance(mlp, Qwen3MoeSparseMoeBlock):
            continue
        expert_count = int(mlp.config.num_experts)
        prefix = f"model.layers.{layer_id}.mlp"
        gate_up_key = f"{prefix}.experts.gate_up_proj"
        down_key = f"{prefix}.experts.down_proj"
        if gate_up_key in state or down_key in state:
            if gate_up_key not in state or down_key not in state:
                raise ValueError(
                    "Packed MoE checkpoint requires gate_up and down weights"
                )
            gate_up = state.pop(gate_up_key)
            down = state.pop(down_key)
            intermediate_size = int(mlp.config.moe_intermediate_size)
            expected_gate_up = (
                expert_count,
                2 * intermediate_size,
                mlp.config.hidden_size,
            )
            expected_down = (
                expert_count,
                mlp.config.hidden_size,
                intermediate_size,
            )
            if (
                tuple(gate_up.shape) != expected_gate_up
                or tuple(down.shape) != expected_down
            ):
                raise ValueError(
                    "Packed MoE checkpoint tensor shape is invalid"
                )
            official_rows = [
                torch.cat(
                    (
                        gate_up[expert_id, :intermediate_size]
                        .reshape(-1),
                        gate_up[expert_id, intermediate_size:]
                        .reshape(-1),
                        down[expert_id].reshape(-1),
                    )
                )
                for expert_id in range(expert_count)
            ]
            packed = torch.stack(official_rows)
            mlp.set_packed_weights(packed)
            state[f"{prefix}.expert_weights"] = packed
            continue
        rows: list[Tensor] = []
        scale_rows: list[Tensor] = []
        offset_rows: list[Tensor] = []
        consumed: list[str] = []
        for expert_id in range(expert_count):
            projections: list[Tensor] = []
            projection_scales: list[Tensor] = []
            projection_offsets: list[Tensor] = []
            for projection in ("gate_proj", "up_proj", "down_proj"):
                key = (
                    f"model.layers.{layer_id}.mlp.experts."
                    f"{expert_id}.{projection}.weight"
                )
                try:
                    tensor = state[key]
                except KeyError as error:
                    raise KeyError(f"Missing MoE expert tensor: {key}") from error
                projections.append(tensor.reshape(-1))
                consumed.append(key)
                scale_key = key.replace(".weight", ".weight_scale")
                offset_key = key.replace(".weight", ".weight_offset")
                if scale_key in state:
                    projection_scales.append(
                        state[scale_key].reshape(-1)
                    )
                    consumed.append(scale_key)
                if offset_key in state:
                    projection_offsets.append(
                        state[offset_key].reshape(-1)
                    )
                    consumed.append(offset_key)
            rows.append(torch.cat(projections))
            if projection_scales:
                if len(projection_scales) != 3:
                    raise ValueError(
                        "Quantized MoE experts require every projection scale"
                    )
                scale_rows.append(
                    torch.stack(
                        [item.reshape(()) for item in projection_scales]
                    )
                )
            if projection_offsets:
                if len(projection_offsets) != 3:
                    raise ValueError(
                        "Quantized MoE experts require every projection offset"
                    )
                offset_rows.append(
                    torch.stack(
                        [item.reshape(()) for item in projection_offsets]
                    )
                )
        packed = torch.stack(rows)
        scales = torch.stack(scale_rows) if scale_rows else None
        offsets = torch.stack(offset_rows) if offset_rows else None
        if packed.dtype == torch.int8 and (
            scales is None or len(scale_rows) != expert_count
        ):
            raise ValueError(
                "INT8 MoE checkpoint requires every projection scale"
            )
        if offsets is not None and len(offset_rows) != expert_count:
            raise ValueError(
                "Quantized MoE checkpoint has incomplete projection offsets"
            )
        mlp.set_packed_weights(
            packed,
            scales=scales,
            offsets=offsets,
        )
        state[f"{prefix}.expert_weights"] = packed
        if scales is not None:
            state[f"{prefix}.quant_scales"] = scales
        if offsets is not None:
            state[f"{prefix}.quant_offsets"] = offsets
        for key in consumed:
            del state[key]


def load_model_state(model: ExportModel, directory: Path) -> None:
    """Load official Qwen3 tensors and pack MoE expert projections."""
    state = load_safetensors(directory)
    _pack_moe_experts(state, model)
    config = model.config
    if (
        config.tie_word_embeddings
        and "lm_head.weight" not in state
        and "model.embed_tokens.weight" in state
    ):
        state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    model.load_state_dict(_compatible_tensors(model, state), strict=False)
    model.requires_grad_(False)
    model.eval()


__all__ = [
    "load_config",
    "load_model_state",
    "load_safetensors",
    "resolve_checkpoint",
]
