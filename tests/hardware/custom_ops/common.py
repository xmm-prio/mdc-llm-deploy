"""Shared deterministic ONNX case generation primitives."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch

from mdc_llm_deploy.custom_ops import create_onnx_export_profile


@dataclass(frozen=True)
class CaseDefinition:
    """Describe one hardware validation case without operator-specific behavior."""

    name: str
    golden_model: torch.nn.Module
    custom_model: torch.nn.Module
    inputs: Mapping[str, torch.Tensor]
    output_names: Sequence[str]
    description: str
    operator_names: tuple[str, ...]
    opset_version: int = 18


def _export_model(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: Sequence[str],
    path: Path,
    opset_version: int,
    operator_names: tuple[str, ...] = (),
) -> None:
    model.eval()
    profile = create_onnx_export_profile(*operator_names)
    torch.onnx.export(
        model,
        inputs,
        path,
        opset_version=opset_version,
        dynamo=True,
        verbose=False,
        input_names=input_names,
        output_names=list(output_names),
        external_data=False,
        optimize=False,
        custom_translation_table=dict(profile.custom_translation_table),
    )


def _write_tensor(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    value.view(torch.uint8).numpy().tofile(path)
    return {
        "path": path.name,
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": list(value.shape),
        "byte_size": path.stat().st_size,
    }


def generate_case(definition: CaseDefinition, output_root: Path) -> Path:
    """Generate one complete case directory and replace stale output."""
    case_dir = output_root / definition.name
    temporary_dir = output_root / f".{definition.name}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    try:
        input_names = list(definition.inputs)
        input_values = tuple(definition.inputs.values())
        _export_model(
            definition.golden_model,
            input_values,
            input_names,
            definition.output_names,
            temporary_dir / "golden.onnx",
            definition.opset_version,
        )
        _export_model(
            definition.custom_model,
            input_values,
            input_names,
            definition.output_names,
            temporary_dir / "custom.onnx",
            definition.opset_version,
            definition.operator_names,
        )

        onnx.checker.check_model(temporary_dir / "golden.onnx", full_check=True)
        onnx.checker.check_model(temporary_dir / "custom.onnx", full_check=True)
        input_manifest = {
            name: _write_tensor(temporary_dir / f"{name}.bin", tensor)
            for name, tensor in definition.inputs.items()
        }
        manifest = {
            "name": definition.name,
            "description": definition.description,
            "opset_version": definition.opset_version,
            "models": {"golden": "golden.onnx", "custom": "custom.onnx"},
            "inputs": input_manifest,
            "outputs": list(definition.output_names),
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output_root.mkdir(parents=True, exist_ok=True)
        if case_dir.exists():
            shutil.rmtree(case_dir)
        temporary_dir.replace(case_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return case_dir


def seeded_generator(seed: int) -> torch.Generator:
    """Create an isolated CPU generator for reproducible case tensors."""
    return torch.Generator(device="cpu").manual_seed(seed)
