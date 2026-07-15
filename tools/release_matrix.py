"""Internal runner for the 28-entry local ONNX release matrix."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from mdc_llm_deploy.capabilities import (
    CAPABILITY_MATRIX,
    Algorithm,
    Artifact,
    Capability,
    ModelKind,
    Phase,
    Target,
)
from mdc_llm_deploy.export import convert_to_decode, export
from mdc_llm_deploy.models import TinyQwen3Dense, TinyQwen3Moe
from mdc_llm_deploy.onnx_export import onnx_export
from mdc_llm_deploy.onnx_export.api import MaskMode as OnnxMaskMode
from mdc_llm_deploy.quantization import oneshot

ROOT = Path(__file__).parents[1]
CONFIG_BY_TARGET = {
    Target.LINEAR: ROOT / "configs" / "minmax-linear-w8a8.json",
    Target.ATTENTION: ROOT / "configs" / "minmax-attention-a8.json",
    Target.MOE: ROOT / "configs" / "minmax-moe-w8a8.json",
}
LOCAL_ONNX_MATRIX = tuple(
    item
    for item in CAPABILITY_MATRIX
    if item.algorithm in {Algorithm.FP16, Algorithm.MINMAX}
    and item.supports(Artifact.ONNX)
)


@dataclass(frozen=True, slots=True)
class MatrixArtifact:
    """One generated and validated release-matrix artifact."""

    capability: Capability
    path: Path
    sha256: str


def _name(capability: Capability) -> str:
    target = capability.target.value if capability.target is not None else "baseline"
    return "-".join(
        (
            capability.model.value,
            capability.algorithm.value,
            target,
            capability.phase.value,
            capability.mask_mode.value,
        )
    )


def build_release_matrix(
    output_directory: str | Path,
    *,
    sequence_length: int = 8,
) -> tuple[MatrixArtifact, ...]:
    """Generate and structurally validate every FP16/MinMax ONNX combination."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[MatrixArtifact] = []
    input_ids = (torch.arange(sequence_length) % 128).reshape(1, sequence_length)
    calibration = {"input_ids": input_ids}
    for capability in LOCAL_ONNX_MATRIX:
        model_type = (
            TinyQwen3Dense
            if capability.model is ModelKind.DENSE
            else TinyQwen3Moe
        )
        graph = export(model_type().eval().half(), calibration)
        if capability.algorithm is Algorithm.MINMAX:
            if capability.target is None:
                raise AssertionError("MinMax matrix entry must declare a target")
            oneshot(graph, CONFIG_BY_TARGET[capability.target], [calibration])
        if capability.phase is Phase.DECODE:
            convert_to_decode(graph)
        path = output / f"{_name(capability)}.onnx"
        onnx_export(
            graph,
            path,
            mask_mode=cast(OnnxMaskMode, capability.mask_mode.value),
        )
        artifacts.append(
            MatrixArtifact(
                capability=capability,
                path=path,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    if len(artifacts) != 28 or len({item.path.name for item in artifacts}) != 28:
        raise AssertionError("Release matrix must produce 28 unique artifacts")
    return tuple(artifacts)
