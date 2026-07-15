"""Internal runner for the 28-entry local ONNX release matrix."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
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
from mdc_llm_deploy.onnx_export.validator import validate_serialized_model
from mdc_llm_deploy.quantization import oneshot

ROOT = Path(__file__).parents[1]
RELEASE_SEQUENCE_LENGTH = 3072
UNBORN_COMMIT_SHA = "0" * 40
FP16_CONFIGURATION = {
    "algorithm": "fp16",
    "schema_version": 1,
}
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
    config_sha256: str
    commit_sha: str
    sequence_length: int
    release_qualified: bool


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _configuration_sha256(capability: Capability) -> str:
    """Return a stable full configuration fingerprint."""
    if capability.algorithm is Algorithm.FP16:
        return _canonical_json_sha256(FP16_CONFIGURATION)
    if capability.target is None:
        raise AssertionError("MinMax matrix entry must declare a target")
    payload = CONFIG_BY_TARGET[capability.target].read_text(encoding="utf-8")
    return _canonical_json_sha256(json.loads(payload))


def _git_commit_sha(repository: Path = ROOT) -> str:
    """Return HEAD SHA, or a stable sentinel for an unborn repository."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    candidate = result.stdout.strip().lower()
    if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", candidate):
        return candidate
    return UNBORN_COMMIT_SHA


def _name(capability: Capability, config_sha256: str, commit_sha: str) -> str:
    target = capability.target.value if capability.target is not None else "baseline"
    return "-".join(
        (
            capability.model.value,
            capability.algorithm.value,
            target,
            capability.mask_mode.value,
            capability.phase.value,
            config_sha256[:8],
            commit_sha[:8],
        )
    )


def _validate_matrix() -> None:
    identities = {
        (
            item.model,
            item.algorithm,
            item.target,
            item.mask_mode,
            item.phase,
        )
        for item in LOCAL_ONNX_MATRIX
    }
    if len(LOCAL_ONNX_MATRIX) != 28 or len(identities) != 28:
        raise AssertionError("Release matrix must contain 28 unique entries")


def build_release_matrix(
    output_directory: str | Path,
    *,
    sequence_length: int = RELEASE_SEQUENCE_LENGTH,
) -> tuple[MatrixArtifact, ...]:
    """Generate and structurally validate every FP16/MinMax ONNX combination.

    Runs using a shorter explicit sequence length are test slices, not release
    validation. Callers must inspect ``release_qualified`` before summarizing.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    _validate_matrix()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[MatrixArtifact] = []
    input_ids = (torch.arange(sequence_length) % 128).reshape(1, sequence_length)
    calibration = {"input_ids": input_ids}
    commit_sha = _git_commit_sha()
    release_qualified = sequence_length == RELEASE_SEQUENCE_LENGTH
    for capability in LOCAL_ONNX_MATRIX:
        config_sha256 = _configuration_sha256(capability)
        model_type = (
            TinyQwen3Dense
            if capability.model is ModelKind.DENSE
            else TinyQwen3Moe
        )
        graph = export(model_type().eval().half(), calibration)
        if capability.algorithm is Algorithm.MINMAX:
            if capability.target is None:
                raise AssertionError("MinMax matrix entry must declare a target")
            oneshot(graph, str(CONFIG_BY_TARGET[capability.target]), [calibration])
        if capability.phase is Phase.DECODE:
            convert_to_decode(graph)
        path = output / f"{_name(capability, config_sha256, commit_sha)}.onnx"
        onnx_export(
            graph,
            path,
            mask_mode=cast(OnnxMaskMode, capability.mask_mode.value),
        )
        validate_serialized_model(str(path))
        artifacts.append(
            MatrixArtifact(
                capability=capability,
                path=path,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                config_sha256=config_sha256,
                commit_sha=commit_sha,
                sequence_length=sequence_length,
                release_qualified=release_qualified,
            )
        )
    if len(artifacts) != 28 or len({item.path.name for item in artifacts}) != 28:
        raise AssertionError("Release matrix must produce 28 unique artifacts")
    return tuple(artifacts)
