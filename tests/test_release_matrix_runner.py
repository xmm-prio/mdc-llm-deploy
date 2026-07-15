"""Tests for the internal 28-entry release-matrix runner."""

from __future__ import annotations

import hashlib
import inspect
import subprocess
from pathlib import Path

import onnx
import pytest

from mdc_llm_deploy.capabilities import Algorithm
from tools import release_matrix
from tools.release_matrix import (
    RELEASE_SEQUENCE_LENGTH,
    UNBORN_COMMIT_SHA,
    build_release_matrix,
)


def test_runner_short_slice_generates_and_validates_all_28_models(tmp_path: Path) -> None:
    artifacts = build_release_matrix(tmp_path, sequence_length=8)

    assert len(artifacts) == 28
    assert len({item.sha256 for item in artifacts}) == 28
    assert all(item.path.is_file() for item in artifacts)
    assert all(len(item.sha256) == 64 for item in artifacts)
    assert all(
        {item.key: item.value for item in onnx.load(artifact.path).metadata_props}[
            "mdc.mask_mode"
        ]
        == artifact.capability.mask_mode.value
        for artifact in artifacts
    )
    assert all(artifact.sequence_length == 8 for artifact in artifacts)
    assert not any(artifact.release_qualified for artifact in artifacts)
    assert all(len(artifact.config_sha256) == 64 for artifact in artifacts)
    assert all(len(artifact.commit_sha) == 40 for artifact in artifacts)
    assert all(
        artifact.path.name
        == "-".join(
            (
                artifact.capability.model.value,
                artifact.capability.algorithm.value,
                artifact.capability.target.value
                if artifact.capability.target is not None
                else "baseline",
                artifact.capability.mask_mode.value,
                artifact.capability.phase.value,
                artifact.config_sha256[:8],
                artifact.commit_sha[:8] + ".onnx",
            )
        )
        for artifact in artifacts
    )
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_release_sequence_length_defaults_to_3072() -> None:
    parameter = inspect.signature(build_release_matrix).parameters["sequence_length"]

    assert parameter.default == RELEASE_SEQUENCE_LENGTH == 3072


def test_fp16_configuration_fingerprint_uses_documented_canonical_payload() -> None:
    fp16 = next(
        item
        for item in release_matrix.LOCAL_ONNX_MATRIX
        if item.algorithm is Algorithm.FP16
    )
    expected = hashlib.sha256(
        b'{"algorithm":"fp16","schema_version":1}'
    ).hexdigest()

    assert release_matrix._configuration_sha256(fp16) == expected
    assert release_matrix._canonical_json_sha256(
        {"schema_version": 1, "algorithm": "fp16"}
    ) == expected


@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [
        (128, ""),
        (0, "not-a-sha\n"),
    ],
)
def test_git_commit_sha_uses_stable_unborn_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout, "fatal")

    monkeypatch.setattr(release_matrix.subprocess, "run", fake_run)
    repository = Path("repository; echo injected")

    assert release_matrix._git_commit_sha(repository) == UNBORN_COMMIT_SHA
    assert calls == [
        (
            ["git", "rev-parse", "--verify", "HEAD"],
            {
                "cwd": repository,
                "check": False,
                "capture_output": True,
                "text": True,
            },
        )
    ]


def test_git_commit_sha_preserves_full_normalized_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
    monkeypatch.setattr(
        release_matrix.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, sha + "\n", ""),
    )

    assert release_matrix._git_commit_sha() == sha.lower()


def test_matrix_validation_rejects_duplicate_entries_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    duplicated = (
        *release_matrix.LOCAL_ONNX_MATRIX[:-1],
        release_matrix.LOCAL_ONNX_MATRIX[0],
    )
    monkeypatch.setattr(release_matrix, "LOCAL_ONNX_MATRIX", duplicated)

    with pytest.raises(AssertionError, match="28 unique entries"):
        build_release_matrix(tmp_path, sequence_length=8)

    assert not tuple(tmp_path.iterdir())


def test_structural_validation_failure_fails_slice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(release_matrix, "export", lambda model, calibration: object())
    monkeypatch.setattr(release_matrix, "onnx_export", lambda graph, path, **kwargs: None)
    monkeypatch.setattr(
        release_matrix,
        "validate_serialized_model",
        lambda path: (_ for _ in ()).throw(ValueError("invalid ONNX structure")),
    )

    with pytest.raises(ValueError, match="invalid ONNX structure"):
        build_release_matrix(tmp_path, sequence_length=8)
