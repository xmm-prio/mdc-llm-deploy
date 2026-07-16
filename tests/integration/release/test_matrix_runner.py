"""Tests for the mask-independent release-matrix runner."""

from __future__ import annotations

import hashlib
import inspect
import json
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
from tools.release_validation import validate_release_artifact

pytestmark = pytest.mark.integration


def _streamed_file_integrity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size_bytes += len(chunk)
            digest.update(chunk)
    return size_bytes, digest.hexdigest()


def test_artifact_manifest_records_ordered_members_and_canonical_bundle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx-content")
    data_path = tmp_path / "model.onnx.data"
    data_path.write_bytes(b"external-data")

    manifest = release_matrix._build_artifact_manifest(path)

    expected_members = (
        release_matrix.ArtifactMember(
            name="model.onnx",
            size_bytes=12,
            sha256=hashlib.sha256(b"onnx-content").hexdigest(),
        ),
        release_matrix.ArtifactMember(
            name="model.onnx.data",
            size_bytes=13,
            sha256=hashlib.sha256(b"external-data").hexdigest(),
        ),
    )
    canonical_payload = json.dumps(
        {
            "members": [
                {
                    "name": member.name,
                    "sha256": member.sha256,
                    "size_bytes": member.size_bytes,
                }
                for member in expected_members
            ],
            "schema_version": 1,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert manifest.schema_version == 1
    assert manifest.members == expected_members
    assert manifest.bundle_sha256 == hashlib.sha256(canonical_payload).hexdigest()


def test_artifact_manifest_bundle_is_independent_of_output_directory(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / directory / "model.onnx" for directory in ("first", "second")]
    for path in paths:
        path.parent.mkdir()
        path.write_bytes(b"onnx-content")
        path.with_name(f"{path.name}.data").write_bytes(b"external-data")

    manifests = tuple(release_matrix._build_artifact_manifest(path) for path in paths)

    assert manifests[0] == manifests[1]


def test_artifact_manifest_propagates_external_data_changes(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx-content")
    data_path = path.with_name(f"{path.name}.data")
    data_path.write_bytes(b"external-data-v1")
    original = release_matrix._build_artifact_manifest(path)

    data_path.write_bytes(b"external-data-v2")
    changed = release_matrix._build_artifact_manifest(path)

    assert original.members[0] == changed.members[0]
    assert original.members[1].sha256 != changed.members[1].sha256
    assert original.bundle_sha256 != changed.bundle_sha256


@pytest.mark.parametrize("member_kind", ["missing", "directory"])
def test_artifact_manifest_rejects_invalid_external_data_member(
    tmp_path: Path,
    member_kind: str,
) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx-content")
    data_path = path.with_name(f"{path.name}.data")
    if member_kind == "directory":
        data_path.mkdir()

    with pytest.raises(ValueError, match="not a regular file"):
        release_matrix._build_artifact_manifest(path)


@pytest.mark.slow
def test_runner_short_slice_generates_and_validates_all_14_models(tmp_path: Path) -> None:
    artifacts = build_release_matrix(tmp_path, sequence_length=8)
    evidence = tuple(
        validate_release_artifact(artifact.path, artifact.capability)
        for artifact in artifacts
    )

    assert len(artifacts) == 14
    assert all(item.output_names == ("logits",) for item in evidence)
    assert {
        model_kind: {
            dict(item.operator_counts)["NPURmsNorm"]
            for artifact, item in zip(artifacts, evidence, strict=True)
            if artifact.capability.model.value == model_kind
        }
        for model_kind in ("dense", "moe")
    } == {"dense": {9}, "moe": {9}}
    assert len({item.manifest.bundle_sha256 for item in artifacts}) == 14
    assert all(item.path.is_file() for item in artifacts)
    assert all(artifact.sequence_length == 8 for artifact in artifacts)
    assert not any(artifact.release_qualified for artifact in artifacts)
    assert all(len(artifact.config_sha256) == 64 for artifact in artifacts)
    assert all(len(artifact.commit_sha) == 40 for artifact in artifacts)
    for artifact in artifacts:
        assert artifact.manifest.schema_version == 1
        assert tuple(member.name for member in artifact.manifest.members) == (
            artifact.path.name,
            artifact.path.name + ".data",
        )
        for member in artifact.manifest.members:
            member_path = artifact.path.with_name(member.name)
            assert member_path.is_file()
            assert _streamed_file_integrity(member_path) == (
                member.size_bytes,
                member.sha256,
            )
        names = [
            node.name
            for node in onnx.load(artifact.path).graph.node
            if node.name
        ]
        assert len(names) == len(set(names))
    assert all(
        artifact.path.name
        == "-".join(
            (
                artifact.capability.model.value,
                artifact.capability.algorithm.value,
                artifact.capability.target.value
                if artifact.capability.target is not None
                else "baseline",
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

    with pytest.raises(AssertionError, match="14 unique entries"):
        build_release_matrix(tmp_path, sequence_length=8)

    assert not tuple(tmp_path.iterdir())


def test_semantic_validation_failure_fails_slice_before_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, object]] = []
    monkeypatch.setattr(release_matrix, "export", lambda model, calibration: object())
    monkeypatch.setattr(release_matrix, "onnx_export", lambda graph, path, **kwargs: None)

    def fail_validation(path: Path, capability: object) -> None:
        calls.append((path, capability))
        raise ValueError("invalid release semantics")

    monkeypatch.setattr(
        release_matrix,
        "validate_release_artifact",
        fail_validation,
    )
    monkeypatch.setattr(
        release_matrix,
        "_build_artifact_manifest",
        lambda path: pytest.fail("manifest must not be built"),
    )

    with pytest.raises(ValueError, match="invalid release semantics"):
        build_release_matrix(tmp_path, sequence_length=8)

    assert calls == [
        (
            tmp_path
            / (
                release_matrix._name(
                    release_matrix.LOCAL_ONNX_MATRIX[0],
                    release_matrix._configuration_sha256(
                        release_matrix.LOCAL_ONNX_MATRIX[0]
                    ),
                    release_matrix._git_commit_sha(),
                )
                + ".onnx"
            ),
            release_matrix.LOCAL_ONNX_MATRIX[0],
        )
    ]
