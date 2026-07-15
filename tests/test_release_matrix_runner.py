"""Tests for the internal 28-entry release-matrix runner."""

from __future__ import annotations

from pathlib import Path

import onnx

from tools.release_matrix import build_release_matrix


def test_runner_generates_and_validates_all_28_models(tmp_path: Path) -> None:
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
    assert not tuple(tmp_path.glob(".*.tmp"))
