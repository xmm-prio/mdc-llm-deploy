from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import onnx
import pytest
from onnx import TensorProto, helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.onnx_export import artifact_io


def _model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "artifact",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1,))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1,))],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _temporary_files(directory: Path) -> list[Path]:
    return list(directory.glob(".*.onnx.tmp"))


def test_commit_validated_onnx_replaces_target_after_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.onnx"
    target.write_bytes(b"old")
    monkeypatch.setattr(artifact_io, "validate_serialized_model", onnx.load)

    validated = artifact_io.commit_validated_onnx(
        _model(),
        target,
        overwrite=True,
    )

    assert validated.graph.name == "artifact"
    assert onnx.load(target).graph.name == "artifact"
    assert _temporary_files(tmp_path) == []


def test_commit_validated_onnx_preserves_target_when_serialization_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.onnx"
    target.write_bytes(b"old")
    original = RuntimeError("save failed")

    def fail_save(*args: object, **kwargs: object) -> NoReturn:
        raise original

    monkeypatch.setattr(onnx, "save_model", fail_save)

    with pytest.raises(OnnxExportError, match="ONNX export failed: save failed") as captured:
        artifact_io.commit_validated_onnx(
            _model(),
            target,
            overwrite=False,
        )

    assert captured.value.__cause__ is original
    assert target.read_bytes() == b"old"
    assert _temporary_files(tmp_path) == []


def test_commit_validated_onnx_preserves_target_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.onnx"
    target.write_bytes(b"old")
    original = OnnxExportError("invalid serialized model")

    def fail_validation(path: str) -> NoReturn:
        raise original

    monkeypatch.setattr(artifact_io, "validate_serialized_model", fail_validation)

    with pytest.raises(OnnxExportError, match="invalid serialized model") as captured:
        artifact_io.commit_validated_onnx(
            _model(),
            target,
            overwrite=False,
        )

    assert captured.value is original
    assert target.read_bytes() == b"old"
    assert _temporary_files(tmp_path) == []


def test_commit_validated_onnx_preserves_target_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.onnx"
    target.write_bytes(b"old")
    original = OSError("replace failed")
    monkeypatch.setattr(artifact_io, "validate_serialized_model", onnx.load)

    def fail_replace(source: Path, destination: Path) -> NoReturn:
        raise original

    monkeypatch.setattr(artifact_io.os, "replace", fail_replace)

    with pytest.raises(
        OnnxExportError,
        match="ONNX export failed: replace failed",
    ) as captured:
        artifact_io.commit_validated_onnx(
            _model(),
            target,
            overwrite=True,
        )

    assert captured.value.__cause__ is original
    assert target.read_bytes() == b"old"
    assert _temporary_files(tmp_path) == []


def test_commit_validated_onnx_does_not_overwrite_concurrently_created_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.onnx"
    target.write_bytes(b"concurrent")
    monkeypatch.setattr(artifact_io, "validate_serialized_model", onnx.load)

    with pytest.raises(FileExistsError):
        artifact_io.commit_validated_onnx(
            _model(),
            target,
            overwrite=False,
        )

    assert target.read_bytes() == b"concurrent"
    assert _temporary_files(tmp_path) == []
