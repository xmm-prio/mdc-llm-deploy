from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.onnx_export import artifact_io
from mdc_llm_deploy.onnx_export.artifact_io import commit_validated_onnx


def _model(weight_value: float = 1.0) -> onnx.ModelProto:
    weight = numpy_helper.from_array(
        np.full((2, 2), weight_value, dtype=np.float32),
        name="weight",
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "artifact",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1, 2))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 2))],
        initializer=[weight],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    )


def _weight_value(target: Path) -> float:
    loaded = onnx.load(target, load_external_data=True)
    onnx.checker.check_model(loaded)
    return float(numpy_helper.to_array(loaded.graph.initializer[0])[0, 0])


def _fail_model_publication_once(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
) -> None:
    real_replace = artifact_io.os.replace
    failed = False

    def replace(source: Path, destination: Path) -> None:
        nonlocal failed
        if not failed and Path(source).name == target.name and Path(destination) == target:
            failed = True
            raise OSError("model publication failed")
        real_replace(source, destination)

    monkeypatch.setattr(artifact_io.os, "replace", replace)


def test_external_data_is_named_and_replaced_atomically(tmp_path: Path) -> None:
    target = tmp_path / "model.onnx"
    commit_validated_onnx(_model(1.0), target, external_data=True)

    commit_validated_onnx(_model(2.0), target, external_data=True)

    assert target.is_file()
    assert (tmp_path / "model.onnx.data").is_file()
    assert _weight_value(target) == 2.0
    assert not any(path.name.startswith(".model") for path in tmp_path.iterdir())


def test_inline_export_removes_obsolete_external_data(tmp_path: Path) -> None:
    target = tmp_path / "model.onnx"
    data = tmp_path / "model.onnx.data"
    data.write_bytes(b"obsolete")

    commit_validated_onnx(_model(), target, external_data=False)

    assert target.is_file()
    assert not data.exists()


def test_external_failure_restores_previous_external_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.onnx"
    data_target = tmp_path / "model.onnx.data"
    commit_validated_onnx(_model(1.0), target, external_data=True)
    original_model = target.read_bytes()
    original_data = data_target.read_bytes()
    _fail_model_publication_once(monkeypatch, target)

    with pytest.raises(OnnxExportError, match="model publication failed"):
        commit_validated_onnx(_model(2.0), target, external_data=True)

    assert target.read_bytes() == original_model
    assert data_target.read_bytes() == original_data
    assert _weight_value(target) == 1.0
    assert not any(path.name.startswith(".model") for path in tmp_path.iterdir())


def test_external_failure_restores_previous_inline_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.onnx"
    data_target = tmp_path / "model.onnx.data"
    commit_validated_onnx(_model(1.0), target, external_data=False)
    original_model = target.read_bytes()
    _fail_model_publication_once(monkeypatch, target)

    with pytest.raises(OnnxExportError, match="model publication failed"):
        commit_validated_onnx(_model(2.0), target, external_data=True)

    assert target.read_bytes() == original_model
    assert not data_target.exists()
    assert _weight_value(target) == 1.0


def test_inline_failure_restores_previous_external_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.onnx"
    data_target = tmp_path / "model.onnx.data"
    commit_validated_onnx(_model(1.0), target, external_data=True)
    original_model = target.read_bytes()
    original_data = data_target.read_bytes()
    real_unlink = Path.unlink
    failed = False

    def unlink(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal failed
        if not failed and path == data_target:
            failed = True
            raise OSError("sidecar removal failed")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)

    with pytest.raises(OnnxExportError, match="sidecar removal failed"):
        commit_validated_onnx(_model(2.0), target, external_data=False)

    assert target.read_bytes() == original_model
    assert data_target.read_bytes() == original_data
    assert _weight_value(target) == 1.0


def test_snapshot_failure_does_not_change_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.onnx"
    data_target = tmp_path / "model.onnx.data"
    commit_validated_onnx(_model(1.0), target, external_data=True)
    original_model = target.read_bytes()
    original_data = data_target.read_bytes()
    replace_calls = 0

    def link(source: Path, destination: Path) -> None:
        raise OSError("snapshot failed")

    def replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1

    monkeypatch.setattr(artifact_io.os, "link", link)
    monkeypatch.setattr(artifact_io.os, "replace", replace)

    with pytest.raises(OnnxExportError, match="snapshot failed"):
        commit_validated_onnx(_model(2.0), target, external_data=True)

    assert replace_calls == 0
    assert target.read_bytes() == original_model
    assert data_target.read_bytes() == original_data


def test_rollback_failure_reports_both_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.onnx"
    data_target = tmp_path / "model.onnx.data"
    commit_validated_onnx(_model(1.0), target, external_data=True)
    real_replace = artifact_io.os.replace

    def replace(source: Path, destination: Path) -> None:
        source = Path(source)
        destination = Path(destination)
        if source.name == target.name and destination == target:
            raise OSError("publication failed")
        if source.name == ".backup-0" and destination == data_target:
            raise OSError("rollback failed")
        real_replace(source, destination)

    monkeypatch.setattr(artifact_io.os, "replace", replace)

    with pytest.raises(
        OnnxExportError,
        match="ONNX publication failed: publication failed; rollback failed: rollback failed",
    ) as raised:
        commit_validated_onnx(_model(2.0), target, external_data=True)

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "publication failed"
