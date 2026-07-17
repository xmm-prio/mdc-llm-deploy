from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.onnx.export import artifacts as artifact_io
from mdc_llm_deploy.onnx.export.artifacts import (
    commit_mdc_onnx,
    commit_standard_onnx,
)


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


def _mdc_model(weight_value: float = 1.0) -> onnx.ModelProto:
    model = _model(weight_value)
    helper.set_model_props(
        model,
        {
            "mdc.graph_schema_version": "1",
            "mdc.stage": "FLOAT_PREFILL",
            "mdc.mask_mode": "masked",
            "mdc.mask_semantics": "explicit-causal",
            "mdc.model_kind": "dense",
            "mdc.algorithm": "fp16",
            "mdc.target": "fp16",
            "mdc.dialect": "MDC ONNX",
            "mdc.numeric_spine": "validated-standard-aten",
            "mdc.lowering_source": "test",
        },
    )
    return model


def _weight_value(target: Path) -> float:
    loaded = onnx.load(target, load_external_data=True)
    onnx.checker.check_model(loaded)
    return float(numpy_helper.to_array(loaded.graph.initializer[0])[0, 0])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda model: setattr(model.opset_import[0], "version", 17),
            "MDC ONNX must use opset",
        ),
        (
            lambda model: model.opset_import.append(
                helper.make_opsetid("unsupported.domain", 1)
            ),
            "unsupported operator domain",
        ),
        (
            lambda model: setattr(
                model.graph.node[0],
                "op_type",
                "ApplyRotaryPosEmb",
            ),
            "output count does not match schema",
        ),
    ],
)
def test_mdc_commit_rejects_invalid_dialect_model(
    tmp_path: Path,
    mutate: Callable[[onnx.ModelProto], None],
    message: str,
) -> None:
    model = _mdc_model()
    mutate(model)

    with pytest.raises(OnnxExportError, match=message):
        commit_mdc_onnx(model, tmp_path / "invalid.onnx", external_data=False)


def test_mdc_commit_rejects_missing_metadata_before_publication(
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.onnx"
    target.write_bytes(b"old artifact")

    with pytest.raises(
        OnnxExportError,
        match="MDC metadata properties are incomplete",
    ):
        commit_mdc_onnx(_model(), target, external_data=False)

    assert target.read_bytes() == b"old artifact"
    assert not (tmp_path / "model.onnx.data").exists()
    assert not any(path.name.startswith(".model") for path in tmp_path.iterdir())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mdc.dialect": "ONNX"}, "Invalid MDC dialect marker"),
        (
            {"mdc.mask_semantics": "all-visible-non-causal"},
            "Mask semantics metadata is inconsistent",
        ),
        (
            {"mdc.algorithm": "minmax", "mdc.target": "linear"},
            "Float stage metadata must declare fp16",
        ),
        (
            {"mdc.stage": "QUANTIZED_PREFILL"},
            "Quantized stage metadata cannot declare fp16",
        ),
        (
            {
                "mdc.stage": "QUANTIZED_PREFILL",
                "mdc.algorithm": "gptq",
                "mdc.target": "linear",
            },
            "MDC ONNX does not support GPTQ metadata",
        ),
    ],
)
def test_mdc_commit_rejects_invalid_metadata_semantics(
    tmp_path: Path,
    changes: dict[str, str],
    message: str,
) -> None:
    model = _mdc_model()
    properties = {item.key: item.value for item in model.metadata_props}
    helper.set_model_props(model, properties | changes)

    with pytest.raises(OnnxExportError, match=message):
        commit_mdc_onnx(model, tmp_path / "invalid.onnx", external_data=False)


@pytest.mark.parametrize("external_data", [False, True])
def test_mdc_commit_round_trips_valid_metadata(
    tmp_path: Path,
    external_data: bool,
) -> None:
    target = tmp_path / "model.onnx"
    expected = {item.key: item.value for item in _mdc_model().metadata_props}

    committed = commit_mdc_onnx(
        _mdc_model(),
        target,
        external_data=external_data,
    )

    assert {item.key: item.value for item in committed.metadata_props} == expected
    assert (tmp_path / "model.onnx.data").exists() is external_data
    assert not any(path.name.startswith(".model") for path in tmp_path.iterdir())


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
    commit_standard_onnx(_model(1.0), target, external_data=True)

    commit_standard_onnx(_model(2.0), target, external_data=True)

    assert target.is_file()
    assert (tmp_path / "model.onnx.data").is_file()
    assert _weight_value(target) == 2.0
    assert not any(path.name.startswith(".model") for path in tmp_path.iterdir())


def test_inline_export_removes_obsolete_external_data(tmp_path: Path) -> None:
    target = tmp_path / "model.onnx"
    data = tmp_path / "model.onnx.data"
    data.write_bytes(b"obsolete")

    commit_standard_onnx(_model(), target, external_data=False)

    assert target.is_file()
    assert not data.exists()


def test_external_failure_restores_previous_external_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.onnx"
    data_target = tmp_path / "model.onnx.data"
    commit_standard_onnx(_model(1.0), target, external_data=True)
    original_model = target.read_bytes()
    original_data = data_target.read_bytes()
    _fail_model_publication_once(monkeypatch, target)

    with pytest.raises(OnnxExportError, match="model publication failed"):
        commit_standard_onnx(_model(2.0), target, external_data=True)

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
    commit_standard_onnx(_model(1.0), target, external_data=False)
    original_model = target.read_bytes()
    _fail_model_publication_once(monkeypatch, target)

    with pytest.raises(OnnxExportError, match="model publication failed"):
        commit_standard_onnx(_model(2.0), target, external_data=True)

    assert target.read_bytes() == original_model
    assert not data_target.exists()
    assert _weight_value(target) == 1.0


def test_inline_failure_restores_previous_external_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.onnx"
    data_target = tmp_path / "model.onnx.data"
    commit_standard_onnx(_model(1.0), target, external_data=True)
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
        commit_standard_onnx(_model(2.0), target, external_data=False)

    assert target.read_bytes() == original_model
    assert data_target.read_bytes() == original_data
    assert _weight_value(target) == 1.0


def test_link_snapshot_failure_falls_back_to_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.onnx"
    data_target = tmp_path / "model.onnx.data"
    commit_standard_onnx(_model(1.0), target, external_data=True)

    def link(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        raise PermissionError("link snapshot failed")

    monkeypatch.setattr(artifact_io.os, "link", link)

    commit_standard_onnx(_model(2.0), target, external_data=True)

    assert target.is_file()
    assert data_target.is_file()
    assert _weight_value(target) == 2.0
    assert not any(path.name.startswith(".model") for path in tmp_path.iterdir())


def test_copy_snapshots_restore_previous_external_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.onnx"
    data_target = tmp_path / "model.onnx.data"
    commit_standard_onnx(_model(1.0), target, external_data=True)
    original_model = target.read_bytes()
    original_data = data_target.read_bytes()

    def link(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        raise OSError("link snapshot failed")

    monkeypatch.setattr(artifact_io.os, "link", link)
    _fail_model_publication_once(monkeypatch, target)

    with pytest.raises(OnnxExportError, match="model publication failed"):
        commit_standard_onnx(_model(2.0), target, external_data=True)

    assert target.read_bytes() == original_model
    assert data_target.read_bytes() == original_data
    assert _weight_value(target) == 1.0
    assert not any(path.name.startswith(".model") for path in tmp_path.iterdir())


def test_partial_copy_snapshot_failure_does_not_start_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.onnx"
    data_target = tmp_path / "model.onnx.data"
    commit_standard_onnx(_model(1.0), target, external_data=True)
    original_model = target.read_bytes()
    original_data = data_target.read_bytes()
    real_copy2 = artifact_io.shutil.copy2
    copy_calls = 0
    replace_calls = 0
    copy_error = OSError("copy snapshot failed")

    def link(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        raise OSError("link snapshot failed")

    def copy2(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> str:
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == 2:
            raise copy_error
        return real_copy2(source, destination, follow_symlinks=follow_symlinks)

    def replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1

    monkeypatch.setattr(artifact_io.os, "link", link)
    monkeypatch.setattr(artifact_io.shutil, "copy2", copy2)
    monkeypatch.setattr(artifact_io.os, "replace", replace)

    with pytest.raises(
        OnnxExportError,
        match=r"link snapshot failed.*copy snapshot failed",
    ) as raised:
        commit_standard_onnx(_model(2.0), target, external_data=True)

    assert raised.value.__cause__ is copy_error
    assert replace_calls == 0
    assert target.read_bytes() == original_model
    assert data_target.read_bytes() == original_data
    assert _weight_value(target) == 1.0
    assert not any(path.name.startswith(".model") for path in tmp_path.iterdir())


def test_rollback_failure_reports_both_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.onnx"
    data_target = tmp_path / "model.onnx.data"
    commit_standard_onnx(_model(1.0), target, external_data=True)
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
        commit_standard_onnx(_model(2.0), target, external_data=True)

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "publication failed"
