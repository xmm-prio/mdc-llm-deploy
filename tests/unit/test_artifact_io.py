from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.onnx_export.artifact_io import commit_validated_onnx


def _model() -> onnx.ModelProto:
    weight = numpy_helper.from_array(
        np.ones((2, 2), dtype=np.float32),
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


def test_external_data_is_named_and_replaced_atomically(tmp_path: Path) -> None:
    target = tmp_path / "model.onnx"
    target.write_bytes(b"old")

    commit_validated_onnx(_model(), target, external_data=True)

    assert target.is_file()
    assert (tmp_path / "model.onnx.data").is_file()
    loaded = onnx.load(target, load_external_data=True)
    onnx.checker.check_model(loaded)
    assert not any(path.name.startswith(".model") for path in tmp_path.iterdir())


def test_inline_export_removes_obsolete_external_data(tmp_path: Path) -> None:
    target = tmp_path / "model.onnx"
    data = tmp_path / "model.onnx.data"
    data.write_bytes(b"obsolete")

    commit_validated_onnx(_model(), target, external_data=False)

    assert target.is_file()
    assert not data.exists()
