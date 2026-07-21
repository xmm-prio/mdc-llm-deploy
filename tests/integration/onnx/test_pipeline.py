from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.onnx import pipeline, process_onnx


def _identity_model(*, opset: int = 21) -> onnx.ModelProto:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="identity")],
        "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _qdq_model() -> onnx.ModelProto:
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "a_scale", "a_zp"], ["a_q"]),
        helper.make_node("DequantizeLinear", ["a_q", "a_scale", "a_zp"], ["a_dq"]),
        helper.make_node(
            "QuantizeLinear",
            ["weight", "w_scale", "w_zp"],
            ["w_q"],
            axis=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["w_q", "w_scale", "w_zp"],
            ["w_dq"],
            axis=1,
        ),
        helper.make_node("MatMul", ["a_dq", "w_dq"], ["y"], name="linear"),
    ]
    graph = helper.make_graph(
        nodes,
        "pipeline",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 2, 4])],
        initializer=[
            numpy_helper.from_array(np.array(0.25, dtype=np.float16), "a_scale"),
            numpy_helper.from_array(np.array(0, dtype=np.int8), "a_zp"),
            numpy_helper.from_array(np.ones((3, 4), dtype=np.float16), "weight"),
            numpy_helper.from_array(np.full((4,), 0.5, dtype=np.float16), "w_scale"),
            numpy_helper.from_array(np.zeros((4,), dtype=np.int8), "w_zp"),
        ],
        value_info=[
            helper.make_tensor_value_info("a_q", TensorProto.INT8, [1, 2, 3]),
            helper.make_tensor_value_info("a_dq", TensorProto.FLOAT16, [1, 2, 3]),
            helper.make_tensor_value_info("w_q", TensorProto.INT8, [3, 4]),
            helper.make_tensor_value_info("w_dq", TensorProto.FLOAT16, [3, 4]),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def test_process_onnx_runs_complete_atomic_pipeline() -> None:
    model = _qdq_model()

    returned = process_onnx(model)

    assert returned is model
    assert model.opset_import[0].version == 18
    assert [node.op_type for node in model.graph.node] == [
        "NPUAscendQuantV2",
        "MatMul",
        "AscendDequant",
    ]
    onnx.checker.check_model(model)


def test_pipeline_stage_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    model = _identity_model()

    def stage(name: str) -> Callable[[onnx.ModelProto], object]:
        def record(_model: onnx.ModelProto) -> object:
            calls.append(name)
            return object()

        return record

    monkeypatch.setattr(pipeline, "lower_qdq_core", stage("lower"))
    monkeypatch.setattr(pipeline, "_register_required_schemas", stage("register"))
    monkeypatch.setattr(pipeline, "downgrade_opset_core", stage("downgrade"))
    monkeypatch.setattr(pipeline, "run_fusion_passes", stage("fusion"))
    monkeypatch.setattr(pipeline, "_validate_final_graph", stage("checker"))

    process_onnx(model)

    assert calls == [
        "lower",
        "register",
        "downgrade",
        "fusion",
        "register",
        "checker",
    ]


@pytest.mark.parametrize("failing_stage", ["lower", "downgrade", "fusion", "checker"])
def test_pipeline_failure_rolls_back_original_model(
    monkeypatch: pytest.MonkeyPatch,
    failing_stage: str,
) -> None:
    model = _identity_model()
    original = model.SerializeToString()

    def fail(working: onnx.ModelProto) -> None:
        working.doc_string = "mutated working clone"
        raise RuntimeError(f"{failing_stage} failed")

    target = {
        "lower": "lower_qdq_core",
        "downgrade": "downgrade_opset_core",
        "fusion": "run_fusion_passes",
        "checker": "_validate_final_graph",
    }[failing_stage]
    monkeypatch.setattr(pipeline, target, fail)

    with pytest.raises(RuntimeError, match=f"{failing_stage} failed"):
        process_onnx(model)

    assert model.SerializeToString() == original


def test_package_import_has_no_schema_registration_side_effect() -> None:
    code = """
import onnx
from mdc_llm_deploy.onnx.schemas import ALL_SCHEMA_NAMES
import mdc_llm_deploy.onnx
for name in ALL_SCHEMA_NAMES:
    try:
        onnx.defs.get_schema(name, 18, "")
    except onnx.defs.SchemaError:
        continue
    raise AssertionError(f"schema registered during import: {name}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_process_registers_only_custom_schemas_present_in_graph() -> None:
    code = """
import onnx
from onnx import TensorProto, helper
from mdc_llm_deploy.onnx import process_onnx
from mdc_llm_deploy.onnx.schemas import ALL_SCHEMA_NAMES, RMS_NORM_OP

graph = helper.make_graph(
    [helper.make_node(RMS_NORM_OP, ["x", "gamma"], ["y", "rstd"], epsilon=1e-6)],
    "custom",
    [
        helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8]),
        helper.make_tensor_value_info("gamma", TensorProto.FLOAT16, [8]),
    ],
    [
        helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 8]),
        helper.make_tensor_value_info("rstd", TensorProto.FLOAT, [1]),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
assert process_onnx(model) is model
assert onnx.defs.get_schema(RMS_NORM_OP, 18, "").since_version == 18
for name in ALL_SCHEMA_NAMES:
    if name == RMS_NORM_OP:
        continue
    try:
        onnx.defs.get_schema(name, 18, "")
    except onnx.defs.SchemaError:
        continue
    raise AssertionError(f"unneeded schema registered: {name}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
