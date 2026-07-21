from __future__ import annotations

import subprocess
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.mdc_onnx import process_onnx


def _model(*, include_gelu: bool = False) -> onnx.ModelProto:
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", "a_scale", "a_zp"],
            ["a_q"],
            name="activation_q",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["a_q", "a_scale", "a_zp"],
            ["a_dq"],
            name="activation_dq",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["weight", "w_scale", "w_zp"],
            ["w_q"],
            name="weight_q",
            axis=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["w_q", "w_scale", "w_zp"],
            ["w_dq"],
            name="weight_dq",
            axis=1,
        ),
        helper.make_node("MatMul", ["a_dq", "w_dq"], ["linear_y"], name="linear"),
    ]
    output_name = "linear_y"
    if include_gelu:
        nodes.append(helper.make_node("Gelu", ["linear_y"], ["y"], name="new_gelu"))
        output_name = "y"
    graph = helper.make_graph(
        nodes,
        "pipeline",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2, 3])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, [1, 2, 4])],
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
            helper.make_tensor_value_info("linear_y", TensorProto.FLOAT16, [1, 2, 4]),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def test_process_onnx_runs_atomic_pipeline() -> None:
    model = _model()

    returned = process_onnx(model)

    assert returned is model
    assert model.opset_import[0].version == 18
    assert [node.op_type for node in model.graph.node] == [
        "NPUAscendQuantV2",
        "MatMul",
        "AscendDequant",
    ]
    onnx.checker.check_model(model)


def test_second_stage_failure_rolls_back_entire_pipeline() -> None:
    model = _model(include_gelu=True)
    original = model.SerializeToString()

    with pytest.raises(ValueError, match="Gelu"):
        process_onnx(model)

    assert model.SerializeToString() == original


def test_conflicting_process_schema_fails_on_import() -> None:
    code = """
from onnx import defs
from onnx.defs import OpSchema
parameter = OpSchema.FormalParameter
defs.register_schema(OpSchema(
    "NPUAscendQuantV2", "", 18,
    inputs=[parameter("wrong", "T")],
    outputs=[parameter("wrong_out", "T")],
    type_constraints=[("T", ["tensor(float)"], "wrong")],
))
import mdc_llm_deploy.mdc_onnx
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "conflicting ONNX schema" in result.stderr
