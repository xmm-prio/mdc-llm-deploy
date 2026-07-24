from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from mdc_llm_deploy.onnx.graph import GraphIndex, constant_array


def test_constant_array_evaluates_constant_reshape() -> None:
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Constant", [], ["axes"], value_ints=[-1]),
                helper.make_node("Cast", ["axes"], ["cast_axes"], to=TensorProto.INT64),
                helper.make_node("Constant", [], ["shape"], value_ints=[-1]),
                helper.make_node("Reshape", ["cast_axes", "shape"], ["reshaped_axes"]),
                helper.make_node("Identity", ["x"], ["y"]),
            ],
            "constant_reshape",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
    )

    value = constant_array(GraphIndex(model), "reshaped_axes")

    assert value is not None
    np.testing.assert_array_equal(value, np.asarray([-1], dtype=np.int64))


def test_constant_array_rejects_dynamic_reshape() -> None:
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Reshape", ["x", "shape"], ["y"])],
            "dynamic_reshape",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
                helper.make_tensor_value_info("shape", TensorProto.INT64, [1]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        )
    )

    assert constant_array(GraphIndex(model), "y") is None
