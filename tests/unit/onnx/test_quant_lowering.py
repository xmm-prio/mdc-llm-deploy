from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.onnx import lower_qdq


def _model(
    *,
    per_token: bool = False,
    transpose_weight: bool = False,
    activation_zero_point: int = 0,
    weight_zero_point: int = 0,
    output_dtype: int = TensorProto.FLOAT16,
) -> onnx.ModelProto:
    float_dtype = np.float16 if output_dtype == TensorProto.FLOAT16 else np.float32
    activation_shape = [1, 2, 3]
    output_shape = [1, 2, 4]
    activation_scale = (
        np.array([0.25, 0.5], dtype=float_dtype)
        if per_token
        else np.array(0.25, dtype=float_dtype)
    )
    activation_zp = (
        np.full((2,), activation_zero_point, dtype=np.int8)
        if per_token
        else np.array(activation_zero_point, dtype=np.int8)
    )
    weight_scale = np.array([0.5, 0.25, 0.125, 0.0625], dtype=float_dtype)
    weight_zp = np.full((4,), weight_zero_point, dtype=np.int8)
    weight_kn = np.array(
        [
            [1.0, -1.0, 0.5, 0.25],
            [0.5, 0.25, -0.5, 1.0],
            [-1.0, 0.5, 0.25, -0.25],
        ],
        dtype=float_dtype,
    )
    weight = weight_kn.T if transpose_weight else weight_kn
    weight_axis = 0 if transpose_weight else 1

    initializers = [
        numpy_helper.from_array(activation_scale, "a_scale"),
        numpy_helper.from_array(activation_zp, "a_zp"),
        numpy_helper.from_array(weight, "weight"),
        numpy_helper.from_array(weight_scale, "w_scale"),
        numpy_helper.from_array(weight_zp, "w_zp"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", "a_scale", "a_zp"],
            ["a_q"],
            name="activation_q",
            axis=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["a_q", "a_scale", "a_zp"],
            ["a_dq"],
            name="activation_dq",
            axis=1,
        ),
        helper.make_node(
            "QuantizeLinear",
            ["weight", "w_scale", "w_zp"],
            ["w_q"],
            name="weight_q",
            axis=weight_axis,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["w_q", "w_scale", "w_zp"],
            ["w_dq"],
            name="weight_dq",
            axis=weight_axis,
        ),
    ]
    weight_input = "w_dq"
    if transpose_weight:
        nodes.append(
            helper.make_node("Transpose", ["w_dq"], ["w_ready"], name="weight_transpose", perm=[1, 0])
        )
        weight_input = "w_ready"
    nodes.append(helper.make_node("MatMul", ["a_dq", weight_input], ["y"], name="linear"))
    graph = helper.make_graph(
        nodes,
        "quant_linear",
        [helper.make_tensor_value_info("x", output_dtype, activation_shape)],
        [helper.make_tensor_value_info("y", output_dtype, output_shape)],
        initializer=initializers,
        value_info=[
            helper.make_tensor_value_info("a_q", TensorProto.INT8, activation_shape),
            helper.make_tensor_value_info("a_dq", output_dtype, activation_shape),
            helper.make_tensor_value_info("w_q", TensorProto.INT8, list(weight.shape)),
            helper.make_tensor_value_info("w_dq", output_dtype, list(weight.shape)),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def _attribute(node: onnx.NodeProto, name: str) -> int:
    return next(int(attribute.i) for attribute in node.attribute if attribute.name == name)


@pytest.mark.parametrize("output_dtype,expected_attribute", [(TensorProto.FLOAT16, 1), (TensorProto.FLOAT, 0)])
@pytest.mark.parametrize("transpose_weight", [False, True])
def test_lower_per_tensor_qdq(
    output_dtype: int,
    expected_attribute: int,
    transpose_weight: bool,
) -> None:
    model = _model(output_dtype=output_dtype, transpose_weight=transpose_weight)

    returned = lower_qdq(model)

    assert returned is model
    assert [node.op_type for node in model.graph.node] == [
        "NPUAscendQuantV2",
        "MatMul",
        "AscendDequant",
    ]
    quant, matmul, dequant = model.graph.node
    assert _attribute(quant, "dtype") == 2
    assert _attribute(dequant, "dtype") == expected_attribute
    assert matmul.input[0] == quant.output[0]
    assert dequant.input[0] == matmul.output[0]
    assert dequant.output[0] == "y"

    initializers = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    quantized_weight = initializers[matmul.input[1]]
    assert quantized_weight.dtype == np.int8
    assert quantized_weight.shape == (3, 4)
    packed = initializers[dequant.input[1]]
    expected = np.array([0.125, 0.0625, 0.03125, 0.015625], dtype=np.float32)
    np.testing.assert_array_equal(packed, expected.view(np.uint32).astype(np.uint64))


def test_lower_per_token_asymmetric_activation() -> None:
    model = _model(per_token=True, activation_zero_point=7)

    lower_qdq(model)

    assert [node.op_type for node in model.graph.node] == [
        "NPUAscendQuantV2",
        "MatMul",
        "AscendDequant",
        "Mul",
    ]
    quant, _, dequant, mul = model.graph.node
    assert _attribute(quant, "axis") == -2
    initializers = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    np.testing.assert_array_equal(initializers[quant.input[2]], np.array([7, 7], dtype=np.float16))
    np.testing.assert_array_equal(
        initializers[mul.input[1]],
        np.array([[[0.25], [0.5]]], dtype=np.float16),
    )
    packed = initializers[dequant.input[1]]
    expected = np.array([0.5, 0.25, 0.125, 0.0625], dtype=np.float32)
    np.testing.assert_array_equal(packed, expected.view(np.uint32).astype(np.uint64))


def test_nonzero_weight_zero_point_rolls_back() -> None:
    model = _model(weight_zero_point=1)
    original = model.SerializeToString()

    with pytest.raises(ValueError, match="weight quantization must be symmetric"):
        lower_qdq(model)

    assert model.SerializeToString() == original


def test_unrelated_residual_qdq_rolls_back() -> None:
    model = _model()
    model.graph.node.extend(
        [
            helper.make_node("QuantizeLinear", ["x", "a_scale", "a_zp"], ["other_q"]),
            helper.make_node(
                "DequantizeLinear",
                ["other_q", "a_scale", "a_zp"],
                ["other_dq"],
            ),
        ]
    )
    original = model.SerializeToString()

    with pytest.raises(ValueError, match="residual QDQ"):
        lower_qdq(model)

    assert model.SerializeToString() == original


def test_half_quantized_matmul_is_rejected() -> None:
    model = _model()
    model.graph.node[-1].input[0] = "x"

    with pytest.raises(ValueError, match="activation and weight must both"):
        lower_qdq(model)


def test_unquantized_attention_transpose_is_ignored() -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["key"], ["key_t"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["query", "key_t"], ["score"]),
        ],
        "attention",
        [
            helper.make_tensor_value_info("query", TensorProto.FLOAT16, [1, 4, 3, 8]),
            helper.make_tensor_value_info("key", TensorProto.FLOAT16, [1, 4, 3, 8]),
        ],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT16, [1, 4, 3, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    original = model.SerializeToString()

    returned = lower_qdq(model)

    assert returned is model
    assert model.SerializeToString() == original


def test_quantized_gemm_is_rejected() -> None:
    model = _model()
    model.graph.node[-1].op_type = "Gemm"

    with pytest.raises(ValueError, match="quantized Gemm is not supported"):
        lower_qdq(model)
