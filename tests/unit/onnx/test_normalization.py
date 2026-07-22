from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.onnx import normalize_graph


def _model(
    nodes: list[onnx.NodeProto],
    *,
    output_name: str = "y",
    elem_type: int = TensorProto.FLOAT,
    value_info: list[onnx.ValueInfoProto] | None = None,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "identity_normalization",
        [helper.make_tensor_value_info("x", elem_type, [2, 3])],
        [helper.make_tensor_value_info(output_name, elem_type, [2, 3])],
        value_info=value_info,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_rewires_all_consumers_and_removes_identity_metadata() -> None:
    model = _model(
        [
            helper.make_node("Identity", ["x"], ["alias"], name="identity"),
            helper.make_node("Relu", ["alias"], ["relu"]),
            helper.make_node("Neg", ["alias"], ["y"]),
        ],
        value_info=[
            helper.make_tensor_value_info("alias", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("relu", TensorProto.FLOAT, [2, 3]),
        ],
    )

    returned = normalize_graph(model)

    assert returned is model
    assert [node.op_type for node in model.graph.node] == ["Relu", "Neg"]
    assert [node.input[0] for node in model.graph.node] == ["x", "x"]
    assert all(value.name != "alias" for value in model.graph.value_info)
    onnx.checker.check_model(model, full_check=True)


def test_collapses_identity_chain_and_is_idempotent() -> None:
    model = _model(
        [
            helper.make_node("Identity", ["x"], ["first"]),
            helper.make_node("Identity", ["first"], ["second"]),
            helper.make_node("Neg", ["second"], ["y"]),
        ],
        value_info=[
            helper.make_tensor_value_info("first", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("second", TensorProto.FLOAT, [2, 3]),
        ],
    )

    normalize_graph(model)
    first_result = model.SerializeToString()
    normalize_graph(model)

    assert len(model.graph.node) == 1
    assert list(model.graph.node[0].input) == ["x"]
    assert model.SerializeToString() == first_result
    onnx.checker.check_model(model, full_check=True)


def test_preserves_identity_that_defines_graph_output() -> None:
    model = _model([helper.make_node("Identity", ["x"], ["y"], name="contract")])
    original = model.SerializeToString()

    normalize_graph(model)

    assert model.SerializeToString() == original
    onnx.checker.check_model(model, full_check=True)


def test_preserves_metadata_when_identity_input_has_no_value_info() -> None:
    model = _model(
        [
            helper.make_node("Relu", ["x"], ["hidden"]),
            helper.make_node("Identity", ["hidden"], ["alias"]),
            helper.make_node("Neg", ["alias"], ["y"]),
        ],
        value_info=[
            helper.make_tensor_value_info("alias", TensorProto.FLOAT, [2, 3]),
        ],
    )

    normalize_graph(model)

    hidden_info = next(value for value in model.graph.value_info if value.name == "hidden")
    assert hidden_info.type.tensor_type.elem_type == TensorProto.FLOAT
    assert [dimension.dim_value for dimension in hidden_info.type.tensor_type.shape.dim] == [2, 3]
    assert model.graph.node[-1].input[0] == "hidden"
    onnx.checker.check_model(model, full_check=True)


def test_removes_lossless_float_cast_round_trip() -> None:
    model = _model(
        [
            helper.make_node("Cast", ["x"], ["wide"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["wide"], ["restored"], to=TensorProto.FLOAT16),
            helper.make_node("Neg", ["restored"], ["y"]),
        ],
        elem_type=TensorProto.FLOAT16,
        value_info=[
            helper.make_tensor_value_info("wide", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("restored", TensorProto.FLOAT16, [2, 3]),
        ],
    )

    normalize_graph(model)

    assert [node.op_type for node in model.graph.node] == ["Neg"]
    assert list(model.graph.node[0].input) == ["x"]
    assert all(
        value.name not in {"wide", "restored"} for value in model.graph.value_info
    )
    onnx.checker.check_model(model, full_check=True)


def test_preserves_lossy_float_cast_round_trip() -> None:
    model = _model(
        [
            helper.make_node("Cast", ["x"], ["narrow"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["narrow"], ["restored"], to=TensorProto.FLOAT),
            helper.make_node("Neg", ["restored"], ["y"]),
        ],
        value_info=[
            helper.make_tensor_value_info("narrow", TensorProto.FLOAT16, [2, 3]),
            helper.make_tensor_value_info("restored", TensorProto.FLOAT, [2, 3]),
        ],
    )

    original = model.SerializeToString()
    normalize_graph(model)

    assert model.SerializeToString() == original
    onnx.checker.check_model(model, full_check=True)


def test_folds_constant_float_cast_and_removes_source_initializer() -> None:
    model = _model(
        [
            helper.make_node("Cast", ["scale_fp32"], ["scale"], to=TensorProto.FLOAT16),
            helper.make_node("Mul", ["x", "scale"], ["y"]),
        ],
        elem_type=TensorProto.FLOAT16,
        value_info=[
            helper.make_tensor_value_info("scale", TensorProto.FLOAT16, []),
        ],
    )
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "scale_fp32")
    )
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), "unrelated")
    )

    normalize_graph(model)

    assert [node.op_type for node in model.graph.node] == ["Mul"]
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    assert set(initializers) == {"scale", "unrelated"}
    assert numpy_helper.to_array(initializers["scale"]) == np.float16(0.5)
    onnx.checker.check_model(model, full_check=True)


def test_rejects_non_model_input_without_mutating_argument() -> None:
    invalid = helper.make_graph([], "invalid", [], [])

    try:
        normalize_graph(invalid)  # type: ignore[arg-type]
    except TypeError as error:
        assert str(error) == "model must be an onnx.ModelProto"
    else:
        raise AssertionError("normalize_graph must reject GraphProto")
