from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.onnx.pipeline.compatibility import lower_opset_compatibility


def _split_model(
    shape: list[int | str],
    *,
    axis: int = -1,
    output_shapes: tuple[list[int], ...] = ([2, 4], [2, 4], [2, 2]),
) -> onnx.ModelProto:
    outputs = [f"y_{index}" for index in range(len(output_shapes))]
    graph = helper.make_graph(
        [
            helper.make_node(
                "Split",
                ["x"],
                outputs,
                name="split",
                axis=axis,
                num_outputs=len(outputs),
            )
        ],
        "split",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape)
            for name, output_shape in zip(outputs, output_shapes, strict=True)
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_num_outputs_is_lowered_to_static_split_input() -> None:
    model = _split_model([2, 10])

    returned = lower_opset_compatibility(model)

    assert returned is model
    node = model.graph.node[0]
    assert len(node.input) == 2
    assert {attribute.name for attribute in node.attribute} == {"axis"}
    initializer = next(
        tensor for tensor in model.graph.initializer if tensor.name == node.input[1]
    )
    np.testing.assert_array_equal(numpy_helper.to_array(initializer), [4, 4, 2])
    onnx.checker.check_model(model)


def test_even_num_outputs_uses_equal_split_sizes() -> None:
    model = _split_model(
        [6, 32],
        axis=1,
        output_shapes=([6, 16], [6, 16]),
    )

    lower_opset_compatibility(model)

    initializer = model.graph.initializer[0]
    np.testing.assert_array_equal(numpy_helper.to_array(initializer), [16, 16])


def test_dynamic_num_outputs_is_rejected_atomically() -> None:
    model = _split_model(["batch", 10])
    model.graph.node[0].attribute[0].i = 0
    original = model.SerializeToString()

    with pytest.raises(ValueError, match="requires a static input shape"):
        lower_opset_compatibility(model)

    assert model.SerializeToString() == original


def test_num_outputs_larger_than_non_empty_axis_is_rejected_atomically() -> None:
    model = _split_model(
        [1],
        axis=0,
        output_shapes=([1], [0], [0]),
    )
    original = model.SerializeToString()

    with pytest.raises(ValueError, match="exceeds the non-empty axis dimension"):
        lower_opset_compatibility(model)

    assert model.SerializeToString() == original


def test_non_integer_axis_is_rejected_atomically() -> None:
    model = _split_model([2, 10])
    axis = next(attribute for attribute in model.graph.node[0].attribute if attribute.name == "axis")
    axis.type = onnx.AttributeProto.FLOAT
    axis.f = -1.0
    original = model.SerializeToString()

    with pytest.raises(ValueError, match="non-integer axis"):
        lower_opset_compatibility(model)

    assert model.SerializeToString() == original


def test_nested_graph_split_uses_captured_static_shape() -> None:
    def branch(name: str) -> onnx.GraphProto:
        return helper.make_graph(
            [
                helper.make_node(
                    "Split",
                    ["x"],
                    [f"{name}_0", f"{name}_1", f"{name}_2"],
                    name=f"{name}_split",
                    axis=-1,
                    num_outputs=3,
                )
            ],
            name,
            [],
            [helper.make_tensor_value_info(f"{name}_0", TensorProto.FLOAT, [2, 4])],
        )

    graph = helper.make_graph(
        [
            helper.make_node(
                "If",
                ["condition"],
                ["y"],
                then_branch=branch("then"),
                else_branch=branch("else"),
            )
        ],
        "nested",
        [
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 10]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    lower_opset_compatibility(model)

    branches = [
        attribute.g
        for attribute in model.graph.node[0].attribute
        if attribute.type == onnx.AttributeProto.GRAPH
    ]
    assert len(branches) == 2
    for nested in branches:
        split = nested.node[0]
        assert len(split.input) == 2
        assert {attribute.name for attribute in split.attribute} == {"axis"}
        np.testing.assert_array_equal(
            numpy_helper.to_array(nested.initializer[0]),
            [4, 4, 2],
        )
    onnx.checker.check_model(model, full_check=True)


def test_existing_split_input_is_unchanged() -> None:
    model = _split_model([2, 10])
    node = model.graph.node[0]
    node.attribute.remove(next(attribute for attribute in node.attribute if attribute.name == "num_outputs"))
    initializer = numpy_helper.from_array(np.array([4, 4, 2], dtype=np.int64), "sizes")
    model.graph.initializer.append(initializer)
    node.input.append(initializer.name)
    original = model.SerializeToString()

    lower_opset_compatibility(model)

    assert model.SerializeToString() == original


def test_static_expand_is_lowered_to_tile() -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Max", ["x", "target"], ["max_output"]),
            helper.make_node("Shape", ["max_output"], ["target_shape"]),
            helper.make_node("Expand", ["x", "target_shape"], ["y"], name="expand"),
        ],
        "expand",
        [
            helper.make_tensor_value_info("x", TensorProto.INT64, [1, 1, 1, 1]),
            helper.make_tensor_value_info("target", TensorProto.INT64, [1, 1, 1, 4]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [1, 1, 1, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    lower_opset_compatibility(model)

    expand = next(node for node in model.graph.node if node.name == "expand")
    assert expand.op_type == "Tile"
    repeats = next(
        tensor for tensor in model.graph.initializer if tensor.name == expand.input[1]
    )
    np.testing.assert_array_equal(numpy_helper.to_array(repeats), [1, 1, 1, 4])
    onnx.checker.check_model(model, full_check=True)


def test_identity_expand_is_lowered_to_identity() -> None:
    shape = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "shape")
    graph = helper.make_graph(
        [helper.make_node("Expand", ["x", "shape"], ["y"], name="expand")],
        "identity_expand",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        [shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    lower_opset_compatibility(model)

    expand = model.graph.node[0]
    assert expand.op_type == "Identity"
    assert list(expand.input) == ["x"]
    onnx.checker.check_model(model, full_check=True)
