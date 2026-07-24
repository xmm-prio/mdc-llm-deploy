"""Lower standard ONNX constructs unsupported by the MDC parser."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, NodeProto, ValueInfoProto, numpy_helper

from ..graph import clone_model


def _static_shape(value: ValueInfoProto) -> tuple[int, ...] | None:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dimensions = tensor_type.shape.dim
    if any(
        not dimension.HasField("dim_value") or dimension.dim_value < 0
        for dimension in dimensions
    ):
        return None
    return tuple(dimension.dim_value for dimension in dimensions)


def _known_shapes(
    graph: GraphProto,
    inherited: Mapping[str, tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    shapes = dict(inherited)
    for value in (*graph.input, *graph.value_info, *graph.output):
        shape = _static_shape(value)
        if shape is not None:
            shapes[value.name] = shape
    for tensor in graph.initializer:
        shapes[tensor.name] = tuple(tensor.dims)
    return shapes


def _attribute(node: NodeProto, name: str) -> AttributeProto | None:
    return next((attribute for attribute in node.attribute if attribute.name == name), None)


def _split_sizes(dimension: int, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("Split num_outputs must be positive")
    if dimension > 0 and count > dimension:
        raise ValueError(
            f"Split num_outputs {count} exceeds the non-empty axis dimension {dimension}"
        )
    chunk = (dimension + count - 1) // count
    return (*([chunk] * (count - 1)), dimension - chunk * (count - 1))


def _unique_name(graph: GraphProto, base: str) -> str:
    names = {
        name
        for node in graph.node
        for name in (*node.input, *node.output)
        if name
    }
    names.update(value.name for value in (*graph.input, *graph.value_info, *graph.output))
    names.update(tensor.name for tensor in graph.initializer)
    if base not in names:
        return base
    suffix = 1
    while f"{base}_{suffix}" in names:
        suffix += 1
    return f"{base}_{suffix}"


def _lower_split(
    graph: GraphProto,
    node: NodeProto,
    shapes: Mapping[str, tuple[int, ...]],
) -> None:
    num_outputs = _attribute(node, "num_outputs")
    if num_outputs is None:
        return
    if num_outputs.type != AttributeProto.INT:
        raise ValueError(f"Split node {node.name!r} has non-integer num_outputs")
    if len(node.input) != 1:
        raise ValueError(
            f"Split node {node.name!r} cannot use both split input and num_outputs"
        )
    input_shape = shapes.get(node.input[0])
    if input_shape is None:
        raise ValueError(
            f"Split node {node.name!r} requires a static input shape to lower num_outputs"
        )
    axis_attribute = _attribute(node, "axis")
    if axis_attribute is not None and axis_attribute.type != AttributeProto.INT:
        raise ValueError(f"Split node {node.name!r} has non-integer axis")
    axis = 0 if axis_attribute is None else axis_attribute.i
    if not -len(input_shape) <= axis < len(input_shape):
        raise ValueError(f"Split node {node.name!r} has invalid axis {axis}")
    dimension = input_shape[axis]
    sizes = _split_sizes(dimension, num_outputs.i)
    if len(sizes) != len(node.output):
        raise ValueError(
            f"Split node {node.name!r} num_outputs does not match its outputs"
        )

    sizes_name = _unique_name(graph, f"{node.name or node.output[0]}_split_sizes")
    graph.initializer.append(
        numpy_helper.from_array(np.asarray(sizes, dtype=np.int64), sizes_name)
    )
    node.input.append(sizes_name)
    node.attribute.remove(num_outputs)


def _lower_expand(
    graph: GraphProto,
    node: NodeProto,
    shapes: Mapping[str, tuple[int, ...]],
) -> None:
    input_shape = shapes.get(node.input[0])
    output_shape = shapes.get(node.output[0])
    if input_shape is None or output_shape is None or len(input_shape) != len(output_shape):
        return
    if input_shape == output_shape:
        node.op_type = "Identity"
        del node.input[1:]
        return

    repeats: list[int] = []
    for source, target in zip(input_shape, output_shape, strict=True):
        if source == target:
            repeats.append(1)
        elif source == 1:
            repeats.append(target)
        else:
            return
    repeats_name = _unique_name(graph, f"{node.name or node.output[0]}_repeats")
    graph.initializer.append(
        numpy_helper.from_array(np.asarray(repeats, dtype=np.int64), repeats_name)
    )
    node.op_type = "Tile"
    node.input[1] = repeats_name


def _lower_graph(
    graph: GraphProto,
    inherited_shapes: Mapping[str, tuple[int, ...]],
) -> None:
    shapes = _known_shapes(graph, inherited_shapes)
    for node in graph.node:
        if node.domain in ("", "ai.onnx"):
            if node.op_type == "Split":
                _lower_split(graph, node, shapes)
            elif node.op_type == "Expand":
                _lower_expand(graph, node, shapes)
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                _lower_graph(attribute.g, shapes)
            elif attribute.type == AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    _lower_graph(subgraph, shapes)


def lower_opset_compatibility_core(model: onnx.ModelProto) -> onnx.ModelProto:
    """Mutate a working model to remove MDC parser-incompatible ONNX forms."""
    inferred = onnx.shape_inference.infer_shapes(
        model,
        strict_mode=False,
        data_prop=True,
    )
    model.CopyFrom(inferred)
    _lower_graph(model.graph, {})
    return model


def lower_opset_compatibility(model: onnx.ModelProto) -> onnx.ModelProto:
    """Atomically lower MDC parser-incompatible ONNX forms in place."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an onnx.ModelProto")
    working = clone_model(model)
    lower_opset_compatibility_core(working)
    onnx.checker.check_model(working)
    model.CopyFrom(working)
    return model


__all__ = ["lower_opset_compatibility"]
