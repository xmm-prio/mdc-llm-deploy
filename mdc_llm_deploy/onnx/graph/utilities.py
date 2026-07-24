"""Shared protobuf graph helpers for MDC ONNX transformations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, ValueInfoProto, helper, numpy_helper


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """Static tensor metadata used by strict lowering validation."""

    elem_type: int
    shape: tuple[int | str | None, ...]


class GraphIndex:
    """Immutable connectivity and tensor metadata index for one ONNX main graph."""

    def __init__(self, model: onnx.ModelProto) -> None:
        graph = model.graph
        self.producers = {output: node for node in graph.node for output in node.output if output}
        consumers: defaultdict[str, list[NodeProto]] = defaultdict(list)
        for node in graph.node:
            for input_name in node.input:
                if input_name:
                    consumers[input_name].append(node)
        self.consumers = dict(consumers)
        self.initializers = {tensor.name: tensor for tensor in graph.initializer}
        self.tensor_info: dict[str, TensorInfo] = {}
        for value in [*graph.input, *graph.value_info, *graph.output]:
            info = tensor_info(value)
            if info is not None:
                self.tensor_info[value.name] = info
        for tensor in graph.initializer:
            self.tensor_info[tensor.name] = TensorInfo(
                tensor.data_type,
                tuple(int(dimension) for dimension in tensor.dims),
            )

    def producer(self, value_name: str) -> NodeProto | None:
        """Return value producer, if any."""
        return self.producers.get(value_name)

    def users(self, value_name: str) -> list[NodeProto]:
        """Return all main-graph consumers of one value."""
        return self.consumers.get(value_name, [])


def clone_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Clone a ModelProto without filesystem access."""
    cloned = onnx.ModelProto()
    cloned.CopyFrom(model)
    return cloned


def tensor_info(value: ValueInfoProto) -> TensorInfo | None:
    """Read tensor dtype and shape from ValueInfo."""
    tensor_type = value.type.tensor_type
    if tensor_type.elem_type == TensorProto.UNDEFINED:
        return None
    dimensions: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return TensorInfo(int(tensor_type.elem_type), tuple(dimensions))


def attribute_int(node: NodeProto, name: str, default: int | None = None) -> int | None:
    """Read an integer node attribute."""
    for attribute in node.attribute:
        if attribute.name == name:
            if attribute.type != onnx.AttributeProto.INT:
                raise ValueError(f"node '{node.name or node.op_type}' attribute '{name}' must be INT")
            return int(attribute.i)
    return default


def attribute_ints(node: NodeProto, name: str) -> tuple[int, ...] | None:
    """Read an integer-list node attribute."""
    for attribute in node.attribute:
        if attribute.name == name:
            if attribute.type != onnx.AttributeProto.INTS:
                raise ValueError(f"node '{node.name or node.op_type}' attribute '{name}' must be INTS")
            return tuple(int(value) for value in attribute.ints)
    return None


def constant_array(index: GraphIndex, value_name: str) -> np.ndarray | None:
    """Evaluate supported constant-only values, or return None when dynamic."""
    return _constant_array(index, value_name, set())


def _constant_array(
    index: GraphIndex,
    value_name: str,
    visiting: set[str],
) -> np.ndarray | None:
    if value_name in visiting:
        return None
    visiting.add(value_name)
    try:
        return _constant_array_impl(index, value_name, visiting)
    finally:
        visiting.remove(value_name)


def _constant_array_impl(
    index: GraphIndex,
    value_name: str,
    visiting: set[str],
) -> np.ndarray | None:
    tensor = index.initializers.get(value_name)
    if tensor is not None:
        if tensor.data_location == TensorProto.EXTERNAL and not tensor.raw_data:
            raise ValueError(
                f"initializer '{value_name}' uses unloaded external data; "
                "load it before calling MDC ONNX"
            )
        return np.asarray(numpy_helper.to_array(tensor))

    producer = index.producer(value_name)
    if producer is None:
        return None
    if producer.op_type == "Constant":
        for attribute in producer.attribute:
            if attribute.name == "value" and attribute.type == onnx.AttributeProto.TENSOR:
                return np.asarray(numpy_helper.to_array(attribute.t))
            if attribute.name == "value_int" and attribute.type == onnx.AttributeProto.INT:
                return np.asarray(attribute.i, dtype=np.int64)
            if attribute.name == "value_ints" and attribute.type == onnx.AttributeProto.INTS:
                return np.asarray(attribute.ints, dtype=np.int64)
            if attribute.name == "value_float" and attribute.type == onnx.AttributeProto.FLOAT:
                return np.asarray(attribute.f, dtype=np.float32)
            if attribute.name == "value_floats" and attribute.type == onnx.AttributeProto.FLOATS:
                return np.asarray(attribute.floats, dtype=np.float32)
        return None
    if producer.op_type == "Identity" and len(producer.input) == 1:
        return _constant_array(index, producer.input[0], visiting)
    if producer.op_type == "Cast" and len(producer.input) == 1:
        source = _constant_array(index, producer.input[0], visiting)
        destination_type = attribute_int(producer, "to")
        if source is None or destination_type is None:
            return None
        try:
            return source.astype(helper.tensor_dtype_to_np_dtype(destination_type))
        except (TypeError, ValueError):
            return None
    if producer.op_type != "Reshape" or len(producer.input) != 2:
        return None

    source = _constant_array(index, producer.input[0], visiting)
    shape_array = _constant_array(index, producer.input[1], visiting)
    if source is None or shape_array is None or not np.issubdtype(shape_array.dtype, np.integer):
        return None
    shape = [int(dimension) for dimension in shape_array.reshape(-1)]
    if attribute_int(producer, "allowzero", 0) == 0:
        shape = [
            source.shape[index] if dimension == 0 and index < source.ndim else dimension
            for index, dimension in enumerate(shape)
        ]
    try:
        return np.asarray(source.reshape(shape))
    except ValueError:
        return None


def unique_name(existing: set[str], preferred: str) -> str:
    """Reserve and return a deterministic unique graph name."""
    base = preferred or "mdc_value"
    candidate = base
    suffix = 1
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    existing.add(candidate)
    return candidate


def graph_names(model: onnx.ModelProto) -> set[str]:
    """Collect node, value and initializer names from the main graph."""
    graph = model.graph
    names = {node.name for node in graph.node if node.name}
    names.update(tensor.name for tensor in graph.initializer)
    for node in graph.node:
        names.update(name for name in [*node.input, *node.output] if name)
    names.update(value.name for value in [*graph.input, *graph.value_info, *graph.output])
    return names


def remove_unused_initializers(model: onnx.ModelProto) -> None:
    """Remove main-graph initializers with no consumer or graph-output use."""
    graph = model.graph
    used = {name for node in graph.node for name in node.input if name}
    used.update(value.name for value in graph.output)
    kept = [tensor for tensor in graph.initializer if tensor.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)


def remove_value_info(model: onnx.ModelProto, names: Iterable[str]) -> None:
    """Remove stale non-contract ValueInfo entries."""
    stale = set(names)
    graph = model.graph
    kept = [value for value in graph.value_info if value.name not in stale]
    del graph.value_info[:]
    graph.value_info.extend(kept)


__all__ = [
    "GraphIndex",
    "TensorInfo",
    "attribute_int",
    "attribute_ints",
    "clone_model",
    "constant_array",
    "graph_names",
    "remove_unused_initializers",
    "remove_value_info",
    "unique_name",
]
