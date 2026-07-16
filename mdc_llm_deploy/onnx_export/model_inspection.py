"""Shared primitives for inspecting ONNX model structure."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import onnx
from onnx import helper

from ..errors import OnnxExportError


def optional_static_shape(
    value: onnx.ValueInfoProto,
) -> tuple[int, ...] | None:
    """Return a fully static positive shape, or None."""
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dimensions: list[int] = []
    for dimension in tensor_type.shape.dim:
        if (
            not dimension.HasField("dim_value")
            or dimension.dim_value <= 0
        ):
            return None
        dimensions.append(dimension.dim_value)
    return tuple(dimensions)


def static_shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    """Return a fully static positive tensor shape."""
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise OnnxExportError(
            f"Value {value.name!r} has no shape"
        )
    result = optional_static_shape(value)
    if result is None:
        raise OnnxExportError(
            f"Value {value.name!r} has a dynamic shape"
        )
    return result


def node_attributes(
    node: onnx.NodeProto,
) -> dict[str, onnx.AttributeProto]:
    """Return node attributes indexed by name."""
    return {item.name: item for item in node.attribute}


def decoded_node_attributes(
    node: onnx.NodeProto,
) -> dict[str, Any]:
    """Return decoded node attribute values indexed by name."""
    return {
        item.name: helper.get_attribute_value(item)
        for item in node.attribute
    }


def require_attributes(
    node: onnx.NodeProto,
    required: Collection[str] | Mapping[str, int],
) -> dict[str, onnx.AttributeProto]:
    """Return attributes after checking names and optional wire types."""
    attributes = node_attributes(node)
    required_names = set(required)
    missing = required_names - attributes.keys()
    if missing:
        raise OnnxExportError(
            f"{node.op_type} attributes are incomplete: "
            f"{sorted(missing)}"
        )
    if isinstance(required, Mapping):
        invalid = [
            name
            for name, expected_type in required.items()
            if attributes[name].type != expected_type
        ]
        if invalid:
            raise OnnxExportError(
                f"{node.op_type} attributes have invalid ONNX "
                f"types: {sorted(invalid)}"
            )
    return attributes
