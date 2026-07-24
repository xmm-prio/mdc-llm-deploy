"""Conservative default-domain ONNX opset downgrade for MDC deployment."""

from __future__ import annotations

from collections.abc import Iterator

import onnx
from onnx import GraphProto, NodeProto, defs

from ._graph import clone_model
from .schemas import ASCEND_DEQUANT_OP, ASCEND_QUANT_OP, MDC_ONNX_OPSET

_MDC_OPERATORS = frozenset({ASCEND_QUANT_OP, ASCEND_DEQUANT_OP})


def _nodes(graph: GraphProto) -> Iterator[NodeProto]:
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _nodes(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _nodes(subgraph)


def _default_opset(model: onnx.ModelProto) -> int:
    versions = [
        int(opset.version)
        for opset in model.opset_import
        if opset.domain in ("", "ai.onnx")
    ]
    if len(versions) != 1:
        raise ValueError("model must contain exactly one default-domain opset import")
    return versions[0]


def _schema(node: NodeProto, version: int) -> defs.OpSchema:
    try:
        return defs.get_schema(node.op_type, version, "")
    except defs.SchemaError as error:
        raise ValueError(
            f"operator '{node.op_type}' has no default-domain schema at opset {version}"
        ) from error


def _validate_compatible(model: onnx.ModelProto, source_opset: int) -> None:
    for node in _nodes(model.graph):
        if node.domain not in ("", "ai.onnx"):
            continue
        _schema(node, source_opset)
        target = _schema(node, MDC_ONNX_OPSET)
        if node.op_type in _MDC_OPERATORS:
            continue
        unsupported_attributes = {
            attribute.name for attribute in node.attribute
        }.difference(target.attributes)
        if unsupported_attributes:
            names = ", ".join(sorted(unsupported_attributes))
            raise ValueError(
                f"operator '{node.name or node.op_type}' uses attributes unavailable at "
                f"opset {MDC_ONNX_OPSET}: {names}"
            )


def downgrade_opset_core(model: onnx.ModelProto) -> onnx.ModelProto:
    """Mutate a working ModelProto by safely setting its default opset to 18."""
    source_opset = _default_opset(model)
    if source_opset < MDC_ONNX_OPSET:
        raise ValueError(
            f"default opset {source_opset} is below target opset {MDC_ONNX_OPSET}; "
            "opset upgrade is not supported"
        )
    _validate_compatible(model, source_opset)
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.domain = ""
            opset.version = MDC_ONNX_OPSET
            break
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as error:
        raise ValueError(
            f"model is not valid at default opset {MDC_ONNX_OPSET}: {error}"
        ) from error
    return model


def downgrade_opset(model: onnx.ModelProto) -> onnx.ModelProto:
    """Atomically downgrade the default opset to 18 in place and return the same model."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an onnx.ModelProto")
    working = clone_model(model)
    downgrade_opset_core(working)
    model.CopyFrom(working)
    return model


__all__ = ["downgrade_opset"]
