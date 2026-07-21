"""Atomic MDC ONNX graph-processing pipeline."""

from __future__ import annotations

from collections.abc import Iterator

import onnx
from onnx import GraphProto, NodeProto

from ._graph import clone_model
from .fusion_pass import run_fusion_passes
from .opset_downgrade import downgrade_opset_core
from .quant_lowering import lower_qdq_core
from .schemas import ALL_SCHEMA_NAMES, register_schemas

_CUSTOM_SCHEMA_NAMES = frozenset(ALL_SCHEMA_NAMES)


def _nodes(graph: GraphProto) -> Iterator[NodeProto]:
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _nodes(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _nodes(subgraph)


def _register_required_schemas(model: onnx.ModelProto) -> None:
    required = tuple(
        dict.fromkeys(
            node.op_type
            for node in _nodes(model.graph)
            if node.domain in ("", "ai.onnx") and node.op_type in _CUSTOM_SCHEMA_NAMES
        )
    )
    if required:
        register_schemas(*required)


def _validate_final_graph(model: onnx.ModelProto) -> None:
    residual = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.domain in ("", "ai.onnx")
            and node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        }
    )
    if residual:
        raise ValueError(f"main graph still contains residual QDQ operators: {residual}")
    onnx.checker.check_model(model)


def process_onnx(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run the atomic MDC pipeline in place and return the same ModelProto."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an onnx.ModelProto")
    working = clone_model(model)
    lower_qdq_core(working)
    _register_required_schemas(working)
    downgrade_opset_core(working)
    run_fusion_passes(working)
    _register_required_schemas(working)
    _validate_final_graph(working)
    model.CopyFrom(working)
    return model


__all__ = ["process_onnx"]
