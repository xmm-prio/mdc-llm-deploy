"""Atomic MDC ONNX graph-processing pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import onnx
from onnx import GraphProto, NodeProto

from .._observability import get_logger, log_stage, progress_task
from ._graph import clone_model
from .compatibility_lowering import lower_opset_compatibility_core
from .fusion_pass import FusionPass, run_fusion_passes
from .normalization import normalize_graph_core
from .opset_downgrade import downgrade_opset_core
from .quant_lowering import lower_qdq_core
from .schemas import ALL_SCHEMA_NAMES, register_schemas

_CUSTOM_SCHEMA_NAMES = frozenset(ALL_SCHEMA_NAMES)
_logger = get_logger(__name__)


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


def _run_stage(
    model: onnx.ModelProto,
    name: str,
    operation: Callable[[onnx.ModelProto], object],
) -> None:
    before = sum(1 for _ in _nodes(model.graph))
    with log_stage(_logger, f"ONNX {name}", details=f"nodes={before}"):
        operation(model)
    after = sum(1 for _ in _nodes(model.graph))
    _logger.info("ONNX %s node change: before=%d after=%d delta=%+d", name, before, after, after - before)


def process_onnx(
    model: onnx.ModelProto,
    *,
    show_progress: bool = True,
    fusion_passes: Sequence[FusionPass] | None = None,
) -> onnx.ModelProto:
    """Run the atomic MDC pipeline with optional ordered fusion-pass selection."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an onnx.ModelProto")
    working = clone_model(model)
    stages: tuple[tuple[str, Callable[[onnx.ModelProto], object]], ...] = (
        ("QDQ lowering", lower_qdq_core),
        ("schema registration before lowering", _register_required_schemas),
        ("compatibility lowering", lower_opset_compatibility_core),
        ("opset downgrade", downgrade_opset_core),
        ("graph normalization", normalize_graph_core),
        ("fusion", lambda graph: run_fusion_passes(graph, passes=fusion_passes)),
        ("schema registration after fusion", _register_required_schemas),
        ("final validation", _validate_final_graph),
    )
    _logger.info("ONNX pipeline started: nodes=%d", sum(1 for _ in _nodes(working.graph)))
    with progress_task(
        "Processing ONNX pipeline",
        total=len(stages),
        show_progress=show_progress,
    ) as advance:
        for name, operation in stages:
            _run_stage(working, name, operation)
            advance()
    model.CopyFrom(working)
    _logger.info("ONNX pipeline completed: nodes=%d", sum(1 for _ in _nodes(model.graph)))
    return model


__all__ = ["process_onnx"]
