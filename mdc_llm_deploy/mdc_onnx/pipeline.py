"""Atomic MDC ONNX graph-processing pipeline."""

from __future__ import annotations

import onnx

from ._graph import clone_model
from .opset_downgrade import downgrade_opset_core
from .quant_lowering import lower_qdq_core


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
    """Atomically lower QDQ and downgrade opset in place, returning the same ModelProto."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an onnx.ModelProto")
    working = clone_model(model)
    lower_qdq_core(working)
    downgrade_opset_core(working)
    _validate_final_graph(working)
    model.CopyFrom(working)
    return model


__all__ = ["process_onnx"]
