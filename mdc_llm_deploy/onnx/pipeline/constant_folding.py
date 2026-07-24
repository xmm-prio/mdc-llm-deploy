"""Fold constant ONNX subgraphs."""

from __future__ import annotations

import onnx
from onnxscript import optimizer


def fold_constants_core(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fold constant subgraphs in place."""
    optimizer.fold_constants(model)
    return model
