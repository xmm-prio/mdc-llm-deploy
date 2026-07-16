"""Finalize the public runtime output contract."""

from __future__ import annotations

import onnx

from ...errors import OnnxExportError


def retain_logits_output(model: onnx.ModelProto) -> None:
    """Expose logits as the only public graph output."""
    logits = [
        output for output in model.graph.output if output.name == "logits"
    ]
    if len(logits) != 1:
        raise OnnxExportError(
            f"ONNX output finalization found {len(logits)} logits outputs"
        )
    del model.graph.output[:]
    model.graph.output.extend(logits)
