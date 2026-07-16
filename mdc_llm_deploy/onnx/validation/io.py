"""MDC runtime input and output ABI validation."""

from __future__ import annotations

import onnx
from onnx import TensorProto

from ...errors import OnnxExportError
from ..inspection import static_shape as _shape


def validate_io_abi(model: onnx.ModelProto, stage: str) -> None:
    """Validate ordered runtime names, dtypes, and static shapes."""
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    input_names = [item.name for item in inputs]
    if not input_names or input_names[0] != "input_ids":
        raise OnnxExportError("MDC runtime input ABI is invalid")
    cache_inputs = inputs[1:]
    if stage.endswith("PREFILL"):
        if cache_inputs:
            raise OnnxExportError("Prefill runtime must not accept KV cache")
    elif len(cache_inputs) % 2:
        raise OnnxExportError("Decode KV inputs must contain key/value pairs")
    else:
        expected_cache_names = [
            f"past.{layer_id}.{edge}"
            for layer_id in range(len(cache_inputs) // 2)
            for edge in ("key", "value")
        ]
        if [item.name for item in cache_inputs] != expected_cache_names:
            raise OnnxExportError("Decode KV input names are invalid")
    if [item.name for item in outputs] != ["logits"]:
        raise OnnxExportError("MDC runtime output ABI is invalid")
    input_ids = inputs[0]
    if input_ids.type.tensor_type.elem_type != TensorProto.INT64:
        raise OnnxExportError("input_ids must use INT64")
    logits_shape = _shape(outputs[0])
    if len(logits_shape) != 3 or logits_shape[0] != 1:
        raise OnnxExportError("logits ABI shape is invalid")
    if stage.endswith("PREFILL"):
        if _shape(input_ids) != (1, logits_shape[1]):
            raise OnnxExportError(
                "Prefill sequence ABI is inconsistent"
            )
        return
    if _shape(input_ids) != (1, 1) or logits_shape[1] != 1:
        raise OnnxExportError("Decode query ABI must use one token")
    for index in range(0, len(cache_inputs), 2):
        key_shape = _shape(cache_inputs[index])
        value_shape = _shape(cache_inputs[index + 1])
        if (
            len(key_shape) != 4
            or key_shape != value_shape
            or key_shape[0] != 1
        ):
            raise OnnxExportError("Decode KV input ABI shape is invalid")
