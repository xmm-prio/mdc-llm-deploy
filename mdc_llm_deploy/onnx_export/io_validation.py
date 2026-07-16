"""MDC runtime input and output ABI validation."""

from __future__ import annotations

import onnx
from onnx import TensorProto

from ..errors import OnnxExportError
from .model_inspection import static_shape as _shape


def validate_io_abi(model: onnx.ModelProto, stage: str) -> None:
    """Validate ordered runtime names, dtypes, and static shapes."""
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    expected_inputs = (
        ["input_ids"]
        if stage.endswith("PREFILL")
        else [
            "input_ids",
            "past_key_values.0.key",
            "past_key_values.0.value",
        ]
    )
    if [item.name for item in inputs] != expected_inputs:
        raise OnnxExportError("MDC runtime input ABI is invalid")
    if [item.name for item in outputs] != [
        "logits",
        "present.0.key",
        "present.0.value",
    ]:
        raise OnnxExportError("MDC runtime output ABI is invalid")
    input_ids = inputs[0]
    if input_ids.type.tensor_type.elem_type != TensorProto.INT64:
        raise OnnxExportError("input_ids must use INT64")
    logits_shape = _shape(outputs[0])
    key_shape = _shape(outputs[1])
    value_shape = _shape(outputs[2])
    if len(logits_shape) != 3 or logits_shape[0] != 1:
        raise OnnxExportError("logits ABI shape is invalid")
    if (
        len(key_shape) != 4
        or key_shape != value_shape
        or key_shape[0] != 1
    ):
        raise OnnxExportError("KV output ABI shape is invalid")
    if stage.endswith("PREFILL"):
        if (
            _shape(input_ids) != (1, logits_shape[1])
            or key_shape[2] != logits_shape[1]
        ):
            raise OnnxExportError(
                "Prefill sequence ABI is inconsistent"
            )
        return
    if _shape(input_ids) != (1, 1) or logits_shape[1] != 1:
        raise OnnxExportError("Decode query ABI must use one token")
    expected_key = (
        key_shape[0],
        key_shape[1],
        key_shape[2] - 1,
        key_shape[3],
    )
    if _shape(inputs[1]) != expected_key:
        raise OnnxExportError(
            "Decode key cache must omit the current token"
        )
    expected_value = (
        value_shape[0],
        value_shape[1],
        value_shape[2] - 1,
        value_shape[3],
    )
    if _shape(inputs[2]) != expected_value:
        raise OnnxExportError(
            "Decode value cache must omit the current token"
        )
