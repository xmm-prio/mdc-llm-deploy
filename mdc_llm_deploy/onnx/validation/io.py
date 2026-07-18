"""MDC runtime input and output ABI validation."""

from __future__ import annotations

from collections.abc import Mapping

import onnx
from onnx import TensorProto

from ...errors import GraphStateError, OnnxExportError
from ...graph.metadata import GraphMetadata, TensorAbi, derive_artifact_io_abi
from ..inspection import static_shape as _shape

_ONNX_DTYPES = {
    "bool": TensorProto.BOOL,
    "bfloat16": TensorProto.BFLOAT16,
    "float16": TensorProto.FLOAT16,
    "float32": TensorProto.FLOAT,
    "int8": TensorProto.INT8,
    "int16": TensorProto.INT16,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
    "uint64": TensorProto.UINT64,
}


def _validate_entries(
    label: str,
    actual: list[onnx.ValueInfoProto],
    expected: tuple[TensorAbi, ...],
    dtype_overrides: Mapping[str, str],
) -> None:
    actual_names = tuple(item.name for item in actual)
    expected_names = tuple(item.name for item in expected)
    if actual_names != expected_names:
        raise OnnxExportError(
            f"{label} names do not match artifact ABI: "
            f"{actual_names!r} != {expected_names!r}"
        )
    for value_info, entry in zip(actual, expected, strict=True):
        actual_dtype = value_info.type.tensor_type.elem_type
        expected_dtype = dtype_overrides.get(entry.name, entry.dtype)
        if actual_dtype != _ONNX_DTYPES[expected_dtype]:
            raise OnnxExportError(
                f"{label} {entry.name!r} dtype does not match artifact ABI"
            )
        if _shape(value_info) != entry.shape:
            raise OnnxExportError(
                f"{label} {entry.name!r} shape does not match artifact ABI"
            )


def validate_io_abi(
    model: onnx.ModelProto,
    metadata: GraphMetadata,
    *,
    output_dtype_overrides: Mapping[str, str] | None = None,
) -> None:
    """Validate ordered artifact names, dtypes, and static shapes."""
    try:
        expected = derive_artifact_io_abi(metadata)
    except GraphStateError as error:
        raise OnnxExportError(f"Invalid graph artifact ABI: {error}") from error
    _validate_entries("Input", list(model.graph.input), expected.inputs, {})
    _validate_entries(
        "Output",
        list(model.graph.output),
        expected.outputs,
        output_dtype_overrides or {},
    )


__all__ = ["validate_io_abi"]
