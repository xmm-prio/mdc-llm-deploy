"""Shared ONNX graph and quantization construction primitives."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ...errors import OnnxExportError
from ...graph_types import GraphMetadata, QuantizedTarget
from ...quantization_properties import (
    ActivationQuantizationParameters,
)
from ..model_inspection import (
    optional_static_shape as static_shape,
)

FLOAT_ONNX_DTYPES: set[int] = {
    int(TensorProto.FLOAT16),
    int(TensorProto.FLOAT),
    int(TensorProto.BFLOAT16),
}
_ONNX_DTYPES = {
    "float16": TensorProto.FLOAT16,
    "float32": TensorProto.FLOAT,
    "bfloat16": TensorProto.BFLOAT16,
    "int8": TensorProto.INT8,
    "int16": TensorProto.INT16,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
    "bool": TensorProto.BOOL,
}
_NP_FLOAT32 = np.dtype(np.float32)


def initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    """Create a contiguous named ONNX initializer."""
    return numpy_helper.from_array(np.ascontiguousarray(value), name=name)


def model_types(
    model: onnx.ModelProto,
) -> dict[str, tuple[int, tuple[int, ...]]]:
    """Collect all statically known tensor dtypes and shapes."""
    result: dict[str, tuple[int, tuple[int, ...]]] = {}
    for item in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        shape = static_shape(item)
        if shape is not None:
            result[item.name] = (item.type.tensor_type.elem_type, shape)
    for item in model.graph.initializer:
        result[item.name] = (item.data_type, tuple(item.dims))
    return result


def unique_name(model: onnx.ModelProto, base: str) -> str:
    """Return a graph-wide unique value name."""
    names = {item.name for item in model.graph.initializer}
    names.update(item.name for item in model.graph.input)
    names.update(output for node in model.graph.node for output in node.output)
    if base not in names:
        return base
    index = 1
    while f"{base}.{index}" in names:
        index += 1
    return f"{base}.{index}"


def append_value(
    model: onnx.ModelProto,
    name: str,
    dtype: int,
    shape: tuple[int, ...],
) -> None:
    """Append static tensor value metadata."""
    model.graph.value_info.append(helper.make_tensor_value_info(name, dtype, shape))


def append_quant(
    model: onnx.ModelProto,
    source: str,
    source_shape: tuple[int, ...],
    target: QuantizedTarget,
    name: str,
) -> str:
    """Append an MDC INT8 activation quantization node."""
    source_dtype = model_types(model)[source][0]
    parameter_dtype: np.dtype[Any] = np.dtype(
        np.float16 if source_dtype == TensorProto.FLOAT16 else np.float32
    )
    scale = scale_initializer(
        model,
        f"{name}.scale",
        target,
        inverse=True,
        dtype=parameter_dtype,
    )
    offset = offset_initializer(
        model,
        f"{name}.offset",
        target,
        dtype=parameter_dtype,
    )
    output = unique_name(model, f"{name}.output")
    axis = -2 if target.granularity == "per_token" else -1
    model.graph.node.append(
        helper.make_node(
            "NPUAscendQuantV2",
            [source, scale, offset],
            [output],
            name=name,
            axis=axis,
            dtype=2,
        )
    )
    append_value(model, output, TensorProto.INT8, source_shape)
    return output


def scale_initializer(
    model: onnx.ModelProto,
    name: str,
    target: QuantizedTarget,
    *,
    inverse: bool,
    dtype: np.dtype[Any] = _NP_FLOAT32,
) -> str:
    """Materialize a quantization scale initializer."""
    values = np.asarray(target.scale, dtype=np.float32)
    if inverse:
        if np.dtype(dtype) == np.dtype(np.float16) and (values < (1.0 / 65504.0)).any():
            raise OnnxExportError(f"FP16 quantization scale for {target.fqn!r} is too small")
        values = 1.0 / values
    result = unique_name(model, name)
    model.graph.initializer.append(initializer(result, values.astype(dtype).squeeze()))
    return result


def offset_initializer(
    model: onnx.ModelProto,
    name: str,
    target: QuantizedTarget,
    *,
    dtype: np.dtype[Any] = _NP_FLOAT32,
) -> str:
    """Materialize a quantization zero-point initializer."""
    result = unique_name(model, name)
    values = np.asarray(target.zero_point, dtype=dtype).squeeze()
    model.graph.initializer.append(initializer(result, values))
    return result


def activation_target(
    value: GraphMetadata,
    target: QuantizedTarget,
) -> QuantizedTarget:
    """Return calibrated activation qparams associated with a weight target."""
    try:
        qparams = ActivationQuantizationParameters.for_target(
            value.properties,
            target.fqn,
        )
    except ValueError as error:
        raise OnnxExportError(str(error)) from error
    if qparams is None:
        return target
    return replace(
        target,
        bits=qparams.bits,
        granularity=qparams.granularity,
        symmetric=qparams.symmetric,
        scale=qparams.scale,
        zero_point=qparams.zero_point,
    )
