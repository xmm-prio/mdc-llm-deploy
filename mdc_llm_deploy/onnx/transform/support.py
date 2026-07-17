"""Shared ONNX graph and quantization construction primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper

from ...errors import OnnxExportError
from ...graph.metadata import GraphMetadata, QuantizedTarget
from ...graph.metadata.quantization import (
    ActivationQuantizationParameters,
)
from ..inspection import (
    optional_static_shape as static_shape,
)

if TYPE_CHECKING:
    from .quantization import QuantizationRegistry

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


@dataclass
class OnnxLoweringContext:
    """Maintain deterministic indexes for one ONNX lowering call."""

    model: onnx.ModelProto
    _names: set[str]
    _node_names: set[str]
    _types: dict[str, tuple[int, tuple[int, ...]]]
    _initializers_by_name: dict[str, onnx.TensorProto]
    _graph_output_sources: dict[str, str] = field(default_factory=dict)
    _quantization: QuantizationRegistry | None = field(default=None, init=False)

    @classmethod
    def from_model(cls, model: onnx.ModelProto) -> OnnxLoweringContext:
        """Build call-local indexes from the current graph."""
        names = {item.name for item in model.graph.initializer}
        names.update(item.name for item in model.graph.input)
        names.update(output for node in model.graph.node for output in node.output)
        return cls(
            model=model,
            _names=names,
            _node_names={node.name for node in model.graph.node if node.name},
            _types=model_types(model),
            _initializers_by_name={
                item.name: item for item in reversed(model.graph.initializer)
            },
        )

    @property
    def types(self) -> dict[str, tuple[int, tuple[int, ...]]]:
        """Return the synchronized static type index."""
        return self._types

    def type_of(
        self,
        name: str,
    ) -> tuple[int, tuple[int, ...]] | None:
        """Return a type, refreshing values added by another lowering pass."""
        result = self._types.get(name)
        if result is not None:
            return result
        result = model_types(self.model).get(name)
        if result is not None:
            self._types[name] = result
            self._names.add(name)
        return result

    @property
    def quantization(self) -> QuantizationRegistry:
        """Return the quantization registry owned by this lowering call."""
        if self._quantization is None:
            from .quantization import QuantizationRegistry

            self._quantization = QuantizationRegistry(self)
        return self._quantization

    def first_initializer(self, name: str) -> onnx.TensorProto | None:
        """Return the initializer currently indexed by name."""
        return self._initializers_by_name.get(name)

    def unique_name(self, base: str) -> str:
        """Allocate and reserve the smallest available value name."""
        result = base
        index = 1
        while result in self._names:
            result = f"{base}.{index}"
            index += 1
        self._names.add(result)
        return result

    def unique_node_name(self, base: str) -> str:
        """Allocate and reserve the smallest available node name."""
        result = base
        index = 1
        while result in self._node_names:
            result = f"{base}.{index}"
            index += 1
        self._node_names.add(result)
        return result

    def append_initializer(self, tensor: onnx.TensorProto) -> str:
        """Append an initializer and synchronize all indexes."""
        self.model.graph.initializer.append(tensor)
        stored = self.model.graph.initializer[-1]
        self._names.add(stored.name)
        self._types[stored.name] = (stored.data_type, tuple(stored.dims))
        self._initializers_by_name[stored.name] = stored
        return str(stored.name)

    def replace_initializers(
        self,
        tensors: list[onnx.TensorProto],
    ) -> None:
        """Replace graph initializers and rebuild synchronized indexes."""
        previous_names = set(self._initializers_by_name)
        del self.model.graph.initializer[:]
        self.model.graph.initializer.extend(tensors)
        self._initializers_by_name = {
            item.name: item for item in reversed(self.model.graph.initializer)
        }
        for name in previous_names - self._initializers_by_name.keys():
            self._types.pop(name, None)
        for item in self.model.graph.initializer:
            self._names.add(item.name)
            self._types[item.name] = (item.data_type, tuple(item.dims))

    def append_value(
        self,
        name: str,
        dtype: int,
        shape: tuple[int, ...],
    ) -> None:
        """Append static value metadata and synchronize its type."""
        append_value(self.model, name, dtype, shape)
        self._names.add(name)
        self._types[name] = (dtype, shape)

    def rebind_graph_output(
        self,
        output_name: str,
        *,
        output_dtype: int = TensorProto.INT8,
    ) -> str:
        """Move a graph output's producer to a stable internal value.

        Call this before requesting its quantizer. The returned internal value
        is the source passed to ``request_quant`` and ``output_name`` is the
        preferred output name.
        """
        previous = self._graph_output_sources.get(output_name)
        if previous is not None:
            return previous
        output = next(
            (item for item in self.model.graph.output if item.name == output_name),
            None,
        )
        if output is None:
            raise OnnxExportError(f"Graph output {output_name!r} does not exist")
        source_type = self._types.get(output_name)
        if source_type is None:
            raise OnnxExportError(f"Graph output {output_name!r} has no static type")
        producer = next(
            (
                node
                for node in self.model.graph.node
                if output_name in node.output
            ),
            None,
        )
        if producer is None:
            raise OnnxExportError(f"Graph output {output_name!r} has no producer")

        internal = self.unique_name(f"{output_name}.float")
        for index, name in enumerate(producer.output):
            if name == output_name:
                producer.output[index] = internal
        for node in self.model.graph.node:
            if node is producer:
                continue
            for index, name in enumerate(node.input):
                if name == output_name:
                    node.input[index] = internal
        for item in self.model.graph.value_info:
            if item.name == output_name:
                item.name = internal

        source_dtype, source_shape = source_type
        if not any(
            item.name == internal for item in self.model.graph.value_info
        ):
            append_value(self.model, internal, source_dtype, source_shape)
        self._types[internal] = source_type
        self._types[output_name] = (output_dtype, source_shape)
        output.type.tensor_type.elem_type = output_dtype
        self._graph_output_sources[output_name] = internal
        return internal

    def request_quant(
        self,
        source: str,
        target: QuantizedTarget,
        *,
        axis: int,
        output_dtype: int = TensorProto.INT8,
        preferred_output: str | None = None,
        name: str | None = None,
    ) -> str:
        """Return one shared quantized value for an effective contract."""
        return self.quantization.request_quant(
            source,
            target,
            axis=axis,
            output_dtype=output_dtype,
            preferred_output=preferred_output,
            name=name,
        )


def initializer(name: str, value: NDArray[Any]) -> onnx.TensorProto:
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
    *,
    name_allocator: Callable[[str], str] | None = None,
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
        name_allocator=name_allocator,
    )
    offset = offset_initializer(
        model,
        f"{name}.offset",
        target,
        dtype=parameter_dtype,
        name_allocator=name_allocator,
    )
    output = (
        name_allocator(f"{name}.output")
        if name_allocator is not None
        else unique_name(model, f"{name}.output")
    )
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
    name_allocator: Callable[[str], str] | None = None,
) -> str:
    """Materialize a quantization scale initializer."""
    values = np.asarray(target.scale, dtype=np.float32)
    if inverse:
        if np.dtype(dtype) == np.dtype(np.float16) and (values < (1.0 / 65504.0)).any():
            raise OnnxExportError(f"FP16 quantization scale for {target.fqn!r} is too small")
        values = 1.0 / values
    result = (
        name_allocator(name)
        if name_allocator is not None
        else unique_name(model, name)
    )
    model.graph.initializer.append(initializer(result, values.astype(dtype).squeeze()))
    return result


def offset_initializer(
    model: onnx.ModelProto,
    name: str,
    target: QuantizedTarget,
    *,
    dtype: np.dtype[Any] = _NP_FLOAT32,
    name_allocator: Callable[[str], str] | None = None,
) -> str:
    """Materialize a quantization zero-point initializer."""
    result = (
        name_allocator(name)
        if name_allocator is not None
        else unique_name(model, name)
    )
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
