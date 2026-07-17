"""Call-local registry for shared ONNX activation quantizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto, helper

from ...errors import OnnxExportError
from ...graph.metadata import QuantizedTarget
from .support import FLOAT_ONNX_DTYPES, initializer

if TYPE_CHECKING:
    from .support import OnnxLoweringContext


@dataclass(frozen=True, slots=True)
class _TensorKey:
    dtype: str
    shape: tuple[int, ...]
    content: bytes


@dataclass(frozen=True, slots=True)
class QuantizationKey:
    """Identify one effective emitted activation quantization contract."""

    source: str
    scale: _TensorKey
    offset: _TensorKey
    axis: int
    output_dtype: int


def _tensor_key(value: NDArray[Any]) -> _TensorKey:
    contiguous = np.ascontiguousarray(value)
    return _TensorKey(
        dtype=contiguous.dtype.str,
        shape=tuple(contiguous.shape),
        content=contiguous.tobytes(),
    )


def _emitted_parameters(
    target: QuantizedTarget,
    dtype: np.dtype[Any],
) -> tuple[NDArray[Any], NDArray[Any]]:
    raw_scale = np.asarray(target.scale, dtype=np.float32)
    if (
        raw_scale.size == 0
        or not np.isfinite(raw_scale).all()
        or (raw_scale <= 0).any()
    ):
        raise OnnxExportError(
            f"Activation quantization scale for {target.fqn!r} must be "
            "finite and positive"
        )
    if dtype == np.dtype(np.float16) and (raw_scale < (1.0 / 65504.0)).any():
        raise OnnxExportError(
            f"FP16 quantization scale for {target.fqn!r} is too small"
        )
    raw_offset = np.asarray(target.zero_point)
    if raw_offset.size == 0 or not np.isfinite(raw_offset).all():
        raise OnnxExportError(
            f"Activation quantization offset for {target.fqn!r} must be finite"
        )
    scale = np.ascontiguousarray((1.0 / raw_scale).astype(dtype).squeeze())
    offset = np.ascontiguousarray(raw_offset.astype(dtype).squeeze())
    if scale.shape != offset.shape:
        raise OnnxExportError(
            f"Activation quantization parameters for {target.fqn!r} "
            "have mismatched emitted shapes"
        )
    return scale, offset


class QuantizationRegistry:
    """Create or reuse quantizers within one ONNX lowering call."""

    def __init__(self, context: OnnxLoweringContext) -> None:
        self._context = context
        self._outputs: dict[QuantizationKey, str] = {}

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
        """Return one quantized value for an effective emitted contract."""
        source_type = self._context.type_of(source)
        if source_type is None:
            raise OnnxExportError(f"Quantization source {source!r} has no static type")
        source_dtype, source_shape = source_type
        if source_dtype not in FLOAT_ONNX_DTYPES:
            raise OnnxExportError(
                f"Quantization source {source!r} must have a floating-point dtype"
            )
        if output_dtype != TensorProto.INT8:
            raise OnnxExportError("NPUAscendQuantV2 only supports INT8 output")
        parameter_dtype = np.dtype(
            np.float16 if source_dtype == TensorProto.FLOAT16 else np.float32
        )
        scale_value, offset_value = _emitted_parameters(target, parameter_dtype)
        key = QuantizationKey(
            source=source,
            scale=_tensor_key(scale_value),
            offset=_tensor_key(offset_value),
            axis=axis,
            output_dtype=int(output_dtype),
        )
        cached = self._outputs.get(key)
        if cached is not None:
            if preferred_output is not None and preferred_output != cached:
                raise OnnxExportError(
                    "Equivalent quantization was requested with conflicting "
                    f"preferred outputs {cached!r} and {preferred_output!r}"
                )
            return cached

        base = name or f"mdc.quant.{target.fqn}"
        node_name = self._context.unique_node_name(base)
        scale_name = self._context.unique_name(f"{base}.scale")
        offset_name = self._context.unique_name(f"{base}.offset")
        if preferred_output is None:
            output = self._context.unique_name(f"{base}.output")
        else:
            self._validate_preferred_output(
                preferred_output,
                source_shape,
                output_dtype,
            )
            output = preferred_output

        self._context.append_initializer(initializer(scale_name, scale_value))
        self._context.append_initializer(initializer(offset_name, offset_value))
        quant = helper.make_node(
            "NPUAscendQuantV2",
            [source, scale_name, offset_name],
            [output],
            name=node_name,
            axis=axis,
            dtype=2,
        )
        self._append_node(quant, source, preferred_output is not None)
        if preferred_output is None:
            self._context.append_value(output, output_dtype, source_shape)
        self._outputs[key] = output
        return output

    def _validate_preferred_output(
        self,
        output: str,
        shape: tuple[int, ...],
        dtype: int,
    ) -> None:
        expected = (dtype, shape)
        if not any(item.name == output for item in self._context.model.graph.output):
            raise OnnxExportError(
                f"Preferred quantization output {output!r} is not a graph output"
            )
        if self._context.types.get(output) != expected:
            raise OnnxExportError(
                f"Preferred quantization output {output!r} was not rebound "
                "for the requested ABI"
            )
        if any(output in node.output for node in self._context.model.graph.node):
            raise OnnxExportError(
                f"Preferred quantization output {output!r} already has a producer"
            )

    def _append_node(
        self,
        node: onnx.NodeProto,
        source: str,
        ordered_after_source: bool,
    ) -> None:
        if not ordered_after_source:
            self._context.model.graph.node.append(node)
            return
        nodes = list(self._context.model.graph.node)
        producer_index = next(
            (
                index
                for index, producer in enumerate(nodes)
                if source in producer.output
            ),
            None,
        )
        if producer_index is None:
            raise OnnxExportError(
                f"Preferred quantization source {source!r} has no producer"
            )
        nodes.insert(producer_index + 1, node)
        del self._context.model.graph.node[:]
        self._context.model.graph.node.extend(nodes)


def request_quant(
    context: OnnxLoweringContext,
    source: str,
    target: QuantizedTarget,
    *,
    axis: int,
    output_dtype: int = TensorProto.INT8,
    preferred_output: str | None = None,
    name: str | None = None,
) -> str:
    """Request a shared activation quantizer from a lowering context."""
    return context.request_quant(
        source,
        target,
        axis=axis,
        output_dtype=output_dtype,
        preferred_output=preferred_output,
        name=name,
    )


__all__ = [
    "QuantizationKey",
    "QuantizationRegistry",
    "request_quant",
]
