"""Quantized linear lowering into the MDC W8A8 ONNX chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper

from ...errors import OnnxExportError
from ...graph.metadata import GraphMetadata, QuantizedTarget
from ..inspection import decoded_node_attributes
from .support import (
    FLOAT_ONNX_DTYPES,
    activation_target,
    append_value,
    initializer,
    model_types,
    offset_initializer,
    scale_initializer,
)


@dataclass
class _LinearLoweringContext:
    """Maintain call-local indexes for quantized linear graph mutations."""

    model: onnx.ModelProto
    _initializers_by_name: dict[str, onnx.TensorProto]
    _linear_nodes_by_weight: dict[str, list[onnx.NodeProto]]
    _types: dict[str, tuple[int, tuple[int, ...]]]
    _names: set[str]

    @classmethod
    def from_model(cls, model: onnx.ModelProto) -> _LinearLoweringContext:
        """Build indexes with the graph's existing lookup semantics."""
        initializers_by_name: dict[str, onnx.TensorProto] = {}
        names: set[str] = set()
        for item in model.graph.initializer:
            initializers_by_name.setdefault(item.name, item)
            names.add(item.name)

        linear_nodes_by_weight: dict[str, list[onnx.NodeProto]] = {}
        for node in model.graph.node:
            weight_name = cls._linear_weight_name(node)
            if weight_name is not None:
                linear_nodes_by_weight.setdefault(weight_name, []).append(node)
            names.update(node.output)
        names.update(item.name for item in model.graph.input)
        return cls(
            model=model,
            _initializers_by_name=initializers_by_name,
            _linear_nodes_by_weight=linear_nodes_by_weight,
            _types=model_types(model),
            _names=names,
        )

    @staticmethod
    def _linear_weight_name(node: onnx.NodeProto) -> str | None:
        if node.op_type not in {"Gemm", "MatMul"} or len(node.input) < 2:
            return None
        return str(node.input[1])

    @property
    def types(self) -> dict[str, tuple[int, tuple[int, ...]]]:
        """Return the synchronized static type index."""
        return self._types

    def first_initializer(self, name: str) -> onnx.TensorProto | None:
        """Return the first initializer with the requested name."""
        return self._initializers_by_name.get(name)

    def linear_nodes(self, weight_name: str) -> list[onnx.NodeProto]:
        """Return matching standard linear nodes in graph order."""
        return self._linear_nodes_by_weight.get(weight_name, [])

    def unique_name(self, base: str) -> str:
        """Allocate and reserve the smallest available value name."""
        if base not in self._names:
            self._names.add(base)
            return base
        index = 1
        while f"{base}.{index}" in self._names:
            index += 1
        result = f"{base}.{index}"
        self._names.add(result)
        return result

    def _record_initializer(self, tensor: onnx.TensorProto) -> None:
        self._names.add(tensor.name)
        self._types[tensor.name] = (tensor.data_type, tuple(tensor.dims))
        self._initializers_by_name.setdefault(tensor.name, tensor)

    def record_appended_initializer(self) -> None:
        """Synchronize an initializer appended by a construction helper."""
        self._record_initializer(self.model.graph.initializer[-1])

    def append_initializer(self, tensor: onnx.TensorProto) -> None:
        """Append an initializer and synchronize all affected indexes."""
        self.model.graph.initializer.append(tensor)
        self._record_initializer(self.model.graph.initializer[-1])

    def append_value(
        self,
        name: str,
        dtype: int,
        shape: tuple[int, ...],
    ) -> None:
        """Append static value metadata and synchronize its type."""
        append_value(self.model, name, dtype, shape)
        self._types[name] = (dtype, shape)

    def replace_node(
        self,
        node: onnx.NodeProto,
        replacement: list[onnx.NodeProto],
    ) -> None:
        """Replace one node and synchronize node and name indexes."""
        nodes = list(self.model.graph.node)
        index = nodes.index(node)
        nodes[index : index + 1] = replacement
        del self.model.graph.node[:]
        self.model.graph.node.extend(nodes)

        weight_name = self._linear_weight_name(node)
        if weight_name is not None:
            bucket = self._linear_nodes_by_weight[weight_name]
            self._linear_nodes_by_weight[weight_name] = [
                item for item in bucket if item is not node
            ]
        inserted = self.model.graph.node[index : index + len(replacement)]
        for offset, item in enumerate(inserted):
            replacement_weight = self._linear_weight_name(item)
            if replacement_weight is not None:
                prior_count = sum(
                    self._linear_weight_name(candidate) == replacement_weight
                    for candidate in nodes[: index + offset]
                )
                self._linear_nodes_by_weight.setdefault(
                    replacement_weight,
                    [],
                ).insert(prior_count, item)
            self._names.update(item.output)


def _linear_weight_array(
    node: onnx.NodeProto,
    weight: onnx.TensorProto,
) -> NDArray[np.float32]:
    array = numpy_helper.to_array(weight).astype(np.float32)
    if node.op_type == "MatMul":
        return array
    attributes = decoded_node_attributes(node)
    alpha = float(attributes.get("alpha", 1.0))
    beta = float(attributes.get("beta", 1.0))
    trans_a = int(attributes.get("transA", 0))
    trans_b = int(attributes.get("transB", 0))
    if alpha != 1.0 or beta != 1.0 or trans_a != 0:
        raise OnnxExportError(
            f"Linear node {node.name!r} uses unsupported Gemm attributes"
        )
    return array.T if trans_b == 1 else array


def _replace_linear(
    context: _LinearLoweringContext,
    value: GraphMetadata,
    target: QuantizedTarget,
) -> None:
    model = context.model
    weight_name = f"graph.{target.fqn}.weight"
    weight = context.first_initializer(weight_name)
    if weight is None:
        raise OnnxExportError(
            f"Cannot locate ONNX weight for linear target {target.fqn!r}"
        )
    matches = context.linear_nodes(weight_name)
    if len(matches) != 1:
        raise OnnxExportError(
            f"Linear target {target.fqn!r} maps to "
            f"{len(matches)} standard ONNX nodes"
        )
    node = matches[0]
    if len(node.output) != 1 or not node.input[0]:
        raise OnnxExportError(
            f"Linear node for {target.fqn!r} has an invalid ABI"
        )
    types = context.types
    source = node.input[0]
    array = _linear_weight_array(node, weight)
    source_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    if node.output[0] not in types:
        query_length = value.sequence_length if value.stage.is_prefill else 1
        source_shape = (1, query_length, array.shape[0])
        output_shape = (1, query_length, array.shape[1])
        context.append_value(
            source,
            weight.data_type,
            source_shape,
        )
        context.append_value(
            node.output[0],
            weight.data_type,
            output_shape,
        )
    output_dtype, output_shape = types[node.output[0]]
    source_dtype = types.get(source, (output_dtype, ()))[0]
    source_shape = types.get(
        source,
        (source_dtype, (*output_shape[:-1], array.shape[0])),
    )[1]
    if source_dtype not in FLOAT_ONNX_DTYPES or output_dtype != source_dtype:
        raise OnnxExportError(
            f"Linear target {target.fqn!r} has unsupported dtypes"
        )

    activation = activation_target(value, target)
    if (
        activation.granularity != "per_tensor"
        or len(activation.scale) != 1
        or any(activation.zero_point)
    ):
        raise OnnxExportError(
            f"Linear activation for {target.fqn!r} "
            "must be symmetric per-tensor"
        )
    if not target.symmetric or any(target.zero_point):
        raise OnnxExportError(
            f"Linear weight for {target.fqn!r} must be symmetric"
        )

    weight_scales = np.asarray(target.scale, dtype=np.float32)
    if weight_scales.size not in {1, array.shape[1]}:
        raise OnnxExportError(
            f"Linear weight scale for {target.fqn!r} has invalid shape"
        )
    packed = np.clip(
        np.rint(array / weight_scales.reshape(1, -1)),
        -128,
        127,
    ).astype(np.int8)
    if output_shape != (*source_shape[:-1], packed.shape[1]):
        raise OnnxExportError(
            f"Linear target {target.fqn!r} has inconsistent output shape"
        )

    prefix = f"mdc.linear.{target.fqn}"
    parameter_dtype: np.dtype[Any] = np.dtype(
        np.float16 if source_dtype == TensorProto.FLOAT16 else np.float32
    )
    quant_scale = scale_initializer(
        model,
        f"{prefix}.quant_scale",
        activation,
        inverse=True,
        dtype=parameter_dtype,
        name_allocator=context.unique_name,
    )
    context.record_appended_initializer()
    quant_offset = offset_initializer(
        model,
        f"{prefix}.quant_offset",
        activation,
        dtype=parameter_dtype,
        name_allocator=context.unique_name,
    )
    context.record_appended_initializer()
    packed_name = context.unique_name(f"{prefix}.weight")
    context.append_initializer(initializer(packed_name, packed))
    combined = (
        weight_scales * float(activation.scale[0])
    ).astype(np.float32)
    dequant_scale = combined.view(np.uint32).astype(np.uint64)
    dequant_scale_name = context.unique_name(f"{prefix}.dequant_scale")
    context.append_initializer(
        initializer(dequant_scale_name, dequant_scale)
    )

    quantized = context.unique_name(f"{prefix}.quantized")
    accumulator = context.unique_name(f"{prefix}.accumulator")
    original_output = node.output[0]
    has_bias = (
        node.op_type == "Gemm"
        and len(node.input) == 3
        and bool(node.input[2])
    )
    dequantized = (
        context.unique_name(f"{prefix}.dequantized")
        if has_bias
        else original_output
    )
    replacement = [
        helper.make_node(
            "NPUAscendQuantV2",
            [source, quant_scale, quant_offset],
            [quantized],
            name=f"{prefix}.quant",
            axis=-1,
            dtype=2,
        ),
        helper.make_node(
            "MatMul",
            [quantized, packed_name],
            [accumulator],
            name=f"{prefix}.matmul",
        ),
        helper.make_node(
            "AscendDequant",
            [accumulator, dequant_scale_name],
            [dequantized],
            name=f"{prefix}.dequant",
            sqrt_mode=0,
            relu_flag=0,
            dtype=1 if source_dtype == TensorProto.FLOAT16 else 0,
        ),
    ]
    if has_bias:
        replacement.append(
            helper.make_node(
                "Add",
                [dequantized, node.input[2]],
                [original_output],
                name=f"{prefix}.bias",
            )
        )
    context.append_value(quantized, TensorProto.INT8, source_shape)
    context.append_value(accumulator, TensorProto.INT32, output_shape)
    if has_bias:
        context.append_value(dequantized, output_dtype, output_shape)

    context.replace_node(node, replacement)


def append_quantized_linears(
    model: onnx.ModelProto,
    value: GraphMetadata,
) -> None:
    """Replace all FQN-matched linear targets with MDC W8A8 chains."""
    targets = [
        item
        for item in value.quantized_targets
        if item.target_type == "linear"
    ]
    if targets:
        context = _LinearLoweringContext.from_model(model)
        for target in targets:
            _replace_linear(context, value, target)
    used_inputs = {
        name
        for node in model.graph.node
        for name in node.input
        if name
    }
    retained = [
        item
        for item in model.graph.initializer
        if item.name in used_inputs or not item.name.startswith("graph.")
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
