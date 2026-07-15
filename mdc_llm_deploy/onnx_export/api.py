"""Atomic lowering from an ATen FX graph to the MDC ONNX dialect."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper, shape_inference
from torch.fx import GraphModule, Node

from ..errors import OnnxExportError, UnsupportedPatternError
from ..graph import (
    GraphMetadata,
    QuantizedTarget,
    metadata,
    validate_capability_request,
)
from .validator import validate_mdc_model, validate_serialized_model

MaskMode = Literal["masked", "maskless"]

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
_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
_FLOAT_ONNX_DTYPES: set[int] = {
    int(TensorProto.FLOAT16),
    int(TensorProto.FLOAT),
    int(TensorProto.BFLOAT16),
}
_NP_FLOAT32 = np.dtype(np.float32)
_ATTENTION_INPUT_COUNT = 29


class _PositionalGraph(torch.nn.Module):
    """Adapt torch.export keyword-only call specs to positional ONNX tracing."""

    def __init__(
        self,
        graph: GraphModule,
        names: tuple[str, ...],
        *,
        use_kwargs: bool,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.names = names
        self.use_kwargs = use_kwargs
        self.training = False

    def forward(self, *args: torch.Tensor) -> Any:
        """Call the captured graph with its original input convention."""
        if self.use_kwargs:
            return self.graph(**dict(zip(self.names, args, strict=True)))
        return self.graph(*args)


def _dtype(name: str) -> int:
    try:
        return _ONNX_DTYPES[name]
    except KeyError as error:
        raise OnnxExportError(f"Unsupported ABI dtype: {name}") from error


def _value(name: str, dtype: int | str, shape: tuple[int, ...]) -> onnx.ValueInfoProto:
    element_type = _dtype(dtype) if isinstance(dtype, str) else dtype
    return helper.make_tensor_value_info(name, element_type, shape)


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(value), name=name)


def _target(value: GraphMetadata, edge: str, target_type: str = "attention") -> QuantizedTarget | None:
    matches = [
        item
        for item in value.quantized_targets
        if item.target_type == target_type and item.fqn.rsplit(".", 1)[-1] == edge
    ]
    return matches[0] if matches else None


def _validate_export_request(graph: GraphModule, mask_mode: str) -> GraphMetadata:
    value = metadata(graph)
    if mask_mode not in {"masked", "maskless"}:
        raise ValueError("mask_mode must be 'masked' or 'maskless'")
    if value.properties.get("gptq") or any(
        target.algorithm == "gptq" or target.bits == 4
        for target in value.quantized_targets
    ):
        raise OnnxExportError("GPTQ and W4 graphs do not support ONNX export")
    for target in value.quantized_targets:
        edge = target.fqn.rsplit(".", 1)[-1]
        if target.target_type == "attention" and edge in {"query", "score"} and not target.symmetric:
            raise OnnxExportError("Asymmetric attention query/score is unsupported")
        if (
            not value.stage.is_prefill
            and target.target_type == "attention"
            and edge in {"key", "value"}
            and target.bits == 4
        ):
            raise OnnxExportError("INT4 decode cache is unsupported")
    epsilon = value.properties.get("rms_norm_epsilon")
    if epsilon is not None and epsilon != 1e-6:
        raise OnnxExportError("MDC ONNX requires RmsNorm epsilon=1e-6")
    if not any(item.kind == "attention" for item in value.boundaries):
        raise OnnxExportError("MDC ONNX lowering requires an attention boundary")
    try:
        validate_capability_request(value, mask_mode=mask_mode, artifact="onnx")
    except UnsupportedPatternError as error:
        raise OnnxExportError(str(error)) from error
    return value


def _device(graph: GraphModule) -> torch.device:
    for tensor in (*tuple(graph.parameters()), *tuple(graph.buffers())):
        return tensor.device
    return torch.device("cpu")


def _example_arguments(graph: GraphModule, value: GraphMetadata) -> tuple[torch.Tensor, ...]:
    device = _device(graph)
    result: list[torch.Tensor] = []
    for item in value.input_abi:
        try:
            dtype = _TORCH_DTYPES[item.dtype]
        except KeyError as error:
            raise OnnxExportError(f"Unsupported input dtype: {item.dtype}") from error
        result.append(torch.zeros(item.shape, dtype=dtype, device=device))
    return tuple(result)


def _standard_onnx(
    graph: GraphModule,
    value: GraphMetadata,
    directory: Path,
) -> onnx.ModelProto:
    descriptor, name = tempfile.mkstemp(
        prefix=".mdc-standard.",
        suffix=".onnx",
        dir=directory,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.onnx.export(
            _PositionalGraph(
                graph,
                tuple(item.name for item in value.input_abi),
                use_kwargs=value.stage.is_prefill,
            ),
            _example_arguments(graph, value),
            temporary,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=[item.name for item in value.input_abi],
            output_names=[item.name for item in value.output_abi],
            training=torch.onnx.TrainingMode.PRESERVE,
            dynamo=False,
        )
        standard = onnx.load(temporary, load_external_data=False)
        _restore_linear_initializer_names(standard, graph)
        onnx.checker.check_model(standard, full_check=True)
        standard = shape_inference.infer_shapes(standard, strict_mode=True, data_prop=True)
        onnx.checker.check_model(standard, full_check=True)
        return standard
    except OnnxExportError:
        raise
    except Exception as error:
        raise OnnxExportError(f"Standard ONNX validation failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _restore_linear_initializer_names(
    model: onnx.ModelProto,
    graph: GraphModule,
) -> None:
    """Restore FX parameter FQNs lost by the legacy ONNX exporter."""
    parameter_names = [
        str(weight.target)
        for node in graph.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.linear.default
        and len(node.args) >= 2
        and isinstance((weight := node.args[1]), Node)
        and weight.op == "get_attr"
    ]
    initializers = {item.name: item for item in model.graph.initializer}
    onnx_weight_names = [
        node.input[1]
        for node in model.graph.node
        if node.op_type in {"Gemm", "MatMul"}
        and len(node.input) >= 2
        and node.input[1] in initializers
        and len(initializers[node.input[1]].dims) == 2
        and initializers[node.input[1]].data_type in _FLOAT_ONNX_DTYPES
        and "embed" not in node.input[1]
    ]
    if len(parameter_names) != len(onnx_weight_names):
        raise OnnxExportError(
            "Cannot map ATen linear parameters to standard ONNX initializers"
        )
    for parameter_name, old_name in zip(parameter_names, onnx_weight_names, strict=True):
        new_name = f"graph.{parameter_name}"
        initializers[old_name].name = new_name
        for node in model.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name == old_name:
                    node.input[index] = new_name


def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        return None
    dimensions: list[int] = []
    for item in tensor.shape.dim:
        if not item.HasField("dim_value") or item.dim_value <= 0:
            return None
        dimensions.append(item.dim_value)
    return tuple(dimensions)


def _types(model: onnx.ModelProto) -> dict[str, tuple[int, tuple[int, ...]]]:
    result: dict[str, tuple[int, tuple[int, ...]]] = {}
    for item in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        shape = _shape(item)
        if shape is not None:
            result[item.name] = (item.type.tensor_type.elem_type, shape)
    for item in model.graph.initializer:
        result[item.name] = (item.data_type, tuple(item.dims))
    return result


def _unique(model: onnx.ModelProto, base: str) -> str:
    names = {item.name for item in model.graph.initializer}
    names.update(item.name for item in model.graph.input)
    names.update(output for node in model.graph.node for output in node.output)
    if base not in names:
        return base
    index = 1
    while f"{base}.{index}" in names:
        index += 1
    return f"{base}.{index}"


def _append_value(
    model: onnx.ModelProto,
    name: str,
    dtype: int,
    shape: tuple[int, ...],
) -> None:
    model.graph.value_info.append(_value(name, dtype, shape))


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output}


def _consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    result: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                result.setdefault(input_name, []).append(node)
    return result


def _single_consumer(
    consumers: dict[str, list[onnx.NodeProto]],
    value_name: str,
    op_type: str,
) -> onnx.NodeProto:
    matches = [
        node for node in consumers.get(value_name, []) if node.op_type == op_type
    ]
    if len(matches) != 1:
        raise OnnxExportError(
            f"Value {value_name!r} maps to {len(matches)} {op_type} consumers"
        )
    return matches[0]


def _single_weighted_node(
    model: onnx.ModelProto,
    fqn: str,
    op_types: set[str],
) -> onnx.NodeProto:
    weight_name = f"graph.{fqn}.weight"
    matches: list[onnx.NodeProto] = [
        node
        for node in model.graph.node
        if node.op_type in op_types
        and len(node.input) >= 2
        and node.input[1] == weight_name
    ]
    if len(matches) != 1:
        raise OnnxExportError(
            f"Boundary {fqn!r} maps to {len(matches)} standard ONNX nodes"
        )
    return matches[0]


def _replace_nodes(
    model: onnx.ModelProto,
    removed: set[int],
    replacement_by_index: dict[int, list[onnx.NodeProto]],
) -> None:
    nodes = list(model.graph.node)
    result: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        result.extend(replacement_by_index.get(index, ()))
        if index not in removed:
            result.append(node)
    del model.graph.node[:]
    model.graph.node.extend(result)


def _prune_unreachable(model: onnx.ModelProto) -> None:
    """Remove standard lowering remnants that cannot affect graph outputs."""
    producers = _producer_map(model)
    required_values = [item.name for item in model.graph.output]
    required_values.extend(
        output
        for node in model.graph.node
        if node.op_type == "MoeExpert"
        for output in node.output
    )
    required_outputs: set[str] = set()
    while required_values:
        value_name = required_values.pop()
        producer = producers.get(value_name)
        if producer is None or any(
            output in required_outputs for output in producer.output
        ):
            continue
        required_outputs.update(producer.output)
        required_values.extend(name for name in producer.input if name)
    retained = [
        node
        for node in model.graph.node
        if any(output in required_outputs for output in node.output)
    ]
    del model.graph.node[:]
    model.graph.node.extend(retained)
    used_initializers = {
        name for node in model.graph.node for name in node.input if name
    }
    retained_initializers = [
        item for item in model.graph.initializer if item.name in used_initializers
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)


def _topologically_sort(model: onnx.ModelProto) -> None:
    """Restore topological order after inserting fused MDC nodes."""
    known = {item.name for item in model.graph.input}
    known.update(item.name for item in model.graph.initializer)
    pending = list(model.graph.node)
    ordered: list[onnx.NodeProto] = []
    while pending:
        ready = next(
            (
                node
                for node in pending
                if all(not name or name in known for name in node.input)
            ),
            None,
        )
        if ready is None:
            blocked = {
                node.name or node.op_type: [
                    name for name in node.input if name and name not in known
                ]
                for node in pending
            }
            raise OnnxExportError(
                f"Lowered ONNX graph cannot be topologically sorted: {blocked}"
            )
        pending.remove(ready)
        ordered.append(ready)
        known.update(ready.output)
    del model.graph.node[:]
    model.graph.node.extend(ordered)


def _remove_dynamic_value_info(model: onnx.ModelProto) -> None:
    static_values = [
        item
        for item in model.graph.value_info
        if _shape(item) is not None
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(static_values)


def _make_maskless_non_causal(model: onnx.ModelProto) -> None:
    producers = _producer_map(model)
    for node in model.graph.node:
        if node.op_type != "Softmax" or not node.input:
            continue
        masked = producers.get(node.input[0])
        if masked is not None and masked.op_type == "Where" and len(masked.input) == 3:
            node.input[0] = masked.input[2]


def _pick_rank(
    model: onnx.ModelProto,
    rank: int,
    *,
    dtype: set[int] = _FLOAT_ONNX_DTYPES,
    shape_filter: Any = None,
) -> tuple[str, int, tuple[int, ...]]:
    initializers = {item.name for item in model.graph.initializer}
    for name, (element_type, shape) in _types(model).items():
        if (
            name not in initializers
            and len(shape) == rank
            and element_type in dtype
            and (shape_filter is None or shape_filter(shape))
        ):
            return name, element_type, shape
    raise OnnxExportError(f"Cannot locate rank-{rank} tensor for MDC lowering")


def _pick_initializer(
    model: onnx.ModelProto,
    rank: int,
    *,
    dtype: set[int] = _FLOAT_ONNX_DTYPES,
) -> tuple[str, int, tuple[int, ...]]:
    for item in model.graph.initializer:
        if len(item.dims) == rank and item.data_type in dtype:
            return item.name, item.data_type, tuple(item.dims)
    raise OnnxExportError(f"Cannot locate rank-{rank} initializer for MDC lowering")


def _as_bsnd(
    model: onnx.ModelProto,
    name: str,
    dtype: int,
    shape: tuple[int, ...],
    heads: int,
) -> tuple[str, tuple[int, ...]]:
    if shape[2] == heads:
        return name, shape
    if shape[1] != heads:
        raise OnnxExportError(f"Tensor {name!r} does not expose {heads} heads")
    output = _unique(model, f"mdc.bsnd.{heads}")
    model.graph.node.append(
        helper.make_node("Transpose", [name], [output], name=f"{output}.transpose", perm=[0, 2, 1, 3])
    )
    result_shape = (shape[0], shape[2], shape[1], shape[3])
    _append_value(model, output, dtype, result_shape)
    return output, result_shape


def _as_bnsd(
    model: onnx.ModelProto,
    name: str,
    dtype: int,
    shape: tuple[int, ...],
    heads: int,
) -> tuple[str, tuple[int, ...]]:
    if shape[1] == heads:
        return name, shape
    if shape[2] != heads:
        raise OnnxExportError(f"Tensor {name!r} does not expose {heads} heads")
    output = _unique(model, f"mdc.bnsd.{heads}")
    model.graph.node.append(
        helper.make_node("Transpose", [name], [output], name=f"{output}.transpose", perm=[0, 2, 1, 3])
    )
    result_shape = (shape[0], shape[2], shape[1], shape[3])
    _append_value(model, output, dtype, result_shape)
    return output, result_shape


def _replace_rms_norms(model: onnx.ModelProto, value: GraphMetadata) -> None:
    """Replace every FQN-owned Tiny RMSNorm terminal with NPURmsNorm."""
    types = _types(model)
    nodes = list(model.graph.node)
    removed: set[int] = set()
    replacements: dict[int, list[onnx.NodeProto]] = {}
    boundaries = [item for item in value.boundaries if item.kind == "rms_norm"]
    if not boundaries:
        raise OnnxExportError("MDC ONNX lowering requires an RmsNorm boundary")
    for boundary in boundaries:
        gamma = f"graph.{boundary.fqn}.weight"
        matches = [
            node
            for node in nodes
            if node.op_type == "Mul" and gamma in node.input
        ]
        if len(matches) != 1:
            raise OnnxExportError(
                f"RmsNorm boundary {boundary.fqn!r} maps to {len(matches)} terminal nodes"
            )
        terminal = matches[0]
        normalized_name = next(name for name in terminal.input if name != gamma)
        normalized = _producer_map(model).get(normalized_name)
        if normalized is None or normalized.op_type != "Mul" or len(normalized.input) != 2:
            raise OnnxExportError(
                f"RmsNorm boundary {boundary.fqn!r} lacks a standard normalization spine"
            )
        source = next(
            (
                name
                for name in normalized.input
                if types.get(name, (0, ()))[1] == types.get(terminal.output[0], (0, ()))[1]
            ),
            None,
        )
        if source is None or source not in types or terminal.output[0] not in types:
            raise OnnxExportError(
                f"RmsNorm boundary {boundary.fqn!r} lacks static type metadata"
            )
        dtype, source_shape = types[source]
        output_dtype, output_shape = types[terminal.output[0]]
        if (
            dtype not in _FLOAT_ONNX_DTYPES
            or output_dtype != dtype
            or output_shape != source_shape
        ):
            raise OnnxExportError(
                f"RmsNorm boundary {boundary.fqn!r} has an invalid tensor contract"
            )
        rstd = _unique(model, f"mdc.rms_norm.{boundary.fqn}.rstd")
        replacement = helper.make_node(
            "NPURmsNorm",
            [source, gamma],
            [terminal.output[0], rstd],
            name=f"mdc.rms_norm.{boundary.fqn}",
            epsilon=1e-6,
        )
        index = nodes.index(terminal)
        removed.add(index)
        replacements[index] = [replacement]
        _append_value(model, rstd, TensorProto.FLOAT, source_shape[:-1])
    _replace_nodes(model, removed, replacements)


def _rope_tables(
    sequence: int,
    head_dim: int,
    theta: float,
    positions: np.ndarray,
    dtype: np.dtype[Any],
) -> tuple[np.ndarray, np.ndarray]:
    inverse = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    frequencies = positions.astype(np.float32)[:, None] * inverse[None, :]
    embedding = np.concatenate((frequencies, frequencies), axis=-1)
    shape = (1, sequence, 1, head_dim)
    return np.cos(embedding).reshape(shape).astype(dtype), np.sin(embedding).reshape(shape).astype(dtype)


def _find_head_tensor(
    model: onnx.ModelProto,
    heads: int,
    head_dim: int,
    query_sequence: int | None = None,
) -> tuple[str, int, tuple[int, ...]]:
    def matches(shape: tuple[int, ...]) -> bool:
        if shape[-1] != head_dim or heads not in shape[1:3]:
            return False
        if query_sequence is None:
            return True
        sequence_axis = 2 if shape[1] == heads else 1
        return shape[sequence_axis] == query_sequence

    output_names = {item.name for item in model.graph.output}
    initializers = {item.name for item in model.graph.initializer}
    candidates = [
        (name, element_type, shape)
        for name, (element_type, shape) in _types(model).items()
        if name not in initializers
        and len(shape) == 4
        and element_type in _FLOAT_ONNX_DTYPES
        and matches(shape)
    ]
    if not candidates:
        raise OnnxExportError(f"Cannot locate {heads}-head Attention tensor")
    return next(
        (item for item in candidates if item[0] not in output_names),
        candidates[0],
    )


def _scale_initializer(
    model: onnx.ModelProto,
    name: str,
    target: QuantizedTarget,
    *,
    inverse: bool,
    dtype: np.dtype[Any] = _NP_FLOAT32,
) -> str:
    values = np.asarray(target.scale, dtype=np.float32)
    if inverse:
        if np.dtype(dtype) == np.dtype(np.float16) and (values < (1.0 / 65504.0)).any():
            raise OnnxExportError(
                f"FP16 quantization scale for {target.fqn!r} is too small"
            )
        values = 1.0 / values
    result = _unique(model, name)
    model.graph.initializer.append(_initializer(result, values.astype(dtype).squeeze()))
    return result


def _offset_initializer(
    model: onnx.ModelProto,
    name: str,
    target: QuantizedTarget,
    *,
    dtype: np.dtype[Any] = _NP_FLOAT32,
) -> str:
    result = _unique(model, name)
    values = np.asarray(target.zero_point, dtype=dtype).squeeze()
    model.graph.initializer.append(_initializer(result, values))
    return result


def _append_quant(
    model: onnx.ModelProto,
    source: str,
    source_shape: tuple[int, ...],
    target: QuantizedTarget,
    name: str,
) -> str:
    source_dtype = _types(model)[source][0]
    parameter_dtype: np.dtype[Any] = np.dtype(
        np.float16 if source_dtype == TensorProto.FLOAT16 else np.float32
    )
    scale = _scale_initializer(
        model,
        f"{name}.scale",
        target,
        inverse=True,
        dtype=parameter_dtype,
    )
    offset = _offset_initializer(
        model,
        f"{name}.offset",
        target,
        dtype=parameter_dtype,
    )
    output = _unique(model, f"{name}.output")
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
    _append_value(model, output, TensorProto.INT8, source_shape)
    return output


def _quantize_graph_output(
    model: onnx.ModelProto,
    output: onnx.ValueInfoProto,
    shape: tuple[int, ...],
    target: QuantizedTarget,
    name: str,
) -> str:
    original = output.name
    source_dtype = output.type.tensor_type.elem_type
    internal = _unique(model, f"{original}.float")
    producer = next(
        (node for node in model.graph.node if original in node.output),
        None,
    )
    if producer is None:
        raise OnnxExportError(f"Cache output {original!r} has no producer")
    for index, produced in enumerate(producer.output):
        if produced == original:
            producer.output[index] = internal
    for node in model.graph.node:
        if node is producer:
            continue
        for index, consumed in enumerate(node.input):
            if consumed == original:
                node.input[index] = internal
    for item in model.graph.value_info:
        if item.name == original:
            item.name = internal
    if not any(item.name == internal for item in model.graph.value_info):
        _append_value(model, internal, source_dtype, shape)
    output.type.tensor_type.elem_type = TensorProto.INT8
    parameter_dtype: np.dtype[Any] = np.dtype(
        np.float16 if source_dtype == TensorProto.FLOAT16 else np.float32
    )
    scale = _scale_initializer(
        model,
        f"{name}.scale",
        target,
        inverse=True,
        dtype=parameter_dtype,
    )
    offset = _offset_initializer(
        model,
        f"{name}.offset",
        target,
        dtype=parameter_dtype,
    )
    axis = -2 if target.granularity == "per_token" else -1
    quant = helper.make_node(
        "NPUAscendQuantV2",
        [internal, scale, offset],
        [original],
        name=name,
        axis=axis,
        dtype=2,
    )
    nodes = list(model.graph.node)
    producer_index = nodes.index(producer)
    nodes.insert(producer_index + 1, quant)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return original


def _append_rope_attention(
    model: onnx.ModelProto,
    value: GraphMetadata,
    mask_mode: MaskMode,
) -> None:
    """Replace FQN-anchored Tiny RoPE and attention standard subgraphs."""
    heads = int(value.properties.get("num_attention_heads") or 4)
    kv_heads = int(value.properties.get("num_key_value_heads") or 2)
    head_dim = int(value.properties.get("head_dim") or 16)
    query_sequence = 1 if not value.stage.is_prefill else value.sequence_length
    attention_boundaries = [
        item for item in value.boundaries if item.kind == "attention"
    ]
    rope_boundaries = [item for item in value.boundaries if item.kind == "rope"]
    if len(attention_boundaries) != 1 or len(rope_boundaries) != 1:
        raise OnnxExportError(
            "Tiny lowering requires exactly one attention and one RoPE boundary"
        )
    attention_fqn = attention_boundaries[0].fqn
    rope_fqn = rope_boundaries[0].fqn
    if not rope_fqn.startswith(f"{attention_fqn}."):
        raise OnnxExportError("RoPE boundary is not owned by the attention boundary")

    types = _types(model)
    producers = _producer_map(model)
    consumers = _consumer_map(model)
    query_projection = _single_weighted_node(
        model, f"{attention_fqn}.q_proj", {"Gemm", "MatMul"}
    )
    key_projection = _single_weighted_node(
        model, f"{attention_fqn}.k_proj", {"Gemm", "MatMul"}
    )
    value_projection = _single_weighted_node(
        model, f"{attention_fqn}.v_proj", {"Gemm", "MatMul"}
    )
    output_projection = _single_weighted_node(
        model, f"{attention_fqn}.o_proj", {"Gemm", "MatMul"}
    )
    query_reshape = _single_consumer(
        consumers, query_projection.output[0], "Reshape"
    )
    key_reshape = _single_consumer(consumers, key_projection.output[0], "Reshape")
    value_reshape = _single_consumer(
        consumers, value_projection.output[0], "Reshape"
    )
    query = query_reshape.output[0]
    key = key_reshape.output[0]
    if query not in types or key not in types:
        raise OnnxExportError("RoPE projection boundaries lack static type metadata")
    query_dtype, query_shape = types[query]
    key_dtype, key_shape = types[key]
    query_bsnd_shape = (1, query_sequence, heads, head_dim)
    key_bsnd_shape = (1, query_sequence, kv_heads, head_dim)
    if query_shape != query_bsnd_shape or key_shape != key_bsnd_shape:
        raise OnnxExportError("RoPE projection boundaries have invalid BSND shapes")
    if query_dtype not in {TensorProto.FLOAT16, TensorProto.FLOAT}:
        raise OnnxExportError("MDC RoPE supports FP16/FP32 lowering inputs")
    if key_dtype != query_dtype:
        raise OnnxExportError("RoPE query and key dtypes must match")

    def rope_terminal(source: str) -> tuple[onnx.NodeProto, str, str]:
        direct = [
            node
            for node in consumers.get(source, [])
            if node.op_type == "Mul" and len(node.output) == 1
        ]
        matches: list[tuple[onnx.NodeProto, onnx.NodeProto]] = []
        for direct_node in direct:
            for terminal in consumers.get(direct_node.output[0], []):
                if terminal.op_type == "Add" and len(terminal.input) == 2:
                    matches.append((terminal, direct_node))
        if len(matches) != 1:
            raise OnnxExportError(
                f"RoPE source {source!r} maps to {len(matches)} standard terminals"
            )
        terminal, direct_node = matches[0]
        cos_name = next(name for name in direct_node.input if name != source)
        rotated_name = next(
            name for name in terminal.input if name != direct_node.output[0]
        )
        rotated = producers.get(rotated_name)
        if rotated is None or rotated.op_type != "Mul":
            raise OnnxExportError("RoPE rotation branch is incomplete")
        cos_shape = types.get(cos_name, (0, ()))[1]
        sin_candidates = [
            name
            for name in rotated.input
            if name in types and types[name][1] == cos_shape
        ]
        if len(sin_candidates) != 1:
            raise OnnxExportError("RoPE sine table mapping is ambiguous")
        return terminal, cos_name, sin_candidates[0]

    query_terminal, cos_name, sin_name = rope_terminal(query)
    key_terminal, key_cos_name, key_sin_name = rope_terminal(key)
    if (key_cos_name, key_sin_name) != (cos_name, sin_name):
        raise OnnxExportError("RoPE query and key do not share position tables")
    query_rope = query_terminal.output[0]
    key_rope = key_terminal.output[0]
    query_transpose = _single_consumer(consumers, query_rope, "Transpose")
    key_output = next(
        (item for item in model.graph.output if item.name == "present.0.key"),
        None,
    )
    value_output = next(
        (item for item in model.graph.output if item.name == "present.0.value"),
        None,
    )
    if key_output is None or value_output is None:
        raise OnnxExportError("Attention lowering requires key/value graph outputs")
    key_cache = producers.get(key_output.name)
    value_cache = producers.get(value_output.name)
    current_key = _single_consumer(consumers, key_rope, "Transpose")
    current_value = _single_consumer(
        consumers, value_reshape.output[0], "Transpose"
    )

    def matches_cache(
        cache: onnx.NodeProto | None,
        current: onnx.NodeProto,
        past_name: str,
    ) -> bool:
        if cache is not None and tuple(cache.output) == tuple(current.output):
            return True
        return (
            cache is not None
            and cache.op_type == "Concat"
            and current.output[0] in cache.input
            and past_name in cache.input
        )

    if value.stage.is_prefill and (
        not matches_cache(
            key_cache,
            current_key,
            "past_key_values.0.key",
        )
        or not matches_cache(
            value_cache,
            current_value,
            "past_key_values.0.value",
        )
    ):
        raise OnnxExportError("Attention cache outputs do not match FQN projections")

    if len(output_projection.input) < 1:
        raise OnnxExportError("Attention output projection has an invalid ABI")
    output_reshape = producers.get(output_projection.input[0])
    output_transpose = (
        producers.get(output_reshape.input[0])
        if output_reshape is not None and output_reshape.op_type == "Reshape"
        else None
    )
    standard_attention = (
        producers.get(output_transpose.input[0])
        if output_transpose is not None and output_transpose.op_type == "Transpose"
        else None
    )
    if standard_attention is None or standard_attention.op_type != "MatMul":
        raise OnnxExportError(
            "Attention output projection lacks a standard attention spine"
        )
    query_bnsd = query_transpose.output[0]
    query_bnsd_shape = (1, heads, query_sequence, head_dim)
    if query_bnsd not in types:
        _append_value(model, query_bnsd, query_dtype, query_bnsd_shape)
    key_shape_full = _shape(key_output)
    value_shape_full = _shape(value_output)
    if key_shape_full is None or value_shape_full is None:
        raise OnnxExportError("Attention cache outputs lack static type metadata")
    key_input = key_output.name
    value_input = value_output.name
    key_target = _target(value, "key")
    value_target = _target(value, "value")
    query_target = _target(value, "query")
    score_target = _target(value, "score")
    if query_target is not None:
        query_bnsd = _append_quant(
            model, query_bnsd, query_bnsd_shape, query_target, "mdc.attention.query_quant"
        )
    if key_target is not None and key_output.type.tensor_type.elem_type != TensorProto.INT8:
        key_input = _quantize_graph_output(
            model, key_output, key_shape_full, key_target, "mdc.attention.key_quant"
        )
    if value_target is not None and value_output.type.tensor_type.elem_type != TensorProto.INT8:
        value_input = _quantize_graph_output(
            model,
            value_output,
            value_shape_full,
            value_target,
            "mdc.attention.value_quant",
        )

    inputs = [""] * _ATTENTION_INPUT_COUNT
    inputs[0:3] = [query_bnsd, key_input, value_input]
    if mask_mode == "masked":
        mask = (
            np.zeros((1, 1, 1, value.sequence_length), dtype=np.bool_)
            if not value.stage.is_prefill
            else np.triu(
                np.ones((1, 1, value.sequence_length, value.sequence_length), dtype=np.bool_),
                k=1,
            )
        )
        mask_name = _unique(model, "mdc.attention.mask")
        model.graph.initializer.append(_initializer(mask_name, mask))
        inputs[4] = mask_name
    if score_target is not None:
        inputs[8] = _scale_initializer(
            model, "mdc.attention.quant_scale1", score_target, inverse=True
        )
    if key_target is not None:
        inputs[17] = _scale_initializer(
            model, "mdc.attention.key_antiquant_scale", key_target, inverse=False
        )
        if not key_target.symmetric:
            inputs[18] = _offset_initializer(
                model, "mdc.attention.key_antiquant_offset", key_target
            )
    if value_target is not None:
        inputs[19] = _scale_initializer(
            model, "mdc.attention.value_antiquant_scale", value_target, inverse=False
        )
        if not value_target.symmetric:
            inputs[20] = _offset_initializer(
                model, "mdc.attention.value_antiquant_offset", value_target
            )
    if query_target is not None:
        inputs[27] = _scale_initializer(
            model, "mdc.attention.dequant_scale_query", query_target, inverse=False
        )
    lse = _unique(model, "mdc.attention.lse")
    attention_output = standard_attention.output[0]
    rope_node = helper.make_node(
        "ApplyRoPE",
        [query, key, cos_name, sin_name],
        [query_rope, key_rope],
        name=f"mdc.rope.{rope_fqn}",
        layout=1,
        rotary_mode="half",
    )
    attention_node = helper.make_node(
        "FusedInferAttentionScore",
        inputs,
        [attention_output, lse],
        name=f"mdc.attention.{attention_fqn}",
        num_heads=heads,
        num_key_value_heads=kv_heads,
        scale=float(1.0 / math.sqrt(head_dim)),
        input_layout="BNSD",
        sparse_mode=0,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        inner_precise=0,
        block_size=0,
        antiquant_mode=0,
        softmax_lse_flag=0,
        key_antiquant_mode=0,
        value_antiquant_mode=0,
        query_quant_mode=0,
    )
    nodes = list(model.graph.node)
    query_index = nodes.index(query_terminal)
    key_index = nodes.index(key_terminal)
    attention_index = nodes.index(standard_attention)
    _replace_nodes(
        model,
        {query_index, key_index, attention_index},
        {
            min(query_index, key_index): [rope_node],
            attention_index: [attention_node],
        },
    )
    if attention_output not in types:
        _append_value(model, attention_output, query_dtype, query_bnsd_shape)
    _append_value(model, lse, TensorProto.FLOAT, (1,))


def _activation_target(
    value: GraphMetadata,
    target: QuantizedTarget,
) -> QuantizedTarget:
    """Return calibrated activation qparams associated with a weight target."""
    all_qparams = value.properties.get("activation_qparams")
    if not isinstance(all_qparams, dict):
        return target
    qparams = all_qparams.get(target.fqn)
    if not isinstance(qparams, dict):
        return target
    scale = qparams.get("scale")
    zero_point = qparams.get("zero_point")
    if not isinstance(scale, list) or not isinstance(zero_point, list):
        return target
    return replace(
        target,
        bits=int(qparams["bits"]),
        granularity=str(qparams["granularity"]),
        symmetric=bool(qparams["symmetric"]),
        scale=tuple(float(item) for item in scale),
        zero_point=tuple(int(item) for item in zero_point),
    )


def _linear_weight_array(node: onnx.NodeProto, weight: onnx.TensorProto) -> np.ndarray:
    """Return a standard linear weight in MatMul [input, output] layout."""
    array = numpy_helper.to_array(weight).astype(np.float32)
    if node.op_type == "MatMul":
        return array
    attributes = _attributes(node)
    alpha = float(attributes.get("alpha", 1.0))
    beta = float(attributes.get("beta", 1.0))
    trans_a = int(attributes.get("transA", 0))
    trans_b = int(attributes.get("transB", 0))
    if alpha != 1.0 or beta != 1.0 or trans_a != 0:
        raise OnnxExportError(f"Linear node {node.name!r} uses unsupported Gemm attributes")
    return array.T if trans_b == 1 else array


def _attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        item.name: helper.get_attribute_value(item)
        for item in node.attribute
    }


def _replace_linear(
    model: onnx.ModelProto,
    value: GraphMetadata,
    target: QuantizedTarget,
) -> None:
    """Replace one FQN-matched standard linear node with the MDC W8A8 chain."""
    weight_name = f"graph.{target.fqn}.weight"
    weight = next(
        (item for item in model.graph.initializer if item.name == weight_name),
        None,
    )
    if weight is None:
        raise OnnxExportError(f"Cannot locate ONNX weight for linear target {target.fqn!r}")
    matches = [
        node
        for node in model.graph.node
        if node.op_type in {"Gemm", "MatMul"}
        and len(node.input) >= 2
        and node.input[1] == weight_name
    ]
    if len(matches) != 1:
        raise OnnxExportError(
            f"Linear target {target.fqn!r} maps to {len(matches)} standard ONNX nodes"
        )
    node = matches[0]
    if len(node.output) != 1 or not node.input[0]:
        raise OnnxExportError(f"Linear node for {target.fqn!r} has an invalid ABI")
    types = _types(model)
    source = node.input[0]
    if node.output[0] not in types:
        raise OnnxExportError(f"Linear target {target.fqn!r} lacks static ONNX type metadata")
    output_dtype, output_shape = types[node.output[0]]
    array = _linear_weight_array(node, weight)
    source_dtype = types.get(source, (output_dtype, ()))[0]
    source_shape = types.get(
        source,
        (source_dtype, (*output_shape[:-1], array.shape[0])),
    )[1]
    if source_dtype not in _FLOAT_ONNX_DTYPES or output_dtype != source_dtype:
        raise OnnxExportError(f"Linear target {target.fqn!r} has unsupported dtypes")

    activation = _activation_target(value, target)
    if (
        activation.granularity != "per_tensor"
        or len(activation.scale) != 1
        or any(activation.zero_point)
    ):
        raise OnnxExportError(
            f"Linear activation for {target.fqn!r} must be symmetric per-tensor"
        )
    if not target.symmetric or any(target.zero_point):
        raise OnnxExportError(f"Linear weight for {target.fqn!r} must be symmetric")

    weight_scales = np.asarray(target.scale, dtype=np.float32)
    if weight_scales.size not in {1, array.shape[1]}:
        raise OnnxExportError(f"Linear weight scale for {target.fqn!r} has invalid shape")
    packed = np.clip(
        np.rint(array / weight_scales.reshape(1, -1)),
        -128,
        127,
    ).astype(np.int8)
    if output_shape != (*source_shape[:-1], packed.shape[1]):
        raise OnnxExportError(f"Linear target {target.fqn!r} has inconsistent output shape")

    prefix = f"mdc.linear.{target.fqn}"
    parameter_dtype: np.dtype[Any] = np.dtype(
        np.float16 if source_dtype == TensorProto.FLOAT16 else np.float32
    )
    quant_scale = _scale_initializer(
        model,
        f"{prefix}.quant_scale",
        activation,
        inverse=True,
        dtype=parameter_dtype,
    )
    quant_offset = _offset_initializer(
        model,
        f"{prefix}.quant_offset",
        activation,
        dtype=parameter_dtype,
    )
    packed_name = _unique(model, f"{prefix}.weight")
    model.graph.initializer.append(_initializer(packed_name, packed))
    combined = (weight_scales * float(activation.scale[0])).astype(np.float32)
    dequant_scale = combined.view(np.uint32).astype(np.uint64)
    dequant_scale_name = _unique(model, f"{prefix}.dequant_scale")
    model.graph.initializer.append(_initializer(dequant_scale_name, dequant_scale))

    quantized = _unique(model, f"{prefix}.quantized")
    accumulator = _unique(model, f"{prefix}.accumulator")
    original_output = node.output[0]
    has_bias = node.op_type == "Gemm" and len(node.input) == 3 and bool(node.input[2])
    dequantized = (
        _unique(model, f"{prefix}.dequantized")
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
    _append_value(model, quantized, TensorProto.INT8, source_shape)
    _append_value(model, accumulator, TensorProto.INT32, output_shape)
    if has_bias:
        _append_value(model, dequantized, output_dtype, output_shape)

    nodes = list(model.graph.node)
    index = nodes.index(node)
    nodes[index : index + 1] = replacement
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def _append_linear(model: onnx.ModelProto, value: GraphMetadata) -> None:
    targets = [item for item in value.quantized_targets if item.target_type == "linear"]
    for target in targets:
        _replace_linear(model, value, target)
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


def _append_moe(model: onnx.ModelProto, value: GraphMetadata) -> None:
    targets = [item for item in value.quantized_targets if item.target_type == "moe"]
    if value.model_kind != "moe" or not targets:
        return
    hidden_size = int(value.properties.get("hidden_size") or 64)
    source, _, source_shape = _pick_rank(
        model,
        3,
        shape_filter=lambda shape: shape[-1] == hidden_size,
    )
    token_count = int(np.prod(source_shape[:-1]))
    activation = _activation_target(value, targets[0])
    quantized = _append_quant(model, source, source_shape, activation, "mdc.moe.quant")
    flattened = _unique(model, "mdc.moe.input")
    reshape_name = _unique(model, "mdc.moe.input_shape")
    model.graph.initializer.append(
        _initializer(reshape_name, np.asarray([token_count, hidden_size], dtype=np.int64))
    )
    model.graph.node.append(
        helper.make_node("Reshape", [quantized, reshape_name], [flattened], name="mdc.moe.reshape")
    )
    _append_value(model, flattened, TensorProto.INT8, (token_count, hidden_size))
    topk = next((node for node in model.graph.node if node.op_type == "TopK"), None)
    if topk is None or len(topk.output) != 2:
        raise OnnxExportError("Cannot locate MoE router TopK outputs")
    routed_values, routed_ids = topk.output
    routed_ids_i16 = _unique(model, "mdc.moe.routed_ids")
    routed_values_fp32 = _unique(model, "mdc.moe.routed_weights_fp32")
    model.graph.node.extend(
        [
            helper.make_node(
                "Cast",
                [routed_ids],
                [routed_ids_i16],
                name="mdc.moe.ids_cast",
                to=TensorProto.INT16,
            ),
            helper.make_node(
                "Cast",
                [routed_values],
                [routed_values_fp32],
                name="mdc.moe.weights_cast",
                to=TensorProto.FLOAT,
            ),
        ]
    )
    routed_shape_name = _unique(model, "mdc.moe.routed_shape")
    axes_name = _unique(model, "mdc.moe.reduce_axes")
    shared_ids_name = _unique(model, "mdc.moe.shared_ids")
    shared_weights_name = _unique(model, "mdc.moe.shared_weights")
    model.graph.initializer.extend(
        [
            _initializer(routed_shape_name, np.asarray([token_count, 2], dtype=np.int64)),
            _initializer(axes_name, np.asarray([-1], dtype=np.int64)),
            _initializer(
                shared_ids_name,
                np.full((token_count, 1), 4, dtype=np.int16),
            ),
            _initializer(
                shared_weights_name,
                np.ones((token_count, 1), dtype=np.float16),
            ),
        ]
    )
    ids_2d = _unique(model, "mdc.moe.routed_ids_2d")
    weights_2d = _unique(model, "mdc.moe.routed_weights_2d")
    weight_sum = _unique(model, "mdc.moe.routed_weight_sum")
    normalized = _unique(model, "mdc.moe.normalized_weights")
    normalized_fp16 = _unique(model, "mdc.moe.normalized_weights_fp16")
    ids_name = _unique(model, "mdc.moe.topk_ids")
    weights_name = _unique(model, "mdc.moe.topk_weight")
    model.graph.node.extend(
        [
            helper.make_node("Reshape", [routed_ids_i16, routed_shape_name], [ids_2d]),
            helper.make_node("Reshape", [routed_values_fp32, routed_shape_name], [weights_2d]),
            helper.make_node("ReduceSum", [weights_2d, axes_name], [weight_sum], keepdims=1),
            helper.make_node("Div", [weights_2d, weight_sum], [normalized]),
            helper.make_node(
                "Cast",
                [normalized],
                [normalized_fp16],
                to=TensorProto.FLOAT16,
            ),
            helper.make_node(
                "Concat",
                [ids_2d, shared_ids_name],
                [ids_name],
                axis=1,
            ),
            helper.make_node(
                "Concat",
                [normalized_fp16, shared_weights_name],
                [weights_name],
                axis=1,
            ),
        ]
    )
    _append_value(model, ids_name, TensorProto.INT16, (token_count, 3))
    _append_value(model, weights_name, TensorProto.FLOAT16, (token_count, 3))

    initializers = {
        item.name: item
        for item in model.graph.initializer
        if len(item.dims) == 2 and item.data_type in _FLOAT_ONNX_DTYPES
    }
    target_by_fqn = {item.fqn: item for item in targets}
    packed_parts: list[np.ndarray] = []
    scales = np.ones(21, dtype=np.float32)
    offsets = np.zeros(21, dtype=np.int32)
    scales[0] = float(activation.scale[0])
    offsets[0] = int(activation.zero_point[0])
    for expert_id, expert_name in enumerate(
        ("experts.0", "experts.1", "experts.2", "experts.3", "shared_expert")
    ):
        base = 1 + expert_id * 4
        for projection_index, projection in enumerate(
            ("gate_proj", "up_proj", "down_proj")
        ):
            initializer = next(
                (
                    item
                    for name, item in initializers.items()
                    if expert_name in name and projection in name
                ),
                None,
            )
            if initializer is None:
                raise OnnxExportError(
                    f"Cannot locate MoE weight {expert_name}.{projection}"
                )
            target = next(
                (
                    item
                    for fqn, item in target_by_fqn.items()
                    if expert_name in fqn and projection in fqn
                ),
                activation,
            )
            scale_index = base + projection_index if projection_index < 2 else base + 3
            scale = float(target.scale[0])
            zero_point = int(target.zero_point[0])
            array = numpy_helper.to_array(initializer).astype(np.float32).T
            packed_parts.append(
                np.clip(np.rint(array / scale) + zero_point, -128, 127)
                .astype(np.int8)
                .reshape(-1)
            )
            scales[scale_index] = scale
            offsets[scale_index] = zero_point
        scales[base + 2] = float(activation.scale[0])
        offsets[base + 2] = int(activation.zero_point[0])
    packed = np.concatenate(packed_parts)
    packed_name = _unique(model, "mdc.moe.expert_weights")
    scales_name = _unique(model, "mdc.moe.quant_scales")
    offsets_name = _unique(model, "mdc.moe.quant_offsets")
    model.graph.initializer.extend(
        [
            _initializer(packed_name, packed),
            _initializer(scales_name, scales),
            _initializer(offsets_name, offsets),
        ]
    )
    output = _unique(model, "mdc.moe.output")
    model.graph.node.append(
        helper.make_node(
            "MoeExpert",
            [flattened, ids_name, weights_name, packed_name, scales_name, offsets_name],
            [output],
            name="mdc.moe",
        )
    )
    _append_value(model, output, TensorProto.FLOAT16, (token_count, hidden_size))


def _lower(
    standard: onnx.ModelProto,
    value: GraphMetadata,
    mask_mode: MaskMode,
) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(standard)
    model.producer_name = "mdc_llm_deploy"
    model.producer_version = "0.1.0"
    del model.opset_import[:]
    model.opset_import.append(helper.make_opsetid("", 18))
    if mask_mode == "maskless":
        _make_maskless_non_causal(model)
    _replace_rms_norms(model, value)
    _append_rope_attention(model, value, mask_mode)
    _append_linear(model, value)
    _append_moe(model, value)
    _prune_unreachable(model)
    _topologically_sort(model)
    algorithms = sorted({item.algorithm for item in value.quantized_targets}) or ["fp16"]
    targets = sorted({item.target_type for item in value.quantized_targets}) or ["fp16"]
    properties = {
        "mdc.graph_schema_version": str(value.schema_version),
        "mdc.stage": value.stage.value,
        "mdc.mask_mode": mask_mode,
        "mdc.mask_semantics": (
            "explicit-causal" if mask_mode == "masked" else "all-visible-non-causal"
        ),
        "mdc.model_kind": value.model_kind,
        "mdc.algorithm": ",".join(algorithms),
        "mdc.target": ",".join(targets),
        "mdc.config_fingerprint": value.config_fingerprint or "",
        "mdc.dialect": "MDC ONNX",
        "mdc.numeric_spine": "validated-standard-aten",
        "mdc.lowering_source": "fx-boundaries-and-graph-metadata",
    }
    linear_target_count = sum(
        item.target_type == "linear" for item in value.quantized_targets
    )
    if linear_target_count:
        properties["mdc.linear.target_count"] = str(linear_target_count)
    if "moe" in targets:
        hidden_size = int(value.properties.get("hidden_size") or 64)
        intermediate_size = int(value.properties.get("moe_intermediate_size") or 64)
        segment = hidden_size * intermediate_size
        properties["mdc.moe.expert_order"] = "0,1,2,3,4(shared)"
        properties["mdc.moe.weight_projection_order"] = "gate_proj,up_proj,down_proj"
        properties["mdc.moe.weight_offsets"] = ",".join(
            str(index * segment) for index in range(15)
        )
        properties["mdc.moe.quant_parameter_count"] = "21"
    _remove_dynamic_value_info(model)
    helper.set_model_props(
        model,
        properties,
    )
    return model


def onnx_export(
    graph: GraphModule,
    output_path: str | Path,
    *,
    mask_mode: MaskMode,
    overwrite: bool = False,
) -> onnx.ModelProto:
    """Lower an FX graph and atomically replace the requested ONNX file."""
    value = _validate_export_request(graph, mask_mode)
    target = Path(output_path)
    if target.suffix.lower() != ".onnx":
        raise OnnxExportError("output_path must use .onnx suffix")
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    standard = _standard_onnx(graph, value, target.parent)
    model = _lower(standard, value, mask_mode)
    validate_mdc_model(model)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=".onnx.tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        onnx.save_model(model, temporary)
        validated = validate_serialized_model(str(temporary))
        os.replace(temporary, target)
        return validated
    except OnnxExportError:
        raise
    except Exception as error:
        raise OnnxExportError(f"ONNX export failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
