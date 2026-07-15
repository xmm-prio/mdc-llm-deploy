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


def _append_rms_norm(model: onnx.ModelProto) -> None:
    initializers = {item.name for item in model.graph.initializer}
    gamma_candidates = [
        (item.name, item.data_type, tuple(item.dims))
        for item in model.graph.initializer
        if len(item.dims) == 1 and item.data_type in _FLOAT_ONNX_DTYPES
    ]
    match = next(
        (
            (name, element_type, shape, gamma)
            for name, (element_type, shape) in _types(model).items()
            for gamma in gamma_candidates
            if name not in initializers
            and len(shape) == 3
            and element_type == gamma[1]
            and shape[-1] == gamma[2][-1]
        ),
        None,
    )
    if match is None:
        raise OnnxExportError("Cannot locate RmsNorm input and gamma")
    x, dtype, x_shape, (gamma, _, _) = match
    output = _unique(model, "mdc.rms_norm.output")
    rstd = _unique(model, "mdc.rms_norm.rstd")
    model.graph.node.append(
        helper.make_node(
            "NPURmsNorm",
            [x, gamma],
            [output, rstd],
            name="mdc.rms_norm",
            epsilon=1e-6,
        )
    )
    _append_value(model, output, dtype, x_shape)
    _append_value(model, rstd, TensorProto.FLOAT, x_shape[:-1])


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
    model.graph.node.append(
        helper.make_node(
            "NPUAscendQuantV2",
            [internal, scale, offset],
            [original],
            name=name,
            axis=axis,
            dtype=2,
        )
    )
    return original


def _append_rope_attention(
    model: onnx.ModelProto,
    value: GraphMetadata,
    mask_mode: MaskMode,
) -> None:
    heads = int(value.properties.get("num_attention_heads") or 4)
    kv_heads = int(value.properties.get("num_key_value_heads") or 2)
    head_dim = int(value.properties.get("head_dim") or 16)
    query_sequence = 1 if not value.stage.is_prefill else value.sequence_length
    query, query_dtype, query_shape = _find_head_tensor(
        model, heads, head_dim, query_sequence
    )
    key, key_dtype, key_shape = _find_head_tensor(
        model, kv_heads, head_dim, query_sequence
    )
    query, query_bsnd_shape = _as_bsnd(model, query, query_dtype, query_shape, heads)
    key_bsnd, key_bsnd_shape = _as_bsnd(model, key, key_dtype, key_shape, kv_heads)
    if query_dtype not in {TensorProto.FLOAT16, TensorProto.FLOAT}:
        raise OnnxExportError("MDC RoPE supports FP16/FP32 lowering inputs")
    np_dtype: np.dtype[Any] = np.dtype(np.float16 if query_dtype == TensorProto.FLOAT16 else np.float32)
    positions = (
        np.asarray([value.absolute_position], dtype=np.int64)
        if not value.stage.is_prefill
        else np.arange(value.sequence_length, dtype=np.int64)
    )
    cos, sin = _rope_tables(
        query_sequence,
        head_dim,
        float(value.properties.get("rope_theta") or 1_000_000.0),
        positions,
        np_dtype,
    )
    cos_name = _unique(model, "mdc.rope.cos")
    sin_name = _unique(model, "mdc.rope.sin")
    model.graph.initializer.extend([_initializer(cos_name, cos), _initializer(sin_name, sin)])
    query_rope = _unique(model, "mdc.rope.query")
    key_rope = _unique(model, "mdc.rope.key")
    model.graph.node.append(
        helper.make_node(
            "ApplyRoPE",
            [query, key_bsnd, cos_name, sin_name],
            [query_rope, key_rope],
            name="mdc.rope",
            layout=1,
            rotary_mode="half",
        )
    )
    _append_value(model, query_rope, query_dtype, query_bsnd_shape)
    _append_value(model, key_rope, key_dtype, key_bsnd_shape)
    query_bnsd, query_bnsd_shape = _as_bnsd(
        model, query_rope, query_dtype, query_bsnd_shape, heads
    )

    output_by_name = {item.name: item for item in model.graph.output}
    key_output = output_by_name.get("present.0.key")
    value_output = output_by_name.get("present.0.value")
    if key_output is None or value_output is None:
        raise OnnxExportError("Attention lowering requires key/value graph outputs")
    key_shape_full = _shape(key_output)
    value_shape_full = _shape(value_output)
    if key_shape_full is None or value_shape_full is None:
        raise OnnxExportError("Cache outputs must have static shapes")
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
    attention_output = _unique(model, "mdc.attention.output")
    lse = _unique(model, "mdc.attention.lse")
    model.graph.node.append(
        helper.make_node(
            "FusedInferAttentionScore",
            inputs,
            [attention_output, lse],
            name="mdc.attention",
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
    )
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


def _append_linear(model: onnx.ModelProto, value: GraphMetadata) -> None:
    targets = [item for item in value.quantized_targets if item.target_type == "linear"]
    if not targets:
        return
    initializers = {
        item.name: item
        for item in model.graph.initializer
        if len(item.dims) == 2 and item.data_type in _FLOAT_ONNX_DTYPES
    }
    target, weight = next(
        (
            (target, item)
            for target in targets
            for name, item in initializers.items()
            if (
                target.fqn in name
                or (
                    "embed" not in name
                    and len(target.scale) in {1, int(item.dims[1])}
                )
            )
        ),
        (targets[0], next(item for name, item in initializers.items() if "embed" not in name)),
    )
    weight_name = weight.name
    weight_shape = tuple(weight.dims)
    source, source_dtype, source_shape = _pick_rank(
        model,
        3,
        shape_filter=lambda shape: shape[-1] == weight_shape[0],
    )
    activation_target = _activation_target(value, target)
    if len(activation_target.scale) != 1:
        activation_target = replace(
            activation_target,
            granularity="per_tensor",
            scale=(max(activation_target.scale),),
            zero_point=(0,),
        )
    quantized = _append_quant(
        model,
        source,
        source_shape,
        activation_target,
        "mdc.linear.quant",
    )
    weight = next(item for item in model.graph.initializer if item.name == weight_name)
    array = numpy_helper.to_array(weight).astype(np.float32)
    weight_scales = np.asarray(target.scale, dtype=np.float32)
    weight_offsets = np.asarray(target.zero_point, dtype=np.float32)
    if weight_scales.size not in {1, array.shape[1]}:
        raise OnnxExportError(f"Linear weight scale for {target.fqn!r} has invalid shape")
    packed = np.clip(
        np.rint(array / weight_scales.reshape(1, -1))
        + weight_offsets.reshape(1, -1),
        -128,
        127,
    ).astype(np.int8)
    packed_name = _unique(model, "mdc.linear.weight")
    model.graph.initializer.append(_initializer(packed_name, packed))
    accumulator = _unique(model, "mdc.linear.accumulator")
    model.graph.node.append(
        helper.make_node("MatMul", [quantized, packed_name], [accumulator], name="mdc.linear.matmul")
    )
    accumulator_shape = (*source_shape[:-1], weight_shape[1])
    _append_value(model, accumulator, TensorProto.INT32, accumulator_shape)
    activation_scale = float(activation_target.scale[0])
    combined = (weight_scales * activation_scale).astype(np.float32)
    dequant_scale = combined.view(np.uint32).astype(np.uint64)
    dequant_name = _unique(model, "mdc.linear.dequant_scale")
    model.graph.initializer.append(_initializer(dequant_name, dequant_scale))
    output = _unique(model, "mdc.linear.output")
    model.graph.node.append(
        helper.make_node(
            "AscendDequant",
            [accumulator, dequant_name],
            [output],
            name="mdc.linear.dequant",
            sqrt_mode=0,
            relu_flag=0,
            dtype=1 if source_dtype == TensorProto.FLOAT16 else 0,
        )
    )
    _append_value(model, output, source_dtype, accumulator_shape)


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
    _append_rms_norm(model)
    _append_rope_attention(model, value, mask_mode)
    _append_linear(model, value)
    _append_moe(model, value)
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
