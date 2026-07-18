"""Lower RMSNorm, RoPE, and Attention subgraphs to MDC ONNX operators."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import onnx
from onnx import TensorProto, helper

from ...errors import GraphStateError, OnnxExportError
from ...graph.fx.ownership import is_fqn_descendant
from ...graph.metadata import (
    GraphMetadata,
    QuantizedTarget,
    derive_artifact_io_abi,
    order_attention_boundaries,
)
from ...graph.metadata.model import AttentionDimensions
from ...operators.contracts.attention import (
    ATTENTION_INPUT_COUNT,
    RELEASE_ATTENTION_ATTRIBUTES,
    AttentionInput,
)
from ..inspection import (
    optional_static_shape as static_shape,
)
from .cleanup import producer_map, replace_nodes
from .support import (
    OnnxLoweringContext,
    append_value,
    initializer,
    model_types,
    offset_initializer,
    scale_initializer,
    unique_name,
)

MaskMode = Literal["masked", "maskless"]

_ABI_DTYPES = {
    "bfloat16": TensorProto.BFLOAT16,
    "float16": TensorProto.FLOAT16,
    "float32": TensorProto.FLOAT,
    "int8": TensorProto.INT8,
}


def _target(
    value: GraphMetadata,
    edge: str,
    attention_fqn: str,
) -> QuantizedTarget | None:
    matches = [
        item
        for item in value.quantized_targets
        if item.target_type == "attention" and item.fqn.rsplit(".", 1)[-1] == edge
        and (
            item.fqn.startswith(f"{attention_fqn}.")
            or f".{attention_fqn}." in f".{item.fqn}."
        )
    ]
    return matches[0] if matches else None


def attention_cache_dtype_overrides(
    value: GraphMetadata,
) -> dict[str, str]:
    """Return lowered cache dtypes that intentionally differ from FX ABI."""
    result: dict[str, str] = {}
    for layer_id, boundary in enumerate(
        order_attention_boundaries(value.boundaries)
    ):
        for edge in ("key", "value"):
            if _target(value, edge, boundary.fqn) is not None:
                result[f"present.{layer_id}.{edge}"] = "int8"
    return result


def validate_lowered_attention_cache(
    model: onnx.ModelProto,
    value: GraphMetadata,
) -> dict[str, str]:
    """Validate lowered present producers, layer mapping, BNSD, and dtype."""
    try:
        artifact_abi = derive_artifact_io_abi(value)
        boundaries = order_attention_boundaries(value.boundaries)
    except GraphStateError as error:
        raise OnnxExportError(
            f"Invalid graph artifact ABI: {error}"
        ) from error
    cache_entries = value.output_abi[1:]
    expected_output_names = tuple(item.name for item in value.output_abi)
    actual_output_names = tuple(item.name for item in model.graph.output)
    if actual_output_names != expected_output_names:
        raise OnnxExportError(
            "Lowered attention outputs do not preserve internal layer order"
        )
    if len(cache_entries) != artifact_abi.layer_count * 2:
        raise OnnxExportError("Lowered attention cache layer count is invalid")

    producers_by_name: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for output_name in node.output:
            if output_name:
                producers_by_name.setdefault(output_name, []).append(node)
    outputs_by_name = {item.name: item for item in model.graph.output}
    overrides = attention_cache_dtype_overrides(value)
    try:
        dimensions = AttentionDimensions.from_properties(value.properties)
    except ValueError as error:
        raise OnnxExportError(str(error)) from error
    expected_sequence = (
        value.sequence_length
        if value.stage.is_prefill
        else (
            value.absolute_position + 1
            if value.absolute_position is not None
            else 0
        )
    )
    if expected_sequence <= 0:
        raise OnnxExportError(
            "Attention cache validation requires a positive sequence length"
        )

    for layer_id, boundary in enumerate(boundaries):
        attention_nodes = [
            node
            for node in model.graph.node
            if node.name == f"mdc.attention.{boundary.fqn}"
            and node.op_type == "FusedInferAttentionScore"
        ]
        if len(attention_nodes) != 1:
            raise OnnxExportError(
                f"Attention layer {layer_id} has invalid lowered producer mapping"
            )
        attention_node = attention_nodes[0]
        for offset, edge in enumerate(("key", "value")):
            entry = cache_entries[layer_id * 2 + offset]
            output = outputs_by_name[entry.name]
            producers = producers_by_name.get(entry.name, ())
            if len(producers) != 1:
                raise OnnxExportError(
                    f"Present cache {entry.name!r} must have one producer"
                )
            shape = static_shape(output)
            if (
                shape != entry.shape
                or len(shape) != 4
                or shape[1] != dimensions.num_key_value_heads
                or shape[2] != expected_sequence
                or shape[3] != dimensions.head_dim
            ):
                raise OnnxExportError(
                    f"Present cache {entry.name!r} must use static BNSD layout"
                )
            expected_dtype = overrides.get(entry.name, entry.dtype)
            if output.type.tensor_type.elem_type != _ABI_DTYPES[expected_dtype]:
                raise OnnxExportError(
                    f"Present cache {entry.name!r} has invalid lowered dtype"
                )
            input_index = (
                AttentionInput.KEY
                if edge == "key"
                else AttentionInput.VALUE
            )
            if attention_node.input[input_index] != entry.name:
                raise OnnxExportError(
                    f"Attention layer {layer_id} {edge} cache mapping is invalid"
                )
    return overrides


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
    matches = [node for node in consumers.get(value_name, []) if node.op_type == op_type]
    if len(matches) != 1:
        raise OnnxExportError(f"Value {value_name!r} maps to {len(matches)} {op_type} consumers")
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
        if node.op_type in op_types and len(node.input) >= 2 and node.input[1] == weight_name
    ]
    if len(matches) != 1:
        raise OnnxExportError(f"Boundary {fqn!r} maps to {len(matches)} standard ONNX nodes")
    return matches[0]


def lower_maskless_attention(model: onnx.ModelProto) -> None:
    """Remove the standard causal mask immediately before attention Softmax nodes."""
    producers = producer_map(model)
    for node in model.graph.node:
        if node.op_type != "Softmax" or not node.input:
            continue
        masked = producers.get(node.input[0])
        if masked is not None and masked.op_type == "Where" and len(masked.input) == 3:
            node.input[0] = masked.input[2]


def lower_rms_norms(model: onnx.ModelProto, value: GraphMetadata) -> None:
    """Replace every FQN-owned Tiny RMSNorm terminal with NPURmsNorm."""
    types = model_types(model)
    initializers = {
        item.name: item for item in model.graph.initializer
    }
    nodes = list(model.graph.node)
    producers = producer_map(model)
    node_indices = {id(node): index for index, node in enumerate(nodes)}
    removed: set[int] = set()
    replacements: dict[int, list[onnx.NodeProto]] = {}
    boundaries = [item for item in value.boundaries if item.kind == "rms_norm"]
    if not boundaries:
        raise OnnxExportError("MDC ONNX lowering requires an RmsNorm boundary")
    for boundary in boundaries:
        gamma = f"graph.{boundary.fqn}.weight"
        gamma_tensor = initializers.get(gamma)
        if gamma_tensor is None:
            raise OnnxExportError(
                f"RmsNorm gamma {gamma!r} must be an initializer"
            )
        matches = [node for node in nodes if node.op_type == "Mul" and gamma in node.input]
        if len(matches) != 1:
            raise OnnxExportError(
                f"RmsNorm boundary {boundary.fqn!r} maps to {len(matches)} terminal nodes"
            )
        terminal = matches[0]
        normalized_name = next(name for name in terminal.input if name != gamma)
        normalized = producers.get(normalized_name)
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
        if source is None:
            source = normalized.input[0]
        if source not in types or terminal.output[0] not in types:
            try:
                dimensions = AttentionDimensions.from_properties(
                    value.properties
                )
            except ValueError as error:
                raise OnnxExportError(str(error)) from error
            sequence = (
                value.sequence_length if value.stage.is_prefill else 1
            )
            shape: tuple[int, ...]
            if boundary.fqn.endswith(".q_norm"):
                shape = (
                    1,
                    sequence,
                    dimensions.num_attention_heads,
                    dimensions.head_dim,
                )
            elif boundary.fqn.endswith(".k_norm"):
                shape = (
                    1,
                    sequence,
                    dimensions.num_key_value_heads,
                    dimensions.head_dim,
                )
            else:
                hidden_size = value.properties.get("hidden_size")
                if type(hidden_size) is not int or hidden_size <= 0:
                    raise OnnxExportError(
                        "RmsNorm lowering requires positive hidden_size"
                    )
                shape = (1, sequence, hidden_size)
            append_value(model, source, gamma_tensor.data_type, shape)
            append_value(
                model,
                terminal.output[0],
                gamma_tensor.data_type,
                shape,
            )
            types[source] = (gamma_tensor.data_type, shape)
            types[terminal.output[0]] = (gamma_tensor.data_type, shape)
        dtype, source_shape = types[source]
        output_dtype, output_shape = types[terminal.output[0]]
        if (
            dtype
            not in {
                int(TensorProto.FLOAT16),
                int(TensorProto.FLOAT),
                int(TensorProto.BFLOAT16),
            }
            or output_dtype != dtype
            or output_shape != source_shape
        ):
            raise OnnxExportError(
                f"RmsNorm boundary {boundary.fqn!r} has an invalid tensor contract"
            )
        rstd = unique_name(model, f"mdc.rms_norm.{boundary.fqn}.rstd")
        replacement = helper.make_node(
            "NPURmsNorm",
            [source, gamma],
            [terminal.output[0], rstd],
            name=f"mdc.rms_norm.{boundary.fqn}",
            epsilon=1e-6,
        )
        index = node_indices[id(terminal)]
        removed.add(index)
        replacements[index] = [replacement]
        append_value(model, rstd, TensorProto.FLOAT, source_shape[:-1])
    replace_nodes(model, removed, replacements)


def _product_scale_initializer(
    model: onnx.ModelProto,
    name: str,
    left: QuantizedTarget,
    right: QuantizedTarget,
    *,
    name_allocator: Callable[[str], str],
) -> str:
    if (
        left.granularity != "per_tensor"
        or right.granularity != "per_tensor"
        or len(left.scale) != 1
        or len(right.scale) != 1
    ):
        raise OnnxExportError("Quantized Attention accumulator scales require per-tensor inputs")
    value = float(left.scale[0]) * float(right.scale[0])
    if not math.isfinite(value) or value <= 0:
        raise OnnxExportError("Quantized Attention accumulator scale must be finite and positive")
    result = name_allocator(name)
    model.graph.initializer.append(initializer(result, np.asarray(value, dtype=np.float32)))
    return result


def lower_rope_attention(
    model: onnx.ModelProto,
    value: GraphMetadata,
    mask_mode: MaskMode,
    *,
    layer_id: int = 0,
    context: OnnxLoweringContext | None = None,
) -> None:
    """Replace FQN-anchored Tiny RoPE and attention standard subgraphs."""
    lowering_context = (
        context
        if context is not None
        else OnnxLoweringContext.from_model(model)
    )
    if lowering_context.model is not model:
        raise ValueError("Lowering context belongs to a different ONNX model")
    try:
        dimensions = AttentionDimensions.from_properties(
            value.properties
        )
    except ValueError as error:
        raise OnnxExportError(str(error)) from error
    heads = dimensions.num_attention_heads
    kv_heads = dimensions.num_key_value_heads
    head_dim = dimensions.head_dim
    query_sequence = 1 if not value.stage.is_prefill else value.sequence_length
    attention_boundaries = [item for item in value.boundaries if item.kind == "attention"]
    rope_boundaries = [item for item in value.boundaries if item.kind == "rope"]
    if len(attention_boundaries) != 1 or len(rope_boundaries) != 1:
        raise OnnxExportError("Tiny lowering requires exactly one attention and one RoPE boundary")
    attention_fqn = attention_boundaries[0].fqn
    rope_fqn = rope_boundaries[0].fqn
    if not is_fqn_descendant(rope_fqn, attention_fqn):
        raise OnnxExportError("RoPE boundary is not owned by the attention boundary")

    types = lowering_context.types
    producers = producer_map(model)
    consumers = _consumer_map(model)
    query_projection = _single_weighted_node(model, f"{attention_fqn}.q_proj", {"Gemm", "MatMul"})
    key_projection = _single_weighted_node(model, f"{attention_fqn}.k_proj", {"Gemm", "MatMul"})
    value_projection = _single_weighted_node(model, f"{attention_fqn}.v_proj", {"Gemm", "MatMul"})
    output_projection = _single_weighted_node(model, f"{attention_fqn}.o_proj", {"Gemm", "MatMul"})
    query_reshape = _single_consumer(consumers, query_projection.output[0], "Reshape")
    key_reshape = _single_consumer(consumers, key_projection.output[0], "Reshape")
    value_reshape = _single_consumer(consumers, value_projection.output[0], "Reshape")
    query = query_reshape.output[0]
    key = key_reshape.output[0]
    query_norms = [
        node
        for node in model.graph.node
        if node.name == f"mdc.rms_norm.{attention_fqn}.q_norm"
    ]
    key_norms = [
        node
        for node in model.graph.node
        if node.name == f"mdc.rms_norm.{attention_fqn}.k_norm"
    ]
    if query_norms or key_norms:
        if len(query_norms) != 1 or len(key_norms) != 1:
            raise OnnxExportError("Q/K normalization mapping is ambiguous")
        query = query_norms[0].output[0]
        key = key_norms[0].output[0]
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
        rotated_name = next(name for name in terminal.input if name != direct_node.output[0])
        rotated = producers.get(rotated_name)
        if rotated is None or rotated.op_type != "Mul":
            raise OnnxExportError("RoPE rotation branch is incomplete")
        cos_shape = types.get(cos_name, (0, ()))[1]
        sin_candidates = [
            name for name in rotated.input if name in types and types[name][1] == cos_shape
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
        (
            item
            for item in model.graph.output
            if item.name == f"present.{layer_id}.key"
        ),
        None,
    )
    value_output = next(
        (
            item
            for item in model.graph.output
            if item.name == f"present.{layer_id}.value"
        ),
        None,
    )
    if key_output is None or value_output is None:
        raise OnnxExportError("Attention lowering requires key/value graph outputs")
    key_cache = producers.get(key_output.name)
    value_cache = producers.get(value_output.name)
    current_key = _single_consumer(consumers, key_rope, "Transpose")
    current_value = _single_consumer(consumers, value_reshape.output[0], "Transpose")

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
            f"past.{layer_id}.key",
        )
        or not matches_cache(
            value_cache,
            current_value,
            f"past.{layer_id}.value",
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
        raise OnnxExportError("Attention output projection lacks a standard attention spine")
    query_bnsd = query_transpose.output[0]
    query_bnsd_shape = (1, heads, query_sequence, head_dim)
    if query_bnsd not in types:
        lowering_context.append_value(query_bnsd, query_dtype, query_bnsd_shape)
    key_shape_full = static_shape(key_output)
    value_shape_full = static_shape(value_output)
    if key_shape_full is None or value_shape_full is None:
        raise OnnxExportError("Attention cache outputs lack static type metadata")
    key_input = key_output.name
    value_input = value_output.name
    key_target = _target(value, "key", attention_fqn)
    value_target = _target(value, "value", attention_fqn)
    query_target = _target(value, "query", attention_fqn)
    score_target = _target(value, "score", attention_fqn)
    quant_name = f"mdc.attention.{attention_fqn}"
    if query_target is not None:
        query_bnsd = lowering_context.request_quant(
            query_bnsd,
            query_target,
            axis=-2 if query_target.granularity == "per_token" else -1,
            name=f"{quant_name}.query_quant",
        )
    if key_target is not None and key_output.type.tensor_type.elem_type != TensorProto.INT8:
        key_source = lowering_context.rebind_graph_output(key_output.name)
        key_input = lowering_context.request_quant(
            key_source,
            key_target,
            axis=-2 if key_target.granularity == "per_token" else -1,
            preferred_output=key_output.name,
            name=f"{quant_name}.key_quant",
        )
        lowering_context.append_value(
            key_input,
            TensorProto.INT8,
            key_shape_full,
        )
    if value_target is not None and value_output.type.tensor_type.elem_type != TensorProto.INT8:
        value_source = lowering_context.rebind_graph_output(value_output.name)
        value_input = lowering_context.request_quant(
            value_source,
            value_target,
            axis=-2 if value_target.granularity == "per_token" else -1,
            preferred_output=value_output.name,
            name=f"{quant_name}.value_quant",
        )
        lowering_context.append_value(
            value_input,
            TensorProto.INT8,
            value_shape_full,
        )

    inputs = [""] * ATTENTION_INPUT_COUNT
    inputs[AttentionInput.QUERY] = query_bnsd
    inputs[AttentionInput.KEY] = key_input
    inputs[AttentionInput.VALUE] = value_input
    if mask_mode == "masked":
        mask = (
            np.zeros((1, 1, 1, value.sequence_length), dtype=np.bool_)
            if not value.stage.is_prefill
            else np.triu(
                np.ones((1, 1, value.sequence_length, value.sequence_length), dtype=np.bool_),
                k=1,
            )
        )
        mask_name = lowering_context.unique_name("mdc.attention.mask")
        model.graph.initializer.append(initializer(mask_name, mask))
        inputs[AttentionInput.ATTEN_MASK] = mask_name
    if query_target is not None and key_target is not None:
        inputs[AttentionInput.DEQUANT_SCALE1] = _product_scale_initializer(
            model,
            "mdc.attention.dequant_scale1",
            query_target,
            key_target,
            name_allocator=lowering_context.unique_name,
        )
    if score_target is not None:
        inputs[AttentionInput.QUANT_SCALE1] = scale_initializer(
            model,
            "mdc.attention.quant_scale1",
            score_target,
            inverse=True,
            name_allocator=lowering_context.unique_name,
        )
    if score_target is not None and value_target is not None:
        inputs[AttentionInput.DEQUANT_SCALE2] = _product_scale_initializer(
            model,
            "mdc.attention.dequant_scale2",
            score_target,
            value_target,
            name_allocator=lowering_context.unique_name,
        )
    if key_target is not None:
        inputs[AttentionInput.KEY_ANTIQUANT_SCALE] = scale_initializer(
            model,
            "mdc.attention.key_antiquant_scale",
            key_target,
            inverse=False,
            name_allocator=lowering_context.unique_name,
        )
        if not key_target.symmetric:
            inputs[AttentionInput.KEY_ANTIQUANT_OFFSET] = offset_initializer(
                model,
                "mdc.attention.key_antiquant_offset",
                key_target,
                name_allocator=lowering_context.unique_name,
            )
    if value_target is not None:
        inputs[AttentionInput.VALUE_ANTIQUANT_SCALE] = scale_initializer(
            model,
            "mdc.attention.value_antiquant_scale",
            value_target,
            inverse=False,
            name_allocator=lowering_context.unique_name,
        )
        if not value_target.symmetric:
            inputs[AttentionInput.VALUE_ANTIQUANT_OFFSET] = offset_initializer(
                model,
                "mdc.attention.value_antiquant_offset",
                value_target,
                name_allocator=lowering_context.unique_name,
            )
    if query_target is not None:
        inputs[AttentionInput.DEQUANT_SCALE_QUERY] = scale_initializer(
            model,
            "mdc.attention.dequant_scale_query",
            query_target,
            inverse=False,
            name_allocator=lowering_context.unique_name,
        )
    lse = lowering_context.unique_name("mdc.attention.lse")
    attention_output = standard_attention.output[0]
    attention_attributes: dict[str, Any] = dict(RELEASE_ATTENTION_ATTRIBUTES)
    rope_node = helper.make_node(
        "ApplyRotaryPosEmb",
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
        **attention_attributes,
    )
    nodes = list(model.graph.node)
    query_index = nodes.index(query_terminal)
    key_index = nodes.index(key_terminal)
    attention_index = nodes.index(standard_attention)
    replace_nodes(
        model,
        {query_index, key_index, attention_index},
        {
            min(query_index, key_index): [rope_node],
            attention_index: [attention_node],
        },
    )
    if attention_output not in types:
        lowering_context.append_value(
            attention_output,
            query_dtype,
            query_bnsd_shape,
        )
    lowering_context.append_value(lse, TensorProto.FLOAT, (1,))
