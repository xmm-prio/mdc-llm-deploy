"""Derive decode ABI and metadata after FX graph rewriting."""

from __future__ import annotations

from dataclasses import replace

from torch.fx import GraphModule

from ..errors import GraphStateError
from ..fx_inspection import node_target
from ..graph_types import GraphMetadata, GraphStage, TensorAbi
from ..model_properties import AttentionDimensions
from .decode_cache import cache_target


def build_decode_metadata(
    candidate: GraphModule,
    value: GraphMetadata,
) -> GraphMetadata:
    """Build metadata matching a rewritten one-token decode graph."""
    live_nodes = {node.name for node in candidate.graph.nodes}
    boundaries = tuple(
        replace(
            boundary,
            nodes=tuple(
                node for node in boundary.nodes if node in live_nodes
            ),
        )
        for boundary in value.boundaries
        if any(node in live_nodes for node in boundary.nodes)
    )

    try:
        dimensions = AttentionDimensions.from_properties(
            value.properties
        )
    except ValueError as error:
        raise GraphStateError(str(error)) from error
    original_inputs = tuple(
        replace(
            item,
            shape=tuple(
                1 if dimension == value.sequence_length else dimension
                for dimension in item.shape
            ),
        )
        for item in value.input_abi
    )
    cache_shape = (
        1,
        dimensions.num_key_value_heads,
        value.sequence_length - 1,
        dimensions.head_dim,
    )
    cache_outputs = value.output_abi[1:]
    if len(cache_outputs) % 2:
        raise GraphStateError("Cache outputs must contain key/value pairs")
    cache_inputs: list[TensorAbi] = []
    for layer_id in range(len(cache_outputs) // 2):
        key_target = cache_target(value, "key", layer_id)
        value_target = cache_target(value, "value", layer_id)
        cache_inputs.extend(
            (
                TensorAbi(
                    f"past.{layer_id}.key",
                    "int8"
                    if key_target is not None
                    else cache_outputs[layer_id * 2].dtype,
                    cache_shape,
                ),
                TensorAbi(
                    f"past.{layer_id}.value",
                    "int8"
                    if value_target is not None
                    else cache_outputs[layer_id * 2 + 1].dtype,
                    cache_shape,
                ),
            )
        )
    updated_inputs = (*original_inputs, *cache_inputs)
    updated_outputs = tuple(
        replace(item, shape=(1, 1, item.shape[-1]))
        if item.name == "logits"
        else replace(
            item,
            dtype=(
                "int8"
                if cache_target(
                    value,
                    "key" if item.name.endswith(".key") else "value",
                    int(item.name.split(".")[1]),
                )
                is not None
                else item.dtype
            ),
        )
        for item in value.output_abi
    )
    properties = dict(value.properties)
    properties.update(
        {
            "decode_rewrite": True,
            "cache_layout": "BNSD",
            "cache_length": value.sequence_length - 1,
            "query_length": 1,
            "position_ids": (value.sequence_length - 1,),
            "mask_semantics": (
                "all cached and current tokens visible"
            ),
            "aten_node_count": sum(
                node.op == "call_function"
                and "aten::" in node_target(node)
                for node in candidate.graph.nodes
            ),
        }
    )
    next_stage = (
        GraphStage.QUANTIZED_DECODE
        if value.stage.is_quantized
        else GraphStage.FLOAT_DECODE
    )
    return replace(
        value,
        stage=next_stage,
        input_abi=updated_inputs,
        output_abi=updated_outputs,
        boundaries=boundaries,
        absolute_position=value.sequence_length - 1,
        properties=properties,
    )
