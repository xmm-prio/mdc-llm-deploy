"""Transactional conversion from prefill FX graphs to static decode graphs."""

from __future__ import annotations

from torch.fx import GraphModule
from torch.fx.graph import CodeGen

from ...errors import GraphStateError, UnsupportedPatternError
from ...graph.fx.inspection import flatten_nodes, node_target
from ...graph.lifecycle import (
    metadata,
    set_metadata,
    transactional_update,
)
from .cache import (
    cache_target as _cache_target,
)
from .cache import (
    insert_cache_quantization as _insert_cache_quantization,
)
from .cache import (
    replace_attention_cache_users as _replace_attention_cache_users,
)
from .metadata import build_decode_metadata
from .rewrite import (
    remove_prefill_causal_mask as _remove_prefill_causal_mask,
)
from .rewrite import (
    rewrite_position_nodes as _rewrite_position_nodes,
)
from .rewrite import (
    rewrite_rotary_cache as _rewrite_rotary_cache,
)
from .rewrite import (
    rewrite_static_shapes as _rewrite_static_shapes,
)


def convert_to_decode(graph: GraphModule) -> GraphModule:
    """Atomically rewrite a static prefill ATen graph into one-token decode."""
    current = metadata(graph)
    if not current.stage.is_prefill:
        raise GraphStateError("convert_to_decode requires a prefill graph")
    if current.sequence_length < 2:
        raise UnsupportedPatternError("Decode conversion requires sequence length >= 2")
    if not any(boundary.kind == "attention" for boundary in current.boundaries):
        raise UnsupportedPatternError(
            "Decode conversion requires an attention boundary"
        )

    def mutate(candidate: GraphModule) -> None:
        value = metadata(candidate)
        output = next(node for node in candidate.graph.nodes if node.op == "output")
        outputs = list(flatten_nodes(output.args[0]))
        if len(outputs) < 3 or (len(outputs) - 1) % 2:
            raise UnsupportedPatternError(
                "Decode conversion requires logits and per-layer key/value outputs"
            )
        first_compute = next(
            node
            for node in candidate.graph.nodes
            if node.op not in {"placeholder", "get_attr"}
        )
        output_values = outputs.copy()
        layer_count = (len(outputs) - 1) // 2
        for layer_id in range(layer_count):
            current_key = outputs[1 + layer_id * 2]
            current_value = outputs[2 + layer_id * 2]
            key_uses = [
                user
                for user in current_key.users
                if "repeat_interleave" in node_target(user)
            ]
            value_uses = [
                user
                for user in current_value.users
                if "repeat_interleave" in node_target(user)
            ]
            if not key_uses or not value_uses:
                raise UnsupportedPatternError(
                    f"Decode conversion cannot locate layer {layer_id} cache use"
                )
            with candidate.graph.inserting_before(first_compute):
                past_key = candidate.graph.placeholder(
                    f"past_{layer_id}_key"
                )
                past_value = candidate.graph.placeholder(
                    f"past_{layer_id}_value"
                )
            cache_use = key_uses[0]
            with candidate.graph.inserting_before(cache_use):
                present_key, attention_key = _insert_cache_quantization(
                    candidate,
                    current_key,
                    past_key,
                    _cache_target(value, "key", layer_id),
                    f"{layer_id}_key",
                    value.sequence_length,
                )
                present_value, attention_value = _insert_cache_quantization(
                    candidate,
                    current_value,
                    past_value,
                    _cache_target(value, "value", layer_id),
                    f"{layer_id}_value",
                    value.sequence_length,
                )
            _replace_attention_cache_users(
                current_key, attention_key, output
            )
            _replace_attention_cache_users(
                current_value, attention_value, output
            )
            output_values[1 + layer_id * 2] = present_key
            output_values[2 + layer_id * 2] = present_value
        output.args = (tuple(output_values),)
        candidate.graph.set_codegen(CodeGen())  # type: ignore[no-untyped-call]

        _rewrite_position_nodes(candidate, value.sequence_length)
        _rewrite_rotary_cache(candidate, value.sequence_length)
        attention_nodes = frozenset(
            node
            for boundary in value.boundaries
            if boundary.kind == "attention"
            for node in boundary.nodes
        )
        _remove_prefill_causal_mask(
            candidate,
            attention_nodes=attention_nodes,
            sequence_length=value.sequence_length,
        )
        _rewrite_static_shapes(candidate, value.sequence_length)
        candidate.graph.eliminate_dead_code()
        set_metadata(
            candidate,
            build_decode_metadata(candidate, value),
        )

    return transactional_update(graph, mutate)
