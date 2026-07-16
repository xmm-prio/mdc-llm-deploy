"""Transactional conversion from prefill FX graphs to static decode graphs."""

from __future__ import annotations

from torch.fx import GraphModule
from torch.fx.graph import CodeGen

from ..errors import GraphStateError, UnsupportedPatternError
from ..fx_inspection import flatten_nodes, node_target
from ..graph import (
    metadata,
    set_metadata,
    transactional_update,
)
from .decode_cache import (
    cache_target as _cache_target,
)
from .decode_cache import (
    insert_cache_quantization as _insert_cache_quantization,
)
from .decode_cache import (
    replace_attention_cache_users as _replace_attention_cache_users,
)
from .decode_metadata import build_decode_metadata
from .decode_rewrite import (
    remove_prefill_causal_mask as _remove_prefill_causal_mask,
)
from .decode_rewrite import (
    replace_static_sequence as _replace_static_sequence,
)
from .decode_rewrite import (
    rewrite_position_nodes as _rewrite_position_nodes,
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
        if len(outputs) < 3:
            raise UnsupportedPatternError(
                "Decode conversion requires logits, key, and value outputs"
            )
        current_key, current_value = outputs[1:3]
        if not any(
            "repeat_interleave" in node_target(user)
            for user in current_key.users
        ):
            raise UnsupportedPatternError(
                "Decode conversion cannot locate key attention use"
            )
        if not any(
            "repeat_interleave" in node_target(user)
            for user in current_value.users
        ):
            raise UnsupportedPatternError(
                "Decode conversion cannot locate value attention use"
            )
        cache_use = next(
            node
            for node in candidate.graph.nodes
            if node in current_key.users or node in current_value.users
            if "repeat_interleave" in node_target(node)
        )

        first_compute = next(
            node
            for node in candidate.graph.nodes
            if node.op not in {"placeholder", "get_attr"}
        )
        with candidate.graph.inserting_before(first_compute):
            past_key = candidate.graph.placeholder("past_key_values_0_key")
            past_value = candidate.graph.placeholder("past_key_values_0_value")
        with candidate.graph.inserting_before(cache_use):
            present_key, attention_key = _insert_cache_quantization(
                candidate,
                current_key,
                past_key,
                _cache_target(value, "key"),
                "key",
                value.sequence_length,
            )
            present_value, attention_value = _insert_cache_quantization(
                candidate,
                current_value,
                past_value,
                _cache_target(value, "value"),
                "value",
                value.sequence_length,
            )
        _replace_attention_cache_users(current_key, attention_key, output)
        _replace_attention_cache_users(current_value, attention_value, output)
        output_values = outputs.copy()
        output_values[1] = present_key
        output_values[2] = present_value
        output.args = (tuple(output_values),)
        candidate.graph.set_codegen(CodeGen())  # type: ignore[no-untyped-call]

        _rewrite_position_nodes(candidate, value.sequence_length)
        _remove_prefill_causal_mask(candidate)
        for node in candidate.graph.nodes:
            if node.op in {"placeholder", "get_attr", "output"}:
                continue
            node.args = _replace_static_sequence(node.args, value.sequence_length)
            node.kwargs = _replace_static_sequence(
                dict(node.kwargs),
                value.sequence_length,
            )
        candidate.graph.eliminate_dead_code()
        set_metadata(
            candidate,
            build_decode_metadata(candidate, value),
        )

    updated = transactional_update(graph, mutate)
    updated.recompile()
    return updated
