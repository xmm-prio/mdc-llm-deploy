"""Static shape, position, and mask rewrites for decode graphs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.fx import GraphModule, Node

from ..errors import UnsupportedPatternError
from ..fx_inspection import node_target


def replace_static_sequence(value: Any, sequence: int) -> Any:
    """Replace captured prefill sequence dimensions with one token."""
    if type(value) is int and value == sequence:
        return 1
    if isinstance(value, tuple):
        return tuple(
            replace_static_sequence(item, sequence) for item in value
        )
    if isinstance(value, list):
        return [
            replace_static_sequence(item, sequence) for item in value
        ]
    if isinstance(value, dict):
        return {
            key: replace_static_sequence(item, sequence)
            for key, item in value.items()
        }
    return value


def rewrite_static_shapes(
    candidate: GraphModule,
    sequence: int,
) -> None:
    """Rewrite sequence axes without changing equal-valued head dimensions."""
    for node in candidate.graph.nodes:
        if (
            node.op != "call_function"
            or not any(
                operation in node_target(node)
                for operation in ("aten::view", "aten::reshape")
            )
            or len(node.args) < 2
            or not isinstance(node.args[1], (list, tuple))
        ):
            continue
        shape = list(node.args[1])
        if len(shape) >= 3 and shape[1] == sequence:
            shape[1] = 1
            node.args = (
                node.args[0],
                type(node.args[1])(shape),
                *node.args[2:],
            )


def rewrite_rotary_cache(
    candidate: GraphModule,
    sequence: int,
) -> None:
    """Narrow precomputed rotary caches to the decode position."""
    for name in ("cos_cache", "sin_cache"):
        value = getattr(candidate, name, None)
        if (
            not isinstance(value, Tensor)
            or value.ndim < 2
            or value.shape[1] != sequence
        ):
            continue
        candidate.register_buffer(
            name,
            value[:, sequence - 1 : sequence].detach().clone(),
        )


def rewrite_position_nodes(
    candidate: GraphModule,
    sequence: int,
) -> None:
    """Replace prefill ranges with the final absolute position."""
    for node in tuple(candidate.graph.nodes):
        if (
            node.op != "call_function"
            or "aten::arange" not in node_target(node)
        ):
            continue
        tensor = node.meta.get("val")
        if not isinstance(tensor, Tensor):
            raise UnsupportedPatternError(
                f"Decode position node {node.name!r} has no tensor device metadata"
            )
        kwargs: dict[str, Any] = {
            "dtype": torch.int64,
            "device": tensor.device,
        }
        with candidate.graph.inserting_before(node):
            position = candidate.graph.call_function(
                torch.ops.aten.full.default,
                args=([1], sequence - 1),
                kwargs=kwargs,
            )
        node.replace_all_uses_with(position)
        candidate.graph.erase_node(node)


def remove_prefill_causal_mask(candidate: GraphModule) -> None:
    """Remove prefill-only causal masking from one-token decode."""
    for node in tuple(candidate.graph.nodes):
        if (
            node.op != "call_function"
            or "masked_fill" not in node_target(node)
        ):
            continue
        if node.args and isinstance(node.args[0], Node):
            node.replace_all_uses_with(node.args[0])
            candidate.graph.erase_node(node)
