"""Static shape, position, and mask rewrites for decode graphs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.fx import GraphModule, Node

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
        kwargs: dict[str, Any] = {"dtype": torch.int64}
        if isinstance(tensor, Tensor):
            kwargs["device"] = tensor.device
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
