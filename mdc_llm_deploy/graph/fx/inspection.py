"""Shared, business-agnostic FX graph inspection helpers."""

from __future__ import annotations

from typing import Any

from torch.fx import Node
from torch.fx.node import map_arg


def flatten_nodes(value: Any) -> tuple[Node, ...]:
    """Collect FX nodes from a nested argument structure."""
    result: list[Node] = []

    def collect(item: Any) -> Any:
        if isinstance(item, Node):
            result.append(item)
        return item

    map_arg(value, collect)
    return tuple(result)


def node_target(node: Node) -> str:
    """Return a stable textual target for an FX node."""
    target = node.target
    if hasattr(target, "_schema"):
        return str(target._schema.name)
    return str(target)


def linear_weight_name(node: Node) -> str | None:
    """Return the get_attr weight name for an ATen linear node."""
    if (
        node.op != "call_function"
        or node_target(node) != "aten::linear"
        or len(node.args) < 2
    ):
        return None
    weight = node.args[1]
    if not isinstance(weight, Node) or weight.op != "get_attr":
        return None
    return str(weight.target)
