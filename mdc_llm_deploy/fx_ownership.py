"""Shared FX node ownership helpers."""

from __future__ import annotations

from collections.abc import Mapping

from torch.fx import Node

__all__ = ["node_belongs_to", "node_owner_fqns"]


def node_owner_fqns(node: Node) -> tuple[str, ...]:
    """Return ordered unique module FQNs recorded for an FX node."""
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, Mapping):
        return ()

    owners: list[str] = []
    seen: set[str] = set()
    for value in stack.values():
        if not isinstance(value, tuple) or not value:
            continue
        owner = value[0]
        if not isinstance(owner, str) or not owner or owner in seen:
            continue
        owners.append(owner)
        seen.add(owner)
    return tuple(owners)


def node_belongs_to(node: Node, owner_fqn: str) -> bool:
    """Return whether an FX node belongs to an exact module or descendant."""
    if not isinstance(owner_fqn, str) or not owner_fqn:
        return False
    descendant_prefix = f"{owner_fqn}."
    return any(
        candidate == owner_fqn or candidate.startswith(descendant_prefix)
        for candidate in node_owner_fqns(node)
    )
