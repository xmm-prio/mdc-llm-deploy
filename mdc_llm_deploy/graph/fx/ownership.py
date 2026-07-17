"""Shared FX node ownership helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from torch.fx import Node

__all__ = [
    "NodeOwnershipIndex",
    "is_fqn_descendant",
    "is_fqn_descendant_or_self",
    "node_belongs_to",
    "node_owner_fqns",
]


def is_fqn_descendant(candidate_fqn: str, ancestor_fqn: str) -> bool:
    """Return whether candidate is a strict dot-delimited FQN descendant."""
    if (
        not isinstance(candidate_fqn, str)
        or not candidate_fqn
        or not isinstance(ancestor_fqn, str)
        or not ancestor_fqn
    ):
        return False
    return candidate_fqn.startswith(f"{ancestor_fqn}.")


def is_fqn_descendant_or_self(
    candidate_fqn: str,
    ancestor_fqn: str,
) -> bool:
    """Return whether candidate is an FQN ancestor match or descendant."""
    if (
        not isinstance(candidate_fqn, str)
        or not candidate_fqn
        or not isinstance(ancestor_fqn, str)
        or not ancestor_fqn
    ):
        return False
    return candidate_fqn == ancestor_fqn or is_fqn_descendant(
        candidate_fqn,
        ancestor_fqn,
    )


def _owner_fqns_belong_to(
    owner_fqns: tuple[str, ...],
    owner_fqn: str,
) -> bool:
    return any(
        is_fqn_descendant_or_self(candidate, owner_fqn)
        for candidate in owner_fqns
    )


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
    return _owner_fqns_belong_to(node_owner_fqns(node), owner_fqn)


class NodeOwnershipIndex:
    """Snapshot FX node ownership for ordered queries within one graph call."""

    def __init__(self, nodes: Iterable[Node]) -> None:
        self._entries = tuple((node, node_owner_fqns(node)) for node in nodes)

    def nodes_belonging_to(self, owner_fqn: str) -> tuple[Node, ...]:
        """Return matching nodes in index construction order."""
        return tuple(
            node
            for node, owner_fqns in self._entries
            if _owner_fqns_belong_to(owner_fqns, owner_fqn)
        )
