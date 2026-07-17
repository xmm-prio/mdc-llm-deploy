from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.fx import Graph, Node

import mdc_llm_deploy.graph.fx.ownership as ownership_module
from mdc_llm_deploy.export.discovery import discover_metadata
from mdc_llm_deploy.graph.fx.ownership import (
    NodeOwnershipIndex,
    is_fqn_descendant,
    is_fqn_descendant_or_self,
    node_belongs_to,
    node_owner_fqns,
)


def _node(stack: Any) -> Node:
    graph = Graph()
    node = graph.placeholder("value")
    node.meta["nn_module_stack"] = stack
    return node


@pytest.mark.parametrize(
    ("candidate", "ancestor", "expected"),
    [
        ("parent", "parent", False),
        ("parent.child", "parent", True),
        ("parent.child.grandchild", "parent", True),
        ("parent2.child", "parent", False),
        ("prefix_child", "prefix", False),
        ("block.parent", "parent", False),
        ("", "parent", False),
        ("parent.child", "", False),
        ("", "", False),
        (object(), "parent", False),
        ("parent.child", object(), False),
    ],
)
def test_is_fqn_descendant_matrix(
    candidate: object,
    ancestor: object,
    expected: bool,
) -> None:
    assert (
        is_fqn_descendant(  # type: ignore[arg-type]
            candidate,
            ancestor,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("candidate", "ancestor", "expected"),
    [
        ("parent", "parent", True),
        ("parent.child", "parent", True),
        ("parent.child.grandchild", "parent", True),
        ("parent2.child", "parent", False),
        ("prefix_child", "prefix", False),
        ("block.parent", "parent", False),
        ("", "parent", False),
        ("parent.child", "", False),
        ("", "", False),
        (object(), "parent", False),
        ("parent.child", object(), False),
    ],
)
def test_is_fqn_descendant_or_self_matrix(
    candidate: object,
    ancestor: object,
    expected: bool,
) -> None:
    assert (
        is_fqn_descendant_or_self(  # type: ignore[arg-type]
            candidate,
            ancestor,
        )
        is expected
    )


@pytest.mark.parametrize(
    "stack",
    [
        None,
        (),
        [],
        "self_attn",
        {"owner": "self_attn"},
        {"owner": ()},
        {"owner": ("", object())},
        {"owner": (None, object())},
    ],
)
def test_node_owner_fqns_rejects_malformed_stack(stack: Any) -> None:
    assert node_owner_fqns(_node(stack)) == ()


def test_node_owner_fqns_returns_stable_unique_tuple_fqns() -> None:
    node = _node(
        {
            "first": ("self_attn", object()),
            "duplicate": ("self_attn", object()),
            "child": ("self_attn.rotary", object()),
            "ignored": "self_attn.q_proj",
        }
    )

    assert node_owner_fqns(node) == ("self_attn", "self_attn.rotary")


@pytest.mark.parametrize(
    ("candidate", "owner", "expected"),
    [
        ("self_attn", "self_attn", True),
        ("self_attn.rotary", "self_attn", True),
        ("self_attn2", "self_attn", False),
        ("self_attn_extra.rotary", "self_attn", False),
        ("block.self_attn", "self_attn", False),
        ("self_attn", "", False),
    ],
)
def test_node_belongs_to_uses_exact_or_descendant_matching(
    candidate: str,
    owner: str,
    expected: bool,
) -> None:
    assert node_belongs_to(_node({"owner": (candidate, object())}), owner) is expected


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("self_attn", True),
        ("self_attn.rotary", True),
        ("self_attn2", False),
    ],
)
def test_discovery_uses_shared_fqn_rule(candidate: str, expected: bool) -> None:
    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(2, 2)
            self.k_proj = torch.nn.Linear(2, 2)
            self.v_proj = torch.nn.Linear(2, 2)
            self.o_proj = torch.nn.Linear(2, 2)

    model = torch.nn.Module()
    model.add_module("self_attn", Attention())
    fx_graph = Graph()
    value = fx_graph.placeholder("x")
    value.meta["val"] = torch.ones(1, 2)
    value.meta["nn_module_stack"] = {"owner": (candidate, Attention)}
    fx_graph.output(value)
    graph = torch.fx.GraphModule(torch.nn.Module(), fx_graph)

    result = discover_metadata(model, graph, {"x": torch.ones(1, 2)})

    assert bool(result.boundaries) is expected


def test_ownership_index_matches_single_node_api_in_input_order() -> None:
    nodes = [
        _node(
            {
                "exact": ("self_attn", object()),
                "duplicate": ("self_attn", object()),
            }
        ),
        _node({"child": ("self_attn.rotary", object())}),
        _node({"prefix": ("self_attn2", object())}),
        _node({"reverse": ("block.self_attn", object())}),
        _node(None),
        _node({"synthetic": ("_empty_nn_module_stack_from_metadata_hook", object())}),
    ]
    index = NodeOwnershipIndex(iter(nodes))

    for owner in (
        "self_attn",
        "self_attn.rotary",
        "block",
        "_empty_nn_module_stack_from_metadata_hook",
        "",
    ):
        expected = tuple(node for node in nodes if node_belongs_to(node, owner))
        actual = index.nodes_belonging_to(owner)

        assert actual == expected
        assert all(
            indexed is original
            for indexed, original in zip(actual, expected, strict=True)
        )

    assert index.nodes_belonging_to(object()) == ()  # type: ignore[arg-type]


def test_ownership_index_parses_each_node_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = [
        _node({"owner": ("self_attn", object())}),
        _node({"owner": ("self_attn.rotary", object())}),
        _node(None),
    ]
    original = ownership_module.node_owner_fqns
    parsed: list[Node] = []

    def counted(node: Node) -> tuple[str, ...]:
        parsed.append(node)
        return original(node)

    monkeypatch.setattr(ownership_module, "node_owner_fqns", counted)

    index = NodeOwnershipIndex(node for node in nodes)

    assert parsed == nodes
    assert index.nodes_belonging_to("self_attn") == tuple(nodes[:2])
    assert index.nodes_belonging_to("self_attn.rotary") == (nodes[1],)
    assert index.nodes_belonging_to("") == ()
    assert parsed == nodes


def test_ownership_index_is_construction_time_snapshot() -> None:
    node = _node({"owner": ("self_attn", object())})
    index = NodeOwnershipIndex([node])

    node.meta["nn_module_stack"] = {"owner": ("other", object())}

    assert index.nodes_belonging_to("self_attn") == (node,)
    assert index.nodes_belonging_to("other") == ()
    assert not node_belongs_to(node, "self_attn")
    assert node_belongs_to(node, "other")
