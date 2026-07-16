from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.fx import Graph, Node

from mdc_llm_deploy.export.discovery import discover_metadata
from mdc_llm_deploy.graph.fx.ownership import node_belongs_to, node_owner_fqns


def _node(stack: Any) -> Node:
    graph = Graph()
    node = graph.placeholder("value")
    node.meta["nn_module_stack"] = stack
    return node


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
