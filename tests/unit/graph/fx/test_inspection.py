"""Contracts for shared FX graph inspection."""

from __future__ import annotations

import operator

import torch
from torch.fx import Graph

from mdc_llm_deploy.graph.fx.inspection import (
    flatten_nodes,
    linear_weight_name,
    node_target,
)


def test_flatten_nodes_preserves_nested_argument_order() -> None:
    graph = Graph()
    first = graph.placeholder("first")
    second = graph.placeholder("second")

    assert flatten_nodes(
        {"left": first, "right": (1, [second, first])}
    ) == (first, second, first)


def test_node_target_normalizes_torch_operator_schema() -> None:
    graph = Graph()
    value = graph.placeholder("value")
    node = graph.call_function(
        torch.ops.aten.relu.default,
        (value,),
    )

    assert node_target(node) == "aten::relu"


def test_linear_weight_name_requires_aten_linear_get_attr() -> None:
    graph = Graph()
    value = graph.placeholder("value")
    weight = graph.get_attr("layer.weight")
    linear = graph.call_function(
        torch.ops.aten.linear.default,
        (value, weight, None),
    )
    other = graph.call_function(operator.add, (value, value))
    malformed = graph.call_function(
        torch.ops.aten.linear.default,
        (value, value, None),
    )

    assert linear_weight_name(linear) == "layer.weight"
    assert linear_weight_name(other) is None
    assert linear_weight_name(malformed) is None
