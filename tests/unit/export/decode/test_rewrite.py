"""Tests for semantic decode graph rewrites."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn
from torch.fx import Graph, GraphModule, Node

from mdc_llm_deploy.export.decode.rewrite import remove_prefill_causal_mask
from mdc_llm_deploy.graph.fx.inspection import node_target


def _masked_fill_decoy(
    scores: Tensor,
    mask: Tensor,
    value: float,
) -> Tensor:
    return scores.masked_fill(mask, value)


def _mask_graph(
    mask: Tensor,
    *,
    invert: bool = False,
    dynamic: bool = False,
    target: Callable[..., Tensor] = torch.ops.aten.masked_fill.Scalar,
) -> tuple[GraphModule, Node]:
    root = nn.Module()
    root.register_buffer("mask", mask)
    graph = Graph()
    scores = graph.placeholder("scores")
    producer = (
        graph.placeholder("mask") if dynamic else graph.get_attr("mask")
    )
    if invert:
        producer = graph.call_function(
            torch.ops.aten.bitwise_not.default,
            (producer,),
        )
    fill = graph.call_function(target, (scores, producer, -100.0))
    graph.output(fill)
    return GraphModule(root, graph), fill


def _masked_fill_nodes(graph: GraphModule) -> tuple[Node, ...]:
    return tuple(
        node
        for node in graph.graph.nodes
        if "masked_fill" in node_target(node)
    )


def test_remove_prefill_causal_mask_removes_proven_attention_mask() -> None:
    sequence_length = 4
    visible = torch.ones(sequence_length, sequence_length, dtype=torch.bool).tril()
    graph, fill = _mask_graph(visible.reshape(1, 1, 4, 4), invert=True)

    remove_prefill_causal_mask(
        graph,
        attention_nodes={fill.name},
        sequence_length=sequence_length,
    )

    assert not _masked_fill_nodes(graph)


@pytest.mark.parametrize(
    ("mask", "dynamic", "remove_attribute"),
    [
        (torch.eye(4, dtype=torch.bool), False, False),
        (torch.ones(3, 4, dtype=torch.bool), False, False),
        (torch.ones(4, 4), False, False),
        (torch.ones(4, 4, dtype=torch.bool).triu(1), False, True),
        (torch.ones(4, 4, dtype=torch.bool), True, False),
    ],
    ids=[
        "non-triangular",
        "wrong-shape",
        "non-bool",
        "missing-attribute",
        "dynamic",
    ],
)
def test_remove_prefill_causal_mask_preserves_unproven_attention_masks(
    mask: Tensor,
    dynamic: bool,
    remove_attribute: bool,
) -> None:
    graph, fill = _mask_graph(mask, dynamic=dynamic)
    if remove_attribute:
        delattr(graph, "mask")

    remove_prefill_causal_mask(
        graph,
        attention_nodes={fill.name},
        sequence_length=4,
    )

    assert _masked_fill_nodes(graph) == (fill,)


def test_remove_prefill_causal_mask_preserves_non_attention_node() -> None:
    mask = torch.ones(4, 4, dtype=torch.bool).triu(1)
    graph, fill = _mask_graph(mask)

    remove_prefill_causal_mask(
        graph,
        attention_nodes=set(),
        sequence_length=4,
    )

    assert _masked_fill_nodes(graph) == (fill,)


def test_remove_prefill_causal_mask_preserves_unsupported_static_mask() -> None:
    mask = torch.sparse_coo_tensor(
        torch.tensor([[0], [1]]),
        torch.tensor([True]),
        (4, 4),
    )
    graph, fill = _mask_graph(mask, invert=True)

    remove_prefill_causal_mask(
        graph,
        attention_nodes={fill.name},
        sequence_length=4,
    )

    assert _masked_fill_nodes(graph) == (fill,)


def test_remove_prefill_causal_mask_requires_exact_aten_target() -> None:
    mask = torch.ones(4, 4, dtype=torch.bool).triu(1)
    graph, fill = _mask_graph(mask, target=_masked_fill_decoy)

    remove_prefill_causal_mask(
        graph,
        attention_nodes={fill.name},
        sequence_length=4,
    )

    assert _masked_fill_nodes(graph) == (fill,)


def test_remove_prefill_causal_mask_rewrites_shared_mask_users() -> None:
    root = nn.Module()
    root.register_buffer("mask", torch.ones(4, 4, dtype=torch.bool).triu(1))
    raw_graph = Graph()
    scores = raw_graph.placeholder("scores")
    mask = raw_graph.get_attr("mask")
    fills = tuple(
        raw_graph.call_function(
            torch.ops.aten.masked_fill.Scalar,
            (scores, mask, value),
        )
        for value in (-100.0, -200.0)
    )
    raw_graph.output(fills)
    graph = GraphModule(root, raw_graph)

    remove_prefill_causal_mask(
        graph,
        attention_nodes={node.name for node in fills},
        sequence_length=4,
    )

    assert not _masked_fill_nodes(graph)
    graph.graph.lint()
    graph.recompile()
    scores = torch.randn(4, 4)
    first, second = graph(scores)
    assert first is scores
    assert second is scores
