from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
import torch
from torch import nn
from torch.fx import Graph, GraphModule

from mdc_llm_deploy.errors import UnsupportedPatternError
from mdc_llm_deploy.export.discovery import _discover_boundaries
from mdc_llm_deploy.graph.metadata import FusionBoundary


class _Rope(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.ones(1))


class _Attention(nn.Module):
    def __init__(self, *, with_rope: bool = False) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2, 2)
        self.k_proj = nn.Linear(2, 2)
        self.v_proj = nn.Linear(2, 2)
        self.o_proj = nn.Linear(2, 2)
        if with_rope:
            self.rotary = _Rope()


class _RmsNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.epsilon = 1e-6


def _graph(
    nodes: Sequence[tuple[str, Mapping[str, tuple[object, ...]]]],
) -> GraphModule:
    graph = Graph()
    created = []
    for name, stack in nodes:
        node = graph.placeholder(name)
        node.meta["nn_module_stack"] = stack
        created.append(node)
    graph.output(tuple(created))
    return GraphModule(nn.Module(), graph)


def test_nested_boundaries_claim_deepest_nodes_and_keep_sort_order() -> None:
    model = nn.Module()
    block = nn.Module()
    block.add_module("attn", _Attention(with_rope=True))
    model.add_module("block", block)
    graph = _graph(
        [
            ("attention_only", {"attn": ("block.attn", _Attention)}),
            (
                "shared",
                {
                    "attn": ("block.attn", _Attention),
                    "rope": ("block.attn.rotary", _Rope),
                },
            ),
        ]
    )

    assert _discover_boundaries(model, graph) == (
        FusionBoundary("attention", "block.attn", ("attention_only",)),
        FusionBoundary("rope", "block.attn.rotary", ("shared",)),
    )


def test_unrelated_boundary_overlap_keeps_existing_error() -> None:
    model = nn.Module()
    model.add_module("first", _Attention())
    model.add_module("second", _Attention())
    graph = _graph(
        [
            (
                "shared",
                {
                    "first": ("first", _Attention),
                    "second": ("second", _Attention),
                },
            )
        ]
    )

    with pytest.raises(
        UnsupportedPatternError,
        match=(
            "Overlapping fusion boundaries must have a strict FQN "
            "parent-child relationship: 'first', 'second'"
        ),
    ):
        _discover_boundaries(model, graph)


def test_boundaries_keep_kind_fqn_and_graph_node_sorting() -> None:
    model = nn.Module()
    model.add_module("z_attention", _Attention())
    model.add_module("a_norm", _RmsNorm())
    model.add_module("m_rope", _Rope())
    graph = _graph(
        [
            ("attention_second", {"owner": ("z_attention", _Attention)}),
            ("rope_node", {"owner": ("m_rope", _Rope)}),
            ("attention_first", {"owner": ("z_attention.q_proj", nn.Linear)}),
            ("norm_node", {"owner": ("a_norm", _RmsNorm)}),
        ]
    )

    assert _discover_boundaries(model, graph) == (
        FusionBoundary(
            "attention",
            "z_attention",
            ("attention_second", "attention_first"),
        ),
        FusionBoundary("rms_norm", "a_norm", ("norm_node",)),
        FusionBoundary("rope", "m_rope", ("rope_node",)),
    )


def test_empty_and_fully_claimed_parent_boundaries_are_skipped() -> None:
    model = nn.Module()
    parent = _Attention(with_rope=True)
    model.add_module("parent", parent)
    model.add_module("no_match", _Attention())
    graph = _graph(
        [("rope_only", {"owner": ("parent.rotary", _Rope)})]
    )

    assert _discover_boundaries(model, graph) == (
        FusionBoundary("rope", "parent.rotary", ("rope_only",)),
    )


def _frozen_resolve_boundary_overlaps(
    discovered: list[tuple[str, str, tuple[str, ...]]],
) -> tuple[FusionBoundary, ...]:
    node_sets = [set(nodes) for _, _, nodes in discovered]
    for index, (_, fqn, _) in enumerate(discovered):
        for other_index in range(index + 1, len(discovered)):
            if not node_sets[index] & node_sets[other_index]:
                continue
            other_fqn = discovered[other_index][1]
            if not (
                (bool(fqn) and other_fqn.startswith(f"{fqn}."))
                or (
                    bool(other_fqn)
                    and fqn.startswith(f"{other_fqn}.")
                )
            ):
                raise UnsupportedPatternError(
                    "Overlapping fusion boundaries must have a strict FQN "
                    f"parent-child relationship: {fqn!r}, {other_fqn!r}"
                )

    claimed: set[str] = set()
    result: list[FusionBoundary] = []
    for kind, fqn, nodes in sorted(
        discovered,
        key=lambda item: (-item[1].count("."), item[0], item[1]),
    ):
        owned = tuple(node for node in nodes if node not in claimed)
        if not owned:
            continue
        claimed.update(owned)
        result.append(FusionBoundary(kind, fqn, owned))
    return tuple(sorted(result, key=lambda item: (item.kind, item.fqn)))


def test_overlap_error_keeps_original_pair_order() -> None:
    from mdc_llm_deploy.export.discovery import _resolve_boundary_overlaps

    discovered = [
        ("attention", "first", ("late_shared",)),
        ("moe", "second", ("early_shared",)),
        ("rope", "third", ("early_shared",)),
        ("rms_norm", "fourth", ("late_shared",)),
    ]

    with pytest.raises(UnsupportedPatternError) as captured:
        _resolve_boundary_overlaps(discovered)

    assert str(captured.value) == (
        "Overlapping fusion boundaries must have a strict FQN "
        "parent-child relationship: 'first', 'fourth'"
    )


def test_overlap_pair_with_duplicate_nodes_is_validated_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    discovery = importlib.import_module("mdc_llm_deploy.export.discovery")

    calls: list[tuple[str, str]] = []

    def record_descendant(candidate: str, ancestor: str) -> bool:
        calls.append((candidate, ancestor))
        return False

    monkeypatch.setattr(discovery, "is_fqn_descendant", record_descendant)
    discovered = [
        ("attention", "first", ("shared_a", "shared_a", "shared_b")),
        ("moe", "second", ("shared_b", "shared_a", "shared_b")),
    ]

    with pytest.raises(
        UnsupportedPatternError,
        match="parent-child relationship: 'first', 'second'",
    ):
        discovery._resolve_boundary_overlaps(discovered)

    assert calls == [("second", "first"), ("first", "second")]


@pytest.mark.parametrize(
    ("first_fqn", "second_fqn", "allowed"),
    [
        ("parent", "parent.child", True),
        ("parent.child", "parent", True),
        ("same", "same", False),
        ("", "child", False),
        ("prefix", "prefix_child", False),
    ],
)
def test_overlap_keeps_strict_fqn_rules(
    first_fqn: str,
    second_fqn: str,
    *,
    allowed: bool,
) -> None:
    from mdc_llm_deploy.export.discovery import _resolve_boundary_overlaps

    discovered = [
        ("attention", first_fqn, ("shared", "first_only")),
        ("rope", second_fqn, ("shared", "second_only")),
    ]
    if allowed:
        assert _resolve_boundary_overlaps(discovered) == (
            _frozen_resolve_boundary_overlaps(discovered)
        )
        return

    with pytest.raises(UnsupportedPatternError) as captured:
        _resolve_boundary_overlaps(discovered)
    assert str(captured.value) == (
        "Overlapping fusion boundaries must have a strict FQN "
        f"parent-child relationship: {first_fqn!r}, {second_fqn!r}"
    )


def test_claim_stage_keeps_tuple_and_final_sorting_semantics() -> None:
    from mdc_llm_deploy.export.discovery import _resolve_boundary_overlaps

    discovered = [
        ("moe", "nest", ("nested_shared",)),
        ("rope", "nest.child", ("nested_shared", "nested_shared")),
        ("moe", "zeta", ("z_second", "z_first")),
        ("attention", "alpha", ("a_second", "a_first")),
    ]

    assert _resolve_boundary_overlaps(discovered) == (
        FusionBoundary("attention", "alpha", ("a_second", "a_first")),
        FusionBoundary("moe", "zeta", ("z_second", "z_first")),
        FusionBoundary(
            "rope",
            "nest.child",
            ("nested_shared", "nested_shared"),
        ),
    )


def test_overlap_index_matches_frozen_reference() -> None:
    import random

    from mdc_llm_deploy.export.discovery import _resolve_boundary_overlaps

    randomizer = random.Random(20260717)
    kinds = ("attention", "moe", "rms_norm", "rope")
    fqns = ("", "a", "a.b", "a.b.c", "ab", "x", "x.y", "same")
    node_names = tuple(f"node_{index}" for index in range(9))

    for _ in range(96):
        discovered = [
            (
                randomizer.choice(kinds),
                randomizer.choice(fqns),
                tuple(
                    randomizer.choice(node_names)
                    for _ in range(randomizer.randrange(6))
                ),
            )
            for _ in range(randomizer.randrange(9))
        ]
        try:
            expected = _frozen_resolve_boundary_overlaps(discovered)
        except UnsupportedPatternError as expected_error:
            with pytest.raises(UnsupportedPatternError) as actual_error:
                _resolve_boundary_overlaps(discovered)
            assert str(actual_error.value) == str(expected_error)
        else:
            assert _resolve_boundary_overlaps(discovered) == expected


def test_overlap_index_matches_reference_for_large_disjoint_load() -> None:
    from mdc_llm_deploy.export.discovery import _resolve_boundary_overlaps

    discovered = [
        (
            ("attention", "moe", "rms_norm", "rope")[boundary_index % 4],
            f"layer_{boundary_index}",
            tuple(
                f"boundary_{boundary_index}_node_{node_index}"
                for node_index in range(120)
            ),
        )
        for boundary_index in range(240)
    ]

    assert _resolve_boundary_overlaps(discovered) == (
        _frozen_resolve_boundary_overlaps(discovered)
    )
