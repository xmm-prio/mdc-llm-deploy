"""Fuse proven BNSD half-rotation subgraphs into ApplyRotaryPosEmb."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeGuard

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper

from ..graph.utilities import (
    GraphIndex,
    TensorInfo,
    attribute_int,
    constant_array,
    graph_names,
    remove_value_info,
    unique_name,
)
from ..schema import ROTARY_POSITION_EMBEDDING_OP
from .contracts import FusionPassResult

_PASS_NAME: Final = "apply_rotary_pos_emb"
_BNSD_LAYOUT: Final = 3
_SUPPORTED_DTYPES: Final = frozenset(
    {TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.FLOAT}
)


@dataclass(frozen=True, slots=True)
class _RotaryBranch:
    input_name: str
    cos_name: str
    sin_name: str
    output_name: str
    nodes: tuple[NodeProto, ...]
    slice_parameter_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RotaryPairMatch:
    query: _RotaryBranch
    key: _RotaryBranch

    @property
    def nodes(self) -> tuple[NodeProto, ...]:
        return (*self.query.nodes, *self.key.nodes)


class ApplyRotaryPosEmbFusionPass:
    """Fuse strict BNSD Q/K half-rotation pairs."""

    @property
    def name(self) -> str:
        """Return the stable pass name."""
        return _PASS_NAME

    def apply(self, model: onnx.ModelProto) -> FusionPassResult:
        """Fuse all non-overlapping supported Q/K RoPE pairs in place."""
        if not isinstance(model, onnx.ModelProto):
            raise TypeError("model must be an onnx.ModelProto")

        fused_node_names: list[str] = []
        while match := _find_next_match(model):
            fused_node_names.append(_replace_match(model, match))
        return FusionPassResult(self.name, len(fused_node_names), tuple(fused_node_names))


def fuse_apply_rotary_pos_emb(model: onnx.ModelProto) -> FusionPassResult:
    """Run the ApplyRotaryPosEmb fusion pass in place."""
    return APPLY_ROTARY_POS_EMB_FUSION_PASS.apply(model)


def _find_next_match(model: onnx.ModelProto) -> _RotaryPairMatch | None:
    index = GraphIndex(model)
    graph_outputs = {value.name for value in model.graph.output}
    branches = [
        branch
        for node in model.graph.node
        if (branch := _match_branch(index, graph_outputs, node)) is not None
    ]
    coefficient_groups: dict[tuple[str, str], list[_RotaryBranch]] = {}
    for branch in branches:
        coefficient_groups.setdefault((branch.cos_name, branch.sin_name), []).append(branch)

    positions = {id(node): position for position, node in enumerate(model.graph.node)}
    for candidates in coefficient_groups.values():
        if len(candidates) < 2 or len(candidates) % 2:
            continue
        pair = _find_nearest_pair(model, index, candidates, positions)
        if pair is not None:
            return pair
    return None


def _find_nearest_pair(
    model: onnx.ModelProto,
    index: GraphIndex,
    candidates: list[_RotaryBranch],
    positions: dict[int, int],
) -> _RotaryPairMatch | None:
    """Find the closest valid Q/K pair among branches sharing RoPE coefficients."""
    ranked_pairs: list[tuple[int, _RotaryBranch, _RotaryBranch]] = []
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            distance = abs(positions[id(first.nodes[-1])] - positions[id(second.nodes[-1])])
            ranked_pairs.append((distance, first, second))

    for _, first, second in sorted(ranked_pairs, key=lambda item: item[0]):
        pair = _validate_pair(index, first, second)
        if pair is not None and _has_replacement_position(model, index, pair):
            return pair
    return None


def _match_branch(
    index: GraphIndex,
    graph_outputs: set[str],
    output_add: NodeProto,
) -> _RotaryBranch | None:
    if not _is_node(output_add, "Add", inputs=2, outputs=1):
        return None
    first_mul = index.producer(output_add.input[0])
    second_mul = index.producer(output_add.input[1])
    if not _is_node(first_mul, "Mul", inputs=2, outputs=1):
        return None
    if not _is_node(second_mul, "Mul", inputs=2, outputs=1):
        return None

    for direct_mul, rotated_mul in ((first_mul, second_mul), (second_mul, first_mul)):
        for input_name, cos_name in _ordered_inputs(direct_mul):
            rotate_concat, sin_name = _node_and_other_input(index, rotated_mul, "Concat")
            if rotate_concat is None or sin_name is None:
                continue
            rotation = _match_half_rotation(index, rotate_concat, input_name)
            if rotation is None:
                continue
            nodes, slice_parameter_names = rotation
            matched_nodes = (*nodes, direct_mul, rotated_mul, output_add)
            if not _closed_branch(index, graph_outputs, matched_nodes, output_add):
                continue
            return _RotaryBranch(
                input_name=input_name,
                cos_name=cos_name,
                sin_name=sin_name,
                output_name=output_add.output[0],
                nodes=matched_nodes,
                slice_parameter_names=slice_parameter_names,
            )
    return None


def _match_half_rotation(
    index: GraphIndex,
    concat: NodeProto,
    input_name: str,
) -> tuple[tuple[NodeProto, ...], tuple[str, ...]] | None:
    if not _is_node(concat, "Concat", inputs=2, outputs=1):
        return None
    try:
        if attribute_int(concat, "axis") != -1:
            return None
    except ValueError:
        return None

    neg = index.producer(concat.input[0])
    first_slice = index.producer(concat.input[1])
    if not _is_node(neg, "Neg", inputs=1, outputs=1):
        return None
    second_slice = index.producer(neg.input[0])
    if not _is_node(first_slice, "Slice", inputs=3, outputs=1) and not _is_node(
        first_slice, "Slice", inputs=4, outputs=1
    ) and not _is_node(first_slice, "Slice", inputs=5, outputs=1):
        return None
    if not _is_node(second_slice, "Slice", inputs=3, outputs=1) and not _is_node(
        second_slice, "Slice", inputs=4, outputs=1
    ) and not _is_node(second_slice, "Slice", inputs=5, outputs=1):
        return None
    if first_slice.input[0] != input_name or second_slice.input[0] != input_name:
        return None

    input_info = index.tensor_info.get(input_name)
    if not _static_rank_four(input_info):
        return None
    assert input_info is not None
    head_dimension = input_info.shape[-1]
    if not isinstance(head_dimension, int) or head_dimension % 2:
        return None
    half = head_dimension // 2
    if not _is_half_slice(index, first_slice, 0, half, head_dimension):
        return None
    if not _is_half_slice(index, second_slice, half, head_dimension, head_dimension):
        return None

    parameter_names = tuple(
        dict.fromkeys((*first_slice.input[1:], *second_slice.input[1:]))
    )
    return (first_slice, second_slice, neg, concat), parameter_names


def _is_half_slice(
    index: GraphIndex,
    node: NodeProto,
    expected_start: int,
    expected_end: int,
    dimension: int,
) -> bool:
    starts = _constant_integers(index, node.input[1])
    ends = _constant_integers(index, node.input[2])
    axes = _constant_integers(index, node.input[3]) if len(node.input) >= 4 else (0,)
    steps = _constant_integers(index, node.input[4]) if len(node.input) >= 5 else (1,)
    if starts != (expected_start,) or axes not in {(-1,), (3,)} or steps != (1,):
        return False
    if ends is None or len(ends) != 1:
        return False
    actual_end = ends[0]
    if expected_end == dimension:
        return actual_end >= dimension
    return actual_end == expected_end


def _validate_pair(
    index: GraphIndex,
    first: _RotaryBranch,
    second: _RotaryBranch,
) -> _RotaryPairMatch | None:
    first_info = index.tensor_info.get(first.input_name)
    second_info = index.tensor_info.get(second.input_name)
    cos_info = index.tensor_info.get(first.cos_name)
    sin_info = index.tensor_info.get(first.sin_name)
    if not all(
        _static_rank_four(info)
        for info in (first_info, second_info, cos_info, sin_info)
    ):
        return None
    assert first_info is not None
    assert second_info is not None
    assert cos_info is not None
    assert sin_info is not None
    if not _valid_dtypes(first_info, second_info, cos_info, sin_info):
        return None
    if cos_info.shape != sin_info.shape or cos_info.shape[1] != 1:
        return None
    if first_info.shape[0] != second_info.shape[0]:
        return None
    if first_info.shape[2:] != second_info.shape[2:]:
        return None
    if cos_info.shape[-1] != first_info.shape[-1]:
        return None
    if any(
        cos_dimension not in (1, input_dimension)
        for cos_dimension, input_dimension in zip(
            (cos_info.shape[0], cos_info.shape[2]),
            (first_info.shape[0], first_info.shape[2]),
            strict=True,
        )
    ):
        return None

    first_heads = first_info.shape[1]
    second_heads = second_info.shape[1]
    if not isinstance(first_heads, int) or not isinstance(second_heads, int):
        return None
    if first_heads >= second_heads and first_heads % second_heads == 0:
        return _RotaryPairMatch(first, second)
    if second_heads % first_heads == 0:
        return _RotaryPairMatch(second, first)
    return None


def _valid_dtypes(*infos: TensorInfo) -> bool:
    elem_types = {info.elem_type for info in infos}
    return len(elem_types) == 1 and next(iter(elem_types)) in _SUPPORTED_DTYPES


def _closed_branch(
    index: GraphIndex,
    graph_outputs: set[str],
    nodes: tuple[NodeProto, ...],
    output_add: NodeProto,
) -> bool:
    matched_ids = {id(node) for node in nodes}
    for node in nodes:
        if node is output_add:
            continue
        for output in node.output:
            if output in graph_outputs:
                return False
            if any(id(user) not in matched_ids for user in index.users(output)):
                return False
    return True


def _has_replacement_position(
    model: onnx.ModelProto,
    index: GraphIndex,
    match: _RotaryPairMatch,
) -> bool:
    positions = {id(node): position for position, node in enumerate(model.graph.node)}
    matched_ids = {id(node) for node in match.nodes}
    input_producers = (
        index.producer(name)
        for name in (
            match.query.input_name,
            match.key.input_name,
            match.query.cos_name,
            match.query.sin_name,
        )
    )
    lower_bound = max(
        (positions[id(node)] for node in input_producers if node is not None),
        default=-1,
    )
    output_users = (
        user
        for name in (match.query.output_name, match.key.output_name)
        for user in index.users(name)
        if id(user) not in matched_ids
    )
    upper_bound = min(
        (positions[id(node)] for node in output_users),
        default=len(model.graph.node),
    )
    return lower_bound < upper_bound


def _replace_match(model: onnx.ModelProto, match: _RotaryPairMatch) -> str:
    graph = model.graph
    index = GraphIndex(model)
    reserved_names = graph_names(model)
    node_name = unique_name(
        reserved_names,
        f"{match.query.output_name}_apply_rotary_pos_emb",
    )
    fused_node = helper.make_node(
        ROTARY_POSITION_EMBEDDING_OP,
        [
            match.query.input_name,
            match.key.input_name,
            match.query.cos_name,
            match.query.sin_name,
        ],
        [match.query.output_name, match.key.output_name],
        name=node_name,
        layout=_BNSD_LAYOUT,
        rotary_mode="half",
    )

    positions = {id(node): position for position, node in enumerate(graph.node)}
    input_producers = (
        index.producer(name)
        for name in fused_node.input
    )
    lower_bound = max(
        (positions[id(node)] for node in input_producers if node is not None),
        default=-1,
    )
    removed_ids = {id(node) for node in match.nodes}
    kept_with_positions = [
        (position, node)
        for position, node in enumerate(graph.node)
        if id(node) not in removed_ids
    ]
    insertion_index = sum(position <= lower_bound for position, _ in kept_with_positions)
    kept_nodes = [node for _, node in kept_with_positions]
    kept_nodes.insert(insertion_index, fused_node)
    del graph.node[:]
    graph.node.extend(kept_nodes)

    stale_values = {
        output
        for node in match.nodes
        for output in node.output
        if output not in {match.query.output_name, match.key.output_name}
    }
    remove_value_info(model, stale_values)
    _remove_dead_slice_parameters(
        model,
        (*match.query.slice_parameter_names, *match.key.slice_parameter_names),
    )
    return node_name


def _remove_dead_slice_parameters(
    model: onnx.ModelProto,
    candidate_names: tuple[str, ...],
) -> None:
    index = GraphIndex(model)
    graph_outputs = {value.name for value in model.graph.output}
    dead_names = {
        name
        for name in candidate_names
        if not index.users(name) and name not in graph_outputs
    }
    if not dead_names:
        return
    kept_initializers = [
        tensor for tensor in model.graph.initializer if tensor.name not in dead_names
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    kept_nodes = [
        node
        for node in model.graph.node
        if not (
            node.op_type == "Constant"
            and len(node.output) == 1
            and node.output[0] in dead_names
        )
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)


def _constant_integers(index: GraphIndex, name: str) -> tuple[int, ...] | None:
    value = constant_array(index, name)
    if value is None or not np.issubdtype(value.dtype, np.integer):
        return None
    return tuple(int(item) for item in np.asarray(value).reshape(-1))


def _ordered_inputs(node: NodeProto) -> tuple[tuple[str, str], tuple[str, str]]:
    return ((node.input[0], node.input[1]), (node.input[1], node.input[0]))


def _node_and_other_input(
    index: GraphIndex,
    node: NodeProto,
    op_type: str,
) -> tuple[NodeProto | None, str | None]:
    for candidate_name, other_name in _ordered_inputs(node):
        candidate = index.producer(candidate_name)
        if candidate is not None and candidate.op_type == op_type:
            return candidate, other_name
    return None, None


def _static_rank_four(info: TensorInfo | None) -> bool:
    return (
        info is not None
        and len(info.shape) == 4
        and all(isinstance(dimension, int) and dimension > 0 for dimension in info.shape)
    )


def _is_node(
    node: NodeProto | None,
    op_type: str,
    *,
    inputs: int,
    outputs: int,
) -> TypeGuard[NodeProto]:
    return (
        node is not None
        and node.domain in ("", "ai.onnx")
        and node.op_type == op_type
        and len(node.input) == inputs
        and len(node.output) == outputs
        and all(node.input)
        and all(node.output)
    )


APPLY_ROTARY_POS_EMB_FUSION_PASS: Final = ApplyRotaryPosEmbFusionPass()

__all__ = [
    "APPLY_ROTARY_POS_EMB_FUSION_PASS",
    "ApplyRotaryPosEmbFusionPass",
    "fuse_apply_rotary_pos_emb",
]
