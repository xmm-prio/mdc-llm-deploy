"""Fuse statically proven BNSD attention into FusedInferAttentionScore."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, TypeGuard, cast

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper

from .._graph import (
    GraphIndex,
    TensorInfo,
    attribute_int,
    attribute_ints,
    constant_array,
    graph_names,
    remove_value_info,
    unique_name,
)
from ..schemas import FUSED_INFER_ATTENTION_SCORE_OP
from .base import FusionPassResult

_PASS_NAME: Final = "fused_infer_attention_score"
_SUPPORTED_DTYPES: Final = frozenset({TensorProto.FLOAT16, TensorProto.BFLOAT16})
_MAX_TOKENS: Final = 2_147_483_647
_BFLOAT16_MIN: Final = -3.3895313892515355e38


@dataclass(frozen=True, slots=True)
class _ScoreMatch:
    query_name: str
    transposed_key_name: str
    scale: float
    nodes: tuple[NodeProto, ...]


@dataclass(frozen=True, slots=True)
class _TensorMatch:
    source_name: str
    nodes: tuple[NodeProto, ...]


@dataclass(frozen=True, slots=True)
class _MaskMatch:
    score_name: str
    mask_name: str | None
    invert_mask: bool
    additive_sentinel: float | None
    nodes: tuple[NodeProto, ...]


@dataclass(frozen=True, slots=True)
class _ProbabilityMatch:
    softmax: NodeProto
    probability_name: str
    nodes: tuple[NodeProto, ...]


@dataclass(frozen=True, slots=True)
class _AttentionMatch:
    query_name: str
    key_name: str
    value_name: str
    mask_name: str | None
    output_name: str
    scale: float
    num_heads: int
    num_key_value_heads: int
    invert_mask: bool
    additive_sentinel: float | None
    nodes: tuple[NodeProto, ...]


class FusedInferAttentionScoreFusionPass:
    """Fuse strict FP16/BF16 eager and decomposed SDPA attention graphs."""

    @property
    def name(self) -> str:
        """Return the stable pass name."""
        return _PASS_NAME

    def apply(self, model: onnx.ModelProto) -> FusionPassResult:
        """Fuse all non-overlapping supported attention subgraphs in place."""
        if not isinstance(model, onnx.ModelProto):
            raise TypeError("model must be an onnx.ModelProto")

        fused_node_names: list[str] = []
        while match := _find_next_match(model):
            fused_node_names.append(_replace_match(model, match))
        return FusionPassResult(self.name, len(fused_node_names), tuple(fused_node_names))


def fuse_fused_infer_attention_score(model: onnx.ModelProto) -> FusionPassResult:
    """Run the FusedInferAttentionScore fusion pass in place."""
    return FUSED_INFER_ATTENTION_SCORE_FUSION_PASS.apply(model)


def _find_next_match(model: onnx.ModelProto) -> _AttentionMatch | None:
    index = GraphIndex(model)
    graph_outputs = {value.name for value in model.graph.output}
    for node in model.graph.node:
        match = _match_from_output_matmul(index, graph_outputs, node)
        if match is not None:
            return match
    return None


def _match_from_output_matmul(
    index: GraphIndex,
    graph_outputs: set[str],
    output_matmul: NodeProto,
) -> _AttentionMatch | None:
    if not _is_node(output_matmul, "MatMul", inputs=2, outputs=1):
        return None
    probabilities = _match_probabilities(index, output_matmul.input[0])
    if probabilities is None:
        return None
    masked = _match_masked_score(index, probabilities.softmax.input[0])
    if masked is None:
        return None
    score = _match_score(index, masked.score_name)
    if score is None:
        return None

    query_info = index.tensor_info.get(score.query_name)
    if not _supported_bnsd(query_info):
        return None
    assert query_info is not None
    query_shape = _static_shape(query_info.shape, rank=4)
    if query_shape is None:
        return None
    batch, num_heads, query_length, head_dim = query_shape

    key = _match_key(index, score.transposed_key_name, query_info)
    value = _match_repeated_bnsd(index, output_matmul.input[1], query_info)
    if key is None or value is None:
        return None
    key_info = index.tensor_info.get(key.source_name)
    value_info = index.tensor_info.get(value.source_name)
    if key_info is None or value_info is None or key_info != value_info:
        return None
    key_shape = _static_shape(key_info.shape, rank=4)
    if key_info.elem_type != query_info.elem_type or key_shape is None:
        return None
    key_batch, kv_heads, key_length, key_head_dim = key_shape
    if (
        key_batch != batch
        or key_head_dim != head_dim
        or kv_heads <= 0
        or key_length <= 0
        or num_heads % kv_heads
    ):
        return None

    output_info = index.tensor_info.get(output_matmul.output[0])
    if output_info != TensorInfo(query_info.elem_type, (batch, num_heads, query_length, head_dim)):
        return None
    if not _valid_mask(index, masked, (batch, num_heads, query_length, key_length), query_info):
        return None

    nodes = _unique_nodes(
        (
            *score.nodes,
            *key.nodes,
            *value.nodes,
            *masked.nodes,
            *probabilities.nodes,
            output_matmul,
        )
    )
    if not _closed_match(index, graph_outputs, nodes, output_matmul.output[0]):
        return None
    return _AttentionMatch(
        query_name=score.query_name,
        key_name=key.source_name,
        value_name=value.source_name,
        mask_name=masked.mask_name,
        output_name=output_matmul.output[0],
        scale=score.scale,
        num_heads=num_heads,
        num_key_value_heads=kv_heads,
        invert_mask=masked.invert_mask,
        additive_sentinel=masked.additive_sentinel,
        nodes=nodes,
    )


def _match_probabilities(index: GraphIndex, value_name: str) -> _ProbabilityMatch | None:
    producer = index.producer(value_name)
    if _is_softmax(producer):
        return _ProbabilityMatch(producer, value_name, (producer,))
    if not _is_node(producer, "Where", inputs=3, outputs=1):
        return None

    condition = index.producer(producer.input[0])
    softmax = index.producer(producer.input[2])
    zero = _scalar(index, producer.input[1])
    if (
        not _is_node(condition, "IsNaN", inputs=1, outputs=1)
        or not _is_softmax(softmax)
        or condition.input[0] != softmax.output[0]
        or producer.input[2] != softmax.output[0]
        or zero != 0.0
    ):
        return None
    return _ProbabilityMatch(
        softmax,
        producer.output[0],
        (softmax, condition, producer),
    )


def _match_masked_score(index: GraphIndex, value_name: str) -> _MaskMatch | None:
    producer = index.producer(value_name)
    if _is_node(producer, "Add", inputs=2, outputs=1):
        for score_name, mask_value_name in (
            (producer.input[0], producer.input[1]),
            (producer.input[1], producer.input[0]),
        ):
            if _match_score(index, score_name) is None:
                continue
            additive_where = _match_additive_where(index, mask_value_name)
            if additive_where is not None:
                mask_name, invert = additive_where
                where = index.producer(mask_value_name)
                assert where is not None
                return _MaskMatch(score_name, mask_name, invert, None, (producer, where))
            sentinel = _strict_additive_mask(index, mask_value_name)
            if sentinel is not None:
                return _MaskMatch(
                    score_name,
                    mask_value_name,
                    False,
                    sentinel,
                    (producer,),
                )
        return None

    if _is_node(producer, "Where", inputs=3, outputs=1):
        mask_info = index.tensor_info.get(producer.input[0])
        negative = _scalar(index, producer.input[1])
        if (
            mask_info is not None
            and mask_info.elem_type == TensorProto.BOOL
            and _match_score(index, producer.input[2]) is not None
            and _is_negative_sentinel(negative, _score_dtype(index, producer.input[2]))
        ):
            return _MaskMatch(
                producer.input[2],
                producer.input[0],
                False,
                None,
                (producer,),
            )
        return None

    if _match_score(index, value_name) is not None:
        return _MaskMatch(value_name, None, False, None, ())
    return None


def _match_additive_where(
    index: GraphIndex,
    value_name: str,
) -> tuple[str, bool] | None:
    where = index.producer(value_name)
    if not _is_node(where, "Where", inputs=3, outputs=1):
        return None
    condition_info = index.tensor_info.get(where.input[0])
    output_info = index.tensor_info.get(where.output[0])
    if condition_info is None or condition_info.elem_type != TensorProto.BOOL or output_info is None:
        return None
    true_value = _scalar(index, where.input[1])
    false_value = _scalar(index, where.input[2])
    if true_value == 0.0 and _is_negative_sentinel(false_value, output_info.elem_type):
        return where.input[0], True
    if false_value == 0.0 and _is_negative_sentinel(true_value, output_info.elem_type):
        return where.input[0], False
    return None


def _match_score(index: GraphIndex, value_name: str) -> _ScoreMatch | None:
    producer = index.producer(value_name)
    if _is_node(producer, "Mul", inputs=2, outputs=1):
        for matmul_name, scale_name in (
            (producer.input[0], producer.input[1]),
            (producer.input[1], producer.input[0]),
        ):
            matmul = index.producer(matmul_name)
            scale = _positive_scalar(index, scale_name)
            if _is_node(matmul, "MatMul", inputs=2, outputs=1) and scale is not None:
                return _ScoreMatch(matmul.input[0], matmul.input[1], scale, (matmul, producer))
        return None
    if not _is_node(producer, "MatMul", inputs=2, outputs=1):
        return None

    query_name, query_scale, query_mul = _unwrap_scaled_value(index, producer.input[0])
    key_name, key_scale, key_mul = _unwrap_scaled_value(index, producer.input[1])
    scale = query_scale * key_scale
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return _ScoreMatch(
        query_name,
        key_name,
        scale,
        _unique_nodes((query_mul, key_mul, producer)),
    )


def _unwrap_scaled_value(
    index: GraphIndex,
    value_name: str,
) -> tuple[str, float, NodeProto | None]:
    producer = index.producer(value_name)
    if not _is_node(producer, "Mul", inputs=2, outputs=1):
        return value_name, 1.0, None
    for tensor_name, scale_name in (
        (producer.input[0], producer.input[1]),
        (producer.input[1], producer.input[0]),
    ):
        scale = _positive_scalar(index, scale_name)
        if scale is not None:
            return tensor_name, scale, producer
    return value_name, 1.0, None


def _match_key(
    index: GraphIndex,
    value_name: str,
    query_info: TensorInfo,
) -> _TensorMatch | None:
    transpose = index.producer(value_name)
    if _is_node(transpose, "Transpose", inputs=1, outputs=1):
        try:
            permutation = attribute_ints(transpose, "perm")
        except ValueError:
            return None
        if permutation == (0, 1, 3, 2):
            repeated = _match_repeated_bnsd(index, transpose.input[0], query_info)
            if repeated is not None:
                return _TensorMatch(repeated.source_name, (*repeated.nodes, transpose))

    outer_reshape = index.producer(value_name)
    if not _is_node(outer_reshape, "Reshape", inputs=2, outputs=1):
        return None
    transpose = index.producer(outer_reshape.input[0])
    if not _is_node(transpose, "Transpose", inputs=1, outputs=1):
        return None
    try:
        permutation = attribute_ints(transpose, "perm")
    except ValueError:
        return None
    inner_reshape = index.producer(transpose.input[0])
    if permutation != (0, 2, 1) or not _is_node(inner_reshape, "Reshape", inputs=2, outputs=1):
        return None
    repeated_info = index.tensor_info.get(inner_reshape.input[0])
    inner_info = index.tensor_info.get(inner_reshape.output[0])
    transposed_info = index.tensor_info.get(transpose.output[0])
    outer_info = index.tensor_info.get(outer_reshape.output[0])
    if repeated_info is None:
        return None
    batch, heads, _, head_dim = query_info.shape
    if len(repeated_info.shape) == 5:
        repeated = _match_expanded_bnsd(index, inner_reshape.input[0], query_info)
        if repeated is None:
            return None
        source_info = index.tensor_info.get(repeated.source_name)
        if source_info is None:
            return None
        sequence = source_info.shape[2]
    elif len(repeated_info.shape) == 4:
        repeated = _match_repeated_bnsd(index, inner_reshape.input[0], query_info)
        if repeated is None:
            return None
        sequence = repeated_info.shape[2]
    else:
        return None
    if (
        not isinstance(batch, int)
        or not isinstance(heads, int)
        or inner_info != TensorInfo(query_info.elem_type, (batch * heads, sequence, head_dim))
        or transposed_info
        != TensorInfo(query_info.elem_type, (batch * heads, head_dim, sequence))
        or outer_info != TensorInfo(query_info.elem_type, (batch, heads, head_dim, sequence))
    ):
        return None
    return _TensorMatch(
        repeated.source_name,
        (*repeated.nodes, inner_reshape, transpose, outer_reshape),
    )


def _match_repeated_bnsd(
    index: GraphIndex,
    value_name: str,
    query_info: TensorInfo,
) -> _TensorMatch | None:
    value_info = index.tensor_info.get(value_name)
    if value_info is None or len(value_info.shape) != 4:
        return None
    query_shape = _static_shape(query_info.shape, rank=4)
    if query_shape is None:
        return None
    batch, query_heads, _, head_dim = query_shape
    if (
        value_info.elem_type != query_info.elem_type
        or value_info.shape[0] != batch
        or value_info.shape[1] != query_heads
        or value_info.shape[3] != head_dim
    ):
        return None

    reshape = index.producer(value_name)
    if not _is_node(reshape, "Reshape", inputs=2, outputs=1):
        return _TensorMatch(value_name, ())
    expand = index.producer(reshape.input[0])
    if not _is_node(expand, "Expand", inputs=2, outputs=1):
        return _TensorMatch(value_name, ())
    unsqueeze = index.producer(expand.input[0])
    if not _is_node(unsqueeze, "Unsqueeze", inputs=2, outputs=1):
        return None
    axes = constant_array(index, unsqueeze.input[1])
    source_info = index.tensor_info.get(unsqueeze.input[0])
    unsqueezed_info = index.tensor_info.get(unsqueeze.output[0])
    expanded_info = index.tensor_info.get(expand.output[0])
    if axes is None or tuple(int(value) for value in axes.reshape(-1)) != (2,):
        return None
    if source_info is None:
        return None
    source_shape = _static_shape(source_info.shape, rank=4)
    if source_shape is None:
        return None
    source_batch, kv_heads, sequence, source_head_dim = source_shape
    if (
        source_info.elem_type != query_info.elem_type
        or source_batch != batch
        or source_head_dim != head_dim
        or kv_heads <= 0
        or query_heads % kv_heads
    ):
        return None
    repeats = query_heads // kv_heads
    expected_unsqueezed = TensorInfo(
        source_info.elem_type,
        (batch, kv_heads, 1, sequence, head_dim),
    )
    expected_expanded = TensorInfo(
        source_info.elem_type,
        (batch, kv_heads, repeats, sequence, head_dim),
    )
    if unsqueezed_info != expected_unsqueezed or expanded_info != expected_expanded:
        return None
    if value_info.shape != (batch, query_heads, sequence, head_dim):
        return None
    return _TensorMatch(unsqueeze.input[0], (unsqueeze, expand, reshape))


def _match_expanded_bnsd(
    index: GraphIndex,
    value_name: str,
    query_info: TensorInfo,
) -> _TensorMatch | None:
    expanded_info = index.tensor_info.get(value_name)
    expand = index.producer(value_name)
    if expanded_info is None or not _is_node(expand, "Expand", inputs=2, outputs=1):
        return None
    unsqueeze = index.producer(expand.input[0])
    if not _is_node(unsqueeze, "Unsqueeze", inputs=2, outputs=1):
        return None
    axes = constant_array(index, unsqueeze.input[1])
    source_info = index.tensor_info.get(unsqueeze.input[0])
    unsqueezed_info = index.tensor_info.get(unsqueeze.output[0])
    if axes is None or tuple(int(value) for value in axes.reshape(-1)) != (2,):
        return None
    if source_info is None:
        return None
    query_shape = _static_shape(query_info.shape, rank=4)
    source_shape = _static_shape(source_info.shape, rank=4)
    if query_shape is None or source_shape is None:
        return None
    batch, query_heads, _, head_dim = query_shape
    source_batch, kv_heads, sequence, source_head_dim = source_shape
    if (
        source_info.elem_type != query_info.elem_type
        or source_batch != batch
        or source_head_dim != head_dim
        or kv_heads <= 0
        or query_heads % kv_heads
    ):
        return None
    repeats = query_heads // kv_heads
    if unsqueezed_info != TensorInfo(
        source_info.elem_type,
        (batch, kv_heads, 1, sequence, head_dim),
    ):
        return None
    if expanded_info != TensorInfo(
        source_info.elem_type,
        (batch, kv_heads, repeats, sequence, head_dim),
    ):
        return None
    return _TensorMatch(unsqueeze.input[0], (unsqueeze, expand))


def _valid_mask(
    index: GraphIndex,
    match: _MaskMatch,
    score_shape: tuple[int, int, int, int],
    query_info: TensorInfo,
) -> bool:
    if match.mask_name is None:
        return True
    info = index.tensor_info.get(match.mask_name)
    if info is None or not _broadcasts_to(info.shape, score_shape):
        return False
    if match.additive_sentinel is not None:
        return info.elem_type == query_info.elem_type
    return info.elem_type == TensorProto.BOOL


def _strict_additive_mask(index: GraphIndex, value_name: str) -> float | None:
    info = index.tensor_info.get(value_name)
    value = constant_array(index, value_name)
    if info is None or info.elem_type not in _SUPPORTED_DTYPES or value is None or value.size == 0:
        return None
    flattened = np.asarray(value).reshape(-1)
    scalars = {float(item) for item in flattened}
    negative = _negative_sentinel(info.elem_type)
    if scalars <= {0.0, -math.inf} and any(math.isinf(item) for item in scalars):
        return -math.inf
    if scalars <= {0.0, negative} and negative in scalars:
        return negative
    return None


def _replace_match(model: onnx.ModelProto, match: _AttentionMatch) -> str:
    graph = model.graph
    reserved_names = graph_names(model)
    node_name = unique_name(reserved_names, f"{match.output_name}_fused_attention")
    lse_name = unique_name(reserved_names, f"{match.output_name}_softmax_lse")
    mask_name = match.mask_name or ""
    preparation_nodes: list[NodeProto] = []

    if match.mask_name is not None and match.invert_mask:
        mask_name = unique_name(reserved_names, f"{match.mask_name}_blocked")
        preparation_nodes.append(
            helper.make_node(
                "Not",
                [match.mask_name],
                [mask_name],
                name=unique_name(reserved_names, f"{mask_name}_not"),
            )
        )
    elif match.mask_name is not None and match.additive_sentinel is not None:
        info = GraphIndex(model).tensor_info[match.mask_name]
        sentinel_name = unique_name(reserved_names, f"{match.mask_name}_sentinel")
        mask_name = unique_name(reserved_names, f"{match.mask_name}_blocked")
        graph.initializer.append(
            helper.make_tensor(
                sentinel_name,
                info.elem_type,
                [],
                [match.additive_sentinel],
            )
        )
        preparation_nodes.append(
            helper.make_node(
                "Equal",
                [match.mask_name, sentinel_name],
                [mask_name],
                name=unique_name(reserved_names, f"{mask_name}_equal"),
            )
        )

    inputs = [match.query_name, match.key_name, match.value_name, "", mask_name]
    inputs.extend([""] * (31 - len(inputs)))
    fused_node = helper.make_node(
        FUSED_INFER_ATTENTION_SCORE_OP,
        inputs,
        [match.output_name, lse_name],
        name=node_name,
        num_heads=match.num_heads,
        scale=match.scale,
        pre_tokens=_MAX_TOKENS,
        next_tokens=_MAX_TOKENS,
        input_layout="BNSD",
        num_key_value_heads=match.num_key_value_heads,
        sparse_mode=0,
        inner_precise=1,
        block_size=0,
        antiquant_mode=0,
        softmax_lse_flag=0,
        key_antiquant_mode=0,
        value_antiquant_mode=0,
        query_quant_mode=0,
        pse_type=0,
        out_dtype=0,
    )

    removed_ids = {id(node) for node in match.nodes}
    output_node_index = next(
        index
        for index, node in enumerate(graph.node)
        if id(node) in removed_ids and match.output_name in node.output
    )
    replacement_index = sum(
        id(node) not in removed_ids for node in graph.node[:output_node_index]
    )
    kept_nodes = [node for node in graph.node if id(node) not in removed_ids]
    kept_nodes[replacement_index:replacement_index] = [*preparation_nodes, fused_node]
    del graph.node[:]
    graph.node.extend(kept_nodes)

    stale_values = {
        output
        for node in match.nodes
        for output in node.output
        if output != match.output_name
    }
    remove_value_info(model, stale_values)
    _remove_dead_constants(model, match.nodes)
    return node_name


def _remove_dead_constants(
    model: onnx.ModelProto,
    removed_nodes: tuple[NodeProto, ...],
) -> None:
    candidates = {name for node in removed_nodes for name in node.input if name}
    index = GraphIndex(model)
    graph_outputs = {value.name for value in model.graph.output}
    dead = {
        name
        for name in candidates
        if name in index.initializers and not index.users(name) and name not in graph_outputs
    }
    kept = [tensor for tensor in model.graph.initializer if tensor.name not in dead]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def _closed_match(
    index: GraphIndex,
    graph_outputs: set[str],
    nodes: tuple[NodeProto, ...],
    preserved_output: str,
) -> bool:
    matched_ids = {id(node) for node in nodes}
    for node in nodes:
        for output in node.output:
            if output == preserved_output:
                continue
            if output in graph_outputs:
                return False
            if any(id(user) not in matched_ids for user in index.users(output)):
                return False
    return True


def _supported_bnsd(info: TensorInfo | None) -> bool:
    return (
        info is not None
        and info.elem_type in _SUPPORTED_DTYPES
        and len(info.shape) == 4
    )


def _static_shape(
    shape: tuple[int | str | None, ...],
    *,
    rank: int,
) -> tuple[int, ...] | None:
    if len(shape) != rank or any(
        not isinstance(dimension, int) or dimension <= 0 for dimension in shape
    ):
        return None
    return cast(tuple[int, ...], shape)


def _score_dtype(index: GraphIndex, score_name: str) -> int | None:
    info = index.tensor_info.get(score_name)
    return None if info is None else info.elem_type


def _negative_sentinel(elem_type: int) -> float:
    if elem_type == TensorProto.FLOAT16:
        return float(np.finfo(np.float16).min)
    if elem_type == TensorProto.BFLOAT16:
        return _BFLOAT16_MIN
    return math.nan


def _is_negative_sentinel(value: float | None, elem_type: int | None) -> bool:
    if value is None or elem_type not in _SUPPORTED_DTYPES:
        return False
    return value == -math.inf or value == _negative_sentinel(elem_type)


def _broadcasts_to(
    source: tuple[int | str | None, ...],
    target: tuple[int, int, int, int],
) -> bool:
    if len(source) > len(target):
        return False
    padded = (1,) * (len(target) - len(source)) + source
    return all(
        value == 1 or value == expected
        for value, expected in zip(padded, target, strict=True)
    )


def _positive_scalar(index: GraphIndex, value_name: str) -> float | None:
    value = _scalar(index, value_name)
    return value if value is not None and math.isfinite(value) and value > 0.0 else None


def _scalar(index: GraphIndex, value_name: str) -> float | None:
    value = constant_array(index, value_name)
    if value is None or value.size != 1:
        return None
    return float(np.asarray(value).reshape(()))


def _is_softmax(node: NodeProto | None) -> TypeGuard[NodeProto]:
    if not _is_node(node, "Softmax", inputs=1, outputs=1):
        return False
    try:
        return attribute_int(node, "axis", -1) == -1
    except ValueError:
        return False


def _unique_nodes(nodes: tuple[NodeProto | None, ...]) -> tuple[NodeProto, ...]:
    result: list[NodeProto] = []
    seen: set[int] = set()
    for node in nodes:
        if node is not None and id(node) not in seen:
            seen.add(id(node))
            result.append(node)
    return tuple(result)


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


FUSED_INFER_ATTENTION_SCORE_FUSION_PASS: Final = FusedInferAttentionScoreFusionPass()

__all__ = [
    "FUSED_INFER_ATTENTION_SCORE_FUSION_PASS",
    "FusedInferAttentionScoreFusionPass",
    "fuse_fused_infer_attention_score",
]
