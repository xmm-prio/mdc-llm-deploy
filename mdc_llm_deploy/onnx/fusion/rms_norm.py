"""Fuse proven FP32-accumulation RMSNorm subgraphs into NPURmsNorm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, TypeGuard

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper, numpy_helper

from .._graph import (
    GraphIndex,
    TensorInfo,
    attribute_int,
    constant_array,
    graph_names,
    remove_value_info,
    unique_name,
)
from ..schemas import RMS_NORM_OP
from .base import FusionPassResult

_PASS_NAME: Final = "rms_norm"
_SUPPORTED_DTYPES: Final = frozenset(
    {TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.FLOAT}
)


@dataclass(frozen=True, slots=True)
class _RmsNormMatch:
    x_name: str
    gamma_name: str
    y_name: str
    reciprocal_name: str
    epsilon: float
    normalized_axes: tuple[int, ...]
    nodes: tuple[NodeProto, ...]
    constant_names: tuple[str, ...]
    preserve_reciprocal: bool


class RmsNormFusionPass:
    """Fuse strict, statically proven RMSNorm decompositions."""

    @property
    def name(self) -> str:
        """Return the stable pass name."""
        return _PASS_NAME

    def apply(self, model: onnx.ModelProto) -> FusionPassResult:
        """Fuse all non-overlapping supported RMSNorm subgraphs in place."""
        if not isinstance(model, onnx.ModelProto):
            raise TypeError("model must be an onnx.ModelProto")

        fused_node_names: list[str] = []
        while match := _find_next_match(model):
            fused_node_names.append(_replace_match(model, match))
        return FusionPassResult(self.name, len(fused_node_names), tuple(fused_node_names))


def fuse_rms_norm(model: onnx.ModelProto) -> FusionPassResult:
    """Run the RMSNorm fusion pass in place."""
    return RMS_NORM_FUSION_PASS.apply(model)


def _find_next_match(model: onnx.ModelProto) -> _RmsNormMatch | None:
    index = GraphIndex(model)
    graph_outputs = {value.name for value in model.graph.output}
    for node in model.graph.node:
        match = _match_from_output_mul(index, graph_outputs, node)
        if match is not None:
            return match
    return None


def _match_from_output_mul(
    index: GraphIndex,
    graph_outputs: set[str],
    output_mul: NodeProto,
) -> _RmsNormMatch | None:
    if not _is_node(output_mul, "Mul", inputs=2, outputs=1):
        return None

    for gamma_name, normalized_name in (
        (output_mul.input[0], output_mul.input[1]),
        (output_mul.input[1], output_mul.input[0]),
    ):
        gamma_info = index.tensor_info.get(gamma_name)
        if gamma_name not in index.initializers or gamma_info is None:
            continue
        match = _match_normalization_path(
            index,
            graph_outputs,
            output_mul,
            gamma_name,
            normalized_name,
        )
        if match is not None:
            return match
    return None


def _match_normalization_path(
    index: GraphIndex,
    graph_outputs: set[str],
    output_mul: NodeProto,
    gamma_name: str,
    normalized_name: str,
) -> _RmsNormMatch | None:
    gamma_info = index.tensor_info[gamma_name]
    if gamma_info.elem_type not in _SUPPORTED_DTYPES or not _static_nonempty(gamma_info.shape):
        return None

    output_cast: NodeProto | None = None
    norm_mul = index.producer(normalized_name)
    if gamma_info.elem_type != TensorProto.FLOAT:
        output_cast = norm_mul
        if output_cast is None or not _is_cast_to(output_cast, gamma_info.elem_type):
            return None
        norm_mul = index.producer(output_cast.input[0])
    if not _is_node(norm_mul, "Mul", inputs=2, outputs=1):
        return None

    for accumulator_name, reciprocal_name in (
        (norm_mul.input[0], norm_mul.input[1]),
        (norm_mul.input[1], norm_mul.input[0]),
    ):
        matched = _match_statistics_path(
            index,
            accumulator_name,
            reciprocal_name,
            gamma_name,
            output_mul,
            norm_mul,
            output_cast,
            graph_outputs,
        )
        if matched is not None:
            return matched
    return None


def _match_statistics_path(
    index: GraphIndex,
    accumulator_name: str,
    reciprocal_name: str,
    gamma_name: str,
    output_mul: NodeProto,
    norm_mul: NodeProto,
    output_cast: NodeProto | None,
    graph_outputs: set[str],
) -> _RmsNormMatch | None:
    reciprocal = index.producer(reciprocal_name)
    if reciprocal is None:
        return None
    sqrt = _single_input_producer(index, reciprocal, "Reciprocal")
    if sqrt is None:
        return None
    add = _single_input_producer(index, sqrt, "Sqrt")
    if not _is_node(add, "Add", inputs=2, outputs=1):
        return None

    mean, epsilon_name = _node_and_other_input(index, add, "ReduceMean")
    if mean is None or epsilon_name is None:
        return None
    power = index.producer(mean.input[0]) if len(mean.input) == 2 else None
    if not _is_node(power, "Pow", inputs=2, outputs=1):
        return None
    if power.input[0] != accumulator_name or not _constant_equals(index, power.input[1], 2.0):
        return None

    input_cast = index.producer(accumulator_name)
    if input_cast is not None and _is_cast_to(input_cast, TensorProto.FLOAT):
        x_name = input_cast.input[0]
    else:
        input_cast = None
        x_name = accumulator_name

    gamma_info = index.tensor_info[gamma_name]
    x_info = index.tensor_info.get(x_name)
    if not _valid_tensor_contract(x_info, gamma_info):
        return None
    if x_info is None:
        return None
    if x_info.elem_type == TensorProto.FLOAT:
        if input_cast is not None or output_cast is not None:
            return None
    elif input_cast is None or output_cast is None:
        return None

    normalized_axes = tuple(range(len(x_info.shape) - len(gamma_info.shape), len(x_info.shape)))
    if not _valid_reduce_mean(index, mean, normalized_axes):
        return None
    epsilon = _positive_scalar(index, epsilon_name)
    if epsilon is None:
        return None

    nodes = tuple(
        node
        for node in (
            input_cast,
            power,
            mean,
            add,
            sqrt,
            reciprocal,
            norm_mul,
            output_cast,
            output_mul,
        )
        if node is not None
    )
    if not _closed_match(index, graph_outputs, nodes, reciprocal, norm_mul, output_mul):
        return None

    constant_names = (power.input[1], mean.input[1], epsilon_name)
    external_reciprocal_use = any(
        user is not norm_mul for user in index.users(reciprocal.output[0])
    )
    preserve_reciprocal = (
        external_reciprocal_use or reciprocal.output[0] in graph_outputs
    )
    return _RmsNormMatch(
        x_name=x_name,
        gamma_name=gamma_name,
        y_name=output_mul.output[0],
        reciprocal_name=reciprocal.output[0],
        epsilon=epsilon,
        normalized_axes=normalized_axes,
        nodes=nodes,
        constant_names=constant_names,
        preserve_reciprocal=preserve_reciprocal,
    )


def _replace_match(model: onnx.ModelProto, match: _RmsNormMatch) -> str:
    graph = model.graph
    reserved_names = graph_names(model)
    node_name = unique_name(reserved_names, f"{match.y_name}_npu_rms_norm")
    raw_rstd_name = unique_name(reserved_names, f"{match.y_name}_rstd")
    fused_node = helper.make_node(
        RMS_NORM_OP,
        [match.x_name, match.gamma_name],
        [match.y_name, raw_rstd_name],
        name=node_name,
        epsilon=match.epsilon,
    )

    replacement_nodes = [fused_node]
    if match.preserve_reciprocal:
        axes_name = unique_name(reserved_names, f"{raw_rstd_name}_axes")
        axes = numpy_helper.from_array(
            np.asarray(match.normalized_axes, dtype=np.int64),
            name=axes_name,
        )
        graph.initializer.append(axes)
        replacement_nodes.append(
            helper.make_node(
                "Unsqueeze",
                [raw_rstd_name, axes_name],
                [match.reciprocal_name],
                name=unique_name(reserved_names, f"{raw_rstd_name}_keepdims"),
            )
        )

    removed_ids = {id(node) for node in match.nodes}
    replacement_index = min(
        index for index, node in enumerate(graph.node) if id(node) in removed_ids
    )
    kept_nodes = [node for node in graph.node if id(node) not in removed_ids]
    kept_nodes[replacement_index:replacement_index] = replacement_nodes
    del graph.node[:]
    graph.node.extend(kept_nodes)

    stale_values = {
        output
        for node in match.nodes
        for output in node.output
        if output not in {match.y_name, match.reciprocal_name}
    }
    if not match.preserve_reciprocal:
        stale_values.add(match.reciprocal_name)
    remove_value_info(model, stale_values)
    _remove_dead_constants(model, match.constant_names)
    return node_name


def _remove_dead_constants(model: onnx.ModelProto, candidate_names: tuple[str, ...]) -> None:
    index = GraphIndex(model)
    graph_outputs = {value.name for value in model.graph.output}
    dead_names = {
        name for name in candidate_names if not index.users(name) and name not in graph_outputs
    }
    if not dead_names:
        return

    kept_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name not in dead_names
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


def _valid_tensor_contract(
    x_info: TensorInfo | None,
    gamma_info: TensorInfo,
) -> bool:
    if x_info is None:
        return False
    if x_info.elem_type != gamma_info.elem_type or x_info.elem_type not in _SUPPORTED_DTYPES:
        return False
    if not 1 <= len(x_info.shape) <= 8 or not 1 <= len(gamma_info.shape) <= len(x_info.shape):
        return False
    if not _static_nonempty(x_info.shape) or not _static_nonempty(gamma_info.shape):
        return False
    return x_info.shape[-len(gamma_info.shape) :] == gamma_info.shape


def _static_nonempty(shape: tuple[int | str | None, ...]) -> bool:
    return bool(shape) and all(isinstance(dimension, int) and dimension > 0 for dimension in shape)


def _valid_reduce_mean(
    index: GraphIndex,
    node: NodeProto,
    normalized_axes: tuple[int, ...],
) -> bool:
    if not _is_node(node, "ReduceMean", inputs=2, outputs=1):
        return False
    try:
        keepdims = attribute_int(node, "keepdims", 1)
        noop_with_empty_axes = attribute_int(node, "noop_with_empty_axes", 0)
    except ValueError:
        return False
    if keepdims != 1 or noop_with_empty_axes != 0:
        return False
    axes = constant_array(index, node.input[1])
    if axes is None:
        return False
    actual = tuple(int(axis) for axis in np.asarray(axes).reshape(-1))
    rank = normalized_axes[-1] + 1
    canonical = tuple(axis + rank if axis < 0 else axis for axis in actual)
    return canonical == normalized_axes


def _positive_scalar(index: GraphIndex, value_name: str) -> float | None:
    value = constant_array(index, value_name)
    if value is None or np.asarray(value).size != 1:
        return None
    scalar = float(np.asarray(value).reshape(()))
    return scalar if math.isfinite(scalar) and scalar > 0.0 else None


def _constant_equals(index: GraphIndex, value_name: str, expected: float) -> bool:
    value = constant_array(index, value_name)
    return bool(
        value is not None
        and np.asarray(value).size == 1
        and float(np.asarray(value).reshape(())) == expected
    )


def _closed_match(
    index: GraphIndex,
    graph_outputs: set[str],
    nodes: tuple[NodeProto, ...],
    reciprocal: NodeProto,
    norm_mul: NodeProto,
    output_mul: NodeProto,
) -> bool:
    matched_ids = {id(node) for node in nodes}
    for node in nodes:
        for output in node.output:
            if node is output_mul:
                continue
            if output in graph_outputs and node is not reciprocal:
                return False
            external_users = [
                user for user in index.users(output) if id(user) not in matched_ids
            ]
            if external_users and node is not reciprocal:
                return False
    return norm_mul in index.users(reciprocal.output[0])


def _single_input_producer(
    index: GraphIndex,
    node: NodeProto | None,
    op_type: str,
) -> NodeProto | None:
    if not _is_node(node, op_type, inputs=1, outputs=1):
        return None
    return index.producer(node.input[0])


def _node_and_other_input(
    index: GraphIndex,
    node: NodeProto,
    op_type: str,
) -> tuple[NodeProto | None, str | None]:
    for candidate_name, other_name in (
        (node.input[0], node.input[1]),
        (node.input[1], node.input[0]),
    ):
        candidate = index.producer(candidate_name)
        if candidate is not None and candidate.op_type == op_type:
            return candidate, other_name
    return None, None


def _is_cast_to(node: NodeProto | None, elem_type: int) -> bool:
    if not _is_node(node, "Cast", inputs=1, outputs=1):
        return False
    try:
        return attribute_int(node, "to") == elem_type
    except ValueError:
        return False


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


RMS_NORM_FUSION_PASS: Final = RmsNormFusionPass()

__all__ = ["RMS_NORM_FUSION_PASS", "RmsNormFusionPass", "fuse_rms_norm"]
