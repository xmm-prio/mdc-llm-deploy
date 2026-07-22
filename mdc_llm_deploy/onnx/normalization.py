"""Canonicalize semantically transparent ONNX graph constructs."""

from __future__ import annotations

from typing import TypeGuard

import numpy as np
import onnx
from onnx import GraphProto, NodeProto, TensorProto, ValueInfoProto, helper, numpy_helper

from ._graph import (
    GraphIndex,
    attribute_int,
    clone_model,
    constant_array,
    remove_value_info,
)

_LOSSLESS_FLOAT_WIDENINGS = frozenset(
    {
        (TensorProto.FLOAT16, TensorProto.FLOAT),
        (TensorProto.BFLOAT16, TensorProto.FLOAT),
        (TensorProto.FLOAT, TensorProto.DOUBLE),
    }
)
_FOLDABLE_CAST_DTYPES = frozenset(
    {
        TensorProto.BOOL,
        TensorProto.INT8,
        TensorProto.INT16,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.UINT8,
        TensorProto.UINT16,
        TensorProto.UINT32,
        TensorProto.UINT64,
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
    }
)


def _is_removable_identity(node: NodeProto, graph_outputs: set[str]) -> bool:
    return (
        node.domain in ("", "ai.onnx")
        and node.op_type == "Identity"
        and len(node.input) == 1
        and len(node.output) == 1
        and bool(node.input[0])
        and bool(node.output[0])
        and node.input[0] != node.output[0]
        and node.output[0] not in graph_outputs
    )


def _copy_missing_value_info(
    graph: GraphProto,
    source_name: str,
    output_info: ValueInfoProto | None,
) -> None:
    if output_info is None:
        return
    known_names = {value.name for value in (*graph.input, *graph.value_info, *graph.output)}
    known_names.update(tensor.name for tensor in graph.initializer)
    if source_name in known_names:
        return
    source_info = ValueInfoProto()
    source_info.CopyFrom(output_info)
    source_info.name = source_name
    graph.value_info.append(source_info)


def _resolve_alias(name: str, aliases: dict[str, str]) -> str:
    visited: set[str] = set()
    while name in aliases and name not in visited:
        visited.add(name)
        name = aliases[name]
    return name


def _apply_aliases(
    model: onnx.ModelProto,
    aliases: dict[str, str],
    removed_outputs: set[str],
) -> None:
    graph = model.graph
    output_info = {value.name: value for value in graph.value_info}
    for output_name, source_name in aliases.items():
        _copy_missing_value_info(
            graph,
            _resolve_alias(source_name, aliases),
            output_info.get(output_name),
        )

    for node in graph.node:
        if any(output_name in removed_outputs for output_name in node.output):
            continue
        for index, input_name in enumerate(node.input):
            if input_name in aliases:
                node.input[index] = _resolve_alias(input_name, aliases)

    kept = [
        node
        for node in graph.node
        if not any(output_name in removed_outputs for output_name in node.output)
    ]
    del graph.node[:]
    graph.node.extend(kept)
    remove_value_info(model, removed_outputs)


def _eliminate_identities(model: onnx.ModelProto) -> bool:
    graph = model.graph
    graph_outputs = {value.name for value in graph.output}
    removable = [node for node in graph.node if _is_removable_identity(node, graph_outputs)]
    if not removable:
        return False

    aliases = {node.output[0]: node.input[0] for node in removable}
    _apply_aliases(model, aliases, set(aliases))
    return True


def _is_cast(node: NodeProto | None) -> TypeGuard[NodeProto]:
    return (
        node is not None
        and node.domain in ("", "ai.onnx")
        and node.op_type == "Cast"
        and len(node.input) == 1
        and len(node.output) == 1
        and bool(node.input[0])
        and bool(node.output[0])
    )


def _fold_constant_expressions(model: onnx.ModelProto) -> bool:
    changed = False
    source_candidates: set[str] = set()
    while True:
        index = GraphIndex(model)
        replacements: dict[str, onnx.TensorProto] = {}
        for node in model.graph.node:
            folded = _fold_constant_node(index, node)
            if folded is not None:
                replacements[node.output[0]] = numpy_helper.from_array(
                    folded,
                    node.output[0],
                )
                source_candidates.update(name for name in node.input if name)

        if not replacements:
            break
        kept = [
            node
            for node in model.graph.node
            if not any(output_name in replacements for output_name in node.output)
        ]
        del model.graph.node[:]
        model.graph.node.extend(kept)
        model.graph.initializer.extend(replacements.values())
        changed = True

    if changed:
        used = {name for node in model.graph.node for name in node.input if name}
        used.update(value.name for value in (*model.graph.input, *model.graph.output))
        kept = [
            tensor
            for tensor in model.graph.initializer
            if tensor.name not in source_candidates or tensor.name in used
        ]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        kept_nodes = [
            node
            for node in model.graph.node
            if not (
                node.domain in ("", "ai.onnx")
                and node.op_type == "Constant"
                and len(node.output) == 1
                and node.output[0] in source_candidates
                and node.output[0] not in used
            )
        ]
        del model.graph.node[:]
        model.graph.node.extend(kept_nodes)
    return changed


def _fold_constant_node(index: GraphIndex, node: NodeProto) -> np.ndarray | None:
    if _is_cast(node):
        try:
            target_type = attribute_int(node, "to")
        except ValueError:
            return None
        if target_type not in _FOLDABLE_CAST_DTYPES:
            return None
        value = constant_array(index, node.input[0])
        if value is None:
            return None
        try:
            target_dtype = helper.tensor_dtype_to_np_dtype(target_type)
            return np.asarray(value).astype(target_dtype)
        except (TypeError, ValueError):
            return None

    if not (
        node.domain in ("", "ai.onnx")
        and node.op_type == "Reshape"
        and len(node.input) == 2
        and len(node.output) == 1
        and all(node.input)
        and node.output[0]
    ):
        return None
    value = constant_array(index, node.input[0])
    shape = constant_array(index, node.input[1])
    if value is None or shape is None or not np.issubdtype(shape.dtype, np.integer):
        return None
    try:
        allowzero = attribute_int(node, "allowzero", 0)
    except ValueError:
        return None
    target = [int(dimension) for dimension in np.asarray(shape).reshape(-1)]
    if any(dimension < -1 for dimension in target) or target.count(-1) > 1:
        return None
    if allowzero == 0:
        if any(dimension == 0 and index >= value.ndim for index, dimension in enumerate(target)):
            return None
        target = [
            value.shape[index] if dimension == 0 else dimension
            for index, dimension in enumerate(target)
        ]
    elif allowzero != 1:
        return None
    try:
        return np.asarray(value).reshape(target)
    except ValueError:
        return None


def _eliminate_lossless_cast_round_trips(model: onnx.ModelProto) -> bool:
    graph = model.graph
    index = GraphIndex(model)
    graph_outputs = {value.name for value in graph.output}
    aliases: dict[str, str] = {}
    removed_outputs: set[str] = set()

    for output_cast in graph.node:
        if not _is_cast(output_cast) or output_cast.output[0] in graph_outputs:
            continue
        input_cast = index.producer(output_cast.input[0])
        if not _is_cast(input_cast):
            continue
        source_info = index.tensor_info.get(input_cast.input[0])
        widened_info = index.tensor_info.get(input_cast.output[0])
        output_info = index.tensor_info.get(output_cast.output[0])
        if source_info is None or widened_info is None or output_info is None:
            continue
        try:
            input_target = attribute_int(input_cast, "to")
            output_target = attribute_int(output_cast, "to")
        except ValueError:
            continue
        if (
            source_info != output_info
            or widened_info.shape != source_info.shape
            or input_target != widened_info.elem_type
            or output_target != source_info.elem_type
            or (source_info.elem_type, widened_info.elem_type) not in _LOSSLESS_FLOAT_WIDENINGS
        ):
            continue
        aliases[output_cast.output[0]] = input_cast.input[0]
        removed_outputs.add(output_cast.output[0])
        if (
            input_cast.output[0] not in graph_outputs
            and len(index.users(input_cast.output[0])) == 1
        ):
            removed_outputs.add(input_cast.output[0])

    if not aliases:
        return False
    _apply_aliases(model, aliases, removed_outputs)
    return True


def normalize_graph_core(model: onnx.ModelProto) -> onnx.ModelProto:
    """Canonicalize transparent main-graph nodes in place."""
    _eliminate_identities(model)
    _fold_constant_expressions(model)
    _eliminate_lossless_cast_round_trips(model)
    return model


def normalize_graph(model: onnx.ModelProto) -> onnx.ModelProto:
    """Atomically canonicalize transparent main-graph nodes."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an onnx.ModelProto")
    working = clone_model(model)
    normalize_graph_core(working)
    onnx.checker.check_model(working)
    model.CopyFrom(working)
    return model


__all__ = ["normalize_graph"]
