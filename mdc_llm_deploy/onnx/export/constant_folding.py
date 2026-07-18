"""Safely fold a bounded whitelist of constant ONNX subgraphs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, NodeProto, TensorProto, helper, numpy_helper

from ...errors import OnnxExportError
from ...operators.contracts.onnx import MDC_ONNX_DOMAIN

_STANDARD_DOMAINS = {"", MDC_ONNX_DOMAIN}
_SUPPORTED_DTYPES = {
    int(TensorProto.BOOL),
    int(TensorProto.INT8),
    int(TensorProto.INT16),
    int(TensorProto.INT32),
    int(TensorProto.INT64),
    int(TensorProto.UINT8),
    int(TensorProto.UINT16),
    int(TensorProto.UINT32),
    int(TensorProto.UINT64),
    int(TensorProto.FLOAT16),
    int(TensorProto.FLOAT),
    int(TensorProto.DOUBLE),
    int(TensorProto.BFLOAT16),
}
_CONTROL_DTYPES = {int(TensorProto.INT64)}


@dataclass(frozen=True)
class ConstantFoldingStats:
    """Immutable summary returned after a constant-folding pass."""

    folded_nodes: int
    materialized_initializers: int
    materialized_bytes: int
    skipped_nodes: int
    skipped_by_reason: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _FoldingBudget:
    max_output_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 512 * 1024 * 1024
    max_rank: int = 8
    max_nodes: int = 4096
    max_expansion: int = 64


_DEFAULT_BUDGET = _FoldingBudget()


@dataclass(frozen=True)
class _Constant:
    array: np.ndarray[Any, Any]
    data_type: int

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes)


@dataclass(frozen=True)
class _Evaluation:
    shape: tuple[int, ...]
    data_type: int
    evaluate: Callable[[], np.ndarray[Any, Any]]


@dataclass(frozen=True)
class _Fold:
    node_index: int
    output_name: str
    initializer: onnx.TensorProto
    nbytes: int


@dataclass(frozen=True)
class _Plan:
    folds: tuple[_Fold, ...]
    skipped: tuple[tuple[str, int], ...]

    @property
    def stats(self) -> ConstantFoldingStats:
        total_bytes = sum(fold.nbytes for fold in self.folds)
        return ConstantFoldingStats(
            folded_nodes=len(self.folds),
            materialized_initializers=len(self.folds),
            materialized_bytes=total_bytes,
            skipped_nodes=sum(count for _, count in self.skipped),
            skipped_by_reason=self.skipped,
        )

    def apply(self, model: onnx.ModelProto) -> None:
        """Apply the fully validated plan in one protobuf mutation phase."""
        folded_indices = {fold.node_index for fold in self.folds}
        retained_nodes = [
            node
            for node_index, node in enumerate(model.graph.node)
            if node_index not in folded_indices
        ]
        del model.graph.node[:]
        model.graph.node.extend(retained_nodes)
        model.graph.initializer.extend(fold.initializer for fold in self.folds)


def fold_constant_subgraphs(model: onnx.ModelProto) -> ConstantFoldingStats:
    """Fold safe constant subgraphs in place and return immutable pass statistics.

    Planning and evaluation finish before the protobuf is changed. Invalid
    constant nodes therefore raise :class:`OnnxExportError` without partial
    graph mutation. Resource-limit rejections remain unchanged in the graph.
    """
    plan = _build_plan(model, _DEFAULT_BUDGET)
    plan.apply(model)
    return plan.stats


def _build_plan(model: onnx.ModelProto, budget: _FoldingBudget) -> _Plan:
    _validate_budget(budget)
    opset = _standard_opset(model)
    graph_inputs = {value.name for value in model.graph.input}
    constants = _load_initializers(model, graph_inputs)
    known_initializers = {initializer.name for initializer in model.graph.initializer}
    _validate_output_names(model, known_initializers)

    pending = [
        index
        for index, node in enumerate(model.graph.node)
        if node.op_type in _EVALUATORS and node.domain in _STANDARD_DOMAINS
    ]
    blocked: set[int] = set()
    folds: list[_Fold] = []
    skipped: Counter[str] = Counter()
    total_bytes = 0

    while True:
        progressed = False
        for node_index in pending:
            if node_index in blocked or any(fold.node_index == node_index for fold in folds):
                continue
            node = model.graph.node[node_index]
            required_inputs = tuple(name for name in node.input if name)
            if not all(name in constants for name in required_inputs):
                continue
            if len(folds) >= budget.max_nodes:
                blocked.add(node_index)
                skipped["node_limit"] += 1
                continue
            inputs = tuple(constants[name] for name in node.input if name)
            evaluation = _evaluate_node(node, inputs, opset)
            reason = _budget_rejection(evaluation, inputs, total_bytes, budget)
            if reason is not None:
                blocked.add(node_index)
                skipped[reason] += 1
                continue
            array = _run_evaluation(node, evaluation)
            expected_bytes = _tensor_nbytes(evaluation.shape, evaluation.data_type)
            if tuple(int(dim) for dim in array.shape) != evaluation.shape:
                raise _node_error(node, "evaluator produced an unexpected shape")
            if int(array.nbytes) != expected_bytes:
                raise _node_error(node, "evaluator produced an unexpected dtype")
            output_name = node.output[0]
            initializer = _make_initializer(output_name, array, evaluation.data_type, node)
            constants[output_name] = _Constant(array, evaluation.data_type)
            folds.append(_Fold(node_index, output_name, initializer, expected_bytes))
            total_bytes += expected_bytes
            progressed = True
        if not progressed:
            break

    return _Plan(
        folds=tuple(folds),
        skipped=tuple(sorted(skipped.items())),
    )


def _validate_budget(budget: _FoldingBudget) -> None:
    values = (
        budget.max_output_bytes,
        budget.max_total_bytes,
        budget.max_rank,
        budget.max_nodes,
        budget.max_expansion,
    )
    if any(value < 0 for value in values):
        raise ValueError("constant-folding budget values must be non-negative")


def _standard_opset(model: onnx.ModelProto) -> int:
    versions = [
        int(item.version)
        for item in model.opset_import
        if item.domain in _STANDARD_DOMAINS
    ]
    if not versions:
        raise OnnxExportError("ONNX model has no standard-domain opset import")
    if len(set(versions)) != 1:
        raise OnnxExportError("ONNX model has conflicting standard-domain opset imports")
    return versions[0]


def _load_initializers(
    model: onnx.ModelProto,
    graph_inputs: set[str],
) -> dict[str, _Constant]:
    constants: dict[str, _Constant] = {}
    names: set[str] = set()
    for initializer in model.graph.initializer:
        if not initializer.name:
            raise OnnxExportError("ONNX initializer has an empty name")
        if initializer.name in names:
            raise OnnxExportError(f"Duplicate ONNX initializer {initializer.name!r}")
        names.add(initializer.name)
        if (
            initializer.name in graph_inputs
            or int(initializer.data_type) not in _SUPPORTED_DTYPES
        ):
            continue
        try:
            array = np.asarray(numpy_helper.to_array(initializer))
        except Exception as error:
            raise OnnxExportError(
                f"Cannot read ONNX initializer {initializer.name!r}: {error}"
            ) from error
        constants[initializer.name] = _Constant(
            array=array,
            data_type=int(initializer.data_type),
        )
    return constants


def _validate_output_names(
    model: onnx.ModelProto,
    initializer_names: set[str],
) -> None:
    produced: set[str] = set()
    for node in model.graph.node:
        for output_name in node.output:
            if not output_name:
                continue
            if output_name in produced or output_name in initializer_names:
                raise _node_error(node, f"output name {output_name!r} is not unique")
            produced.add(output_name)


def _evaluate_node(
    node: NodeProto,
    inputs: tuple[_Constant, ...],
    opset: int,
) -> _Evaluation:
    try:
        return _EVALUATORS[node.op_type](node, inputs, opset)
    except OnnxExportError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise _node_error(node, str(error)) from error


def _run_evaluation(
    node: NodeProto,
    evaluation: _Evaluation,
) -> np.ndarray[Any, Any]:
    try:
        with np.errstate(all="ignore"):
            return np.ascontiguousarray(evaluation.evaluate())
    except (ArithmeticError, IndexError, TypeError, ValueError) as error:
        raise _node_error(node, f"evaluation failed: {error}") from error


def _budget_rejection(
    evaluation: _Evaluation,
    inputs: Sequence[_Constant],
    total_bytes: int,
    budget: _FoldingBudget,
) -> str | None:
    if len(evaluation.shape) > budget.max_rank:
        return "rank_limit"
    output_bytes = _tensor_nbytes(evaluation.shape, evaluation.data_type)
    if output_bytes > budget.max_output_bytes:
        return "output_bytes_limit"
    if total_bytes + output_bytes > budget.max_total_bytes:
        return "total_bytes_limit"
    input_bytes = max((value.nbytes for value in inputs), default=0)
    if output_bytes > budget.max_expansion * max(input_bytes, 1):
        return "expansion_limit"
    return None


def _make_initializer(
    name: str,
    array: np.ndarray[Any, Any],
    data_type: int,
    node: NodeProto,
) -> onnx.TensorProto:
    try:
        initializer = numpy_helper.from_array(array, name=name)
    except Exception as error:
        raise _node_error(node, f"cannot serialize result: {error}") from error
    if int(initializer.data_type) != data_type:
        raise _node_error(node, "serialized result has an unexpected dtype")
    return initializer


def _node_error(node: NodeProto, reason: str) -> OnnxExportError:
    identity = node.name or (node.output[0] if node.output else "<unnamed>")
    return OnnxExportError(
        f"Invalid constant ONNX node {identity!r} ({node.op_type}): {reason}"
    )


def _attributes(
    node: NodeProto,
    allowed: Mapping[str, int],
) -> dict[str, AttributeProto]:
    result: dict[str, AttributeProto] = {}
    for attribute in node.attribute:
        expected_type = allowed.get(attribute.name)
        if expected_type is None:
            raise _node_error(node, f"unsupported attribute {attribute.name!r}")
        if attribute.name in result:
            raise _node_error(node, f"duplicate attribute {attribute.name!r}")
        if int(attribute.type) != expected_type:
            raise _node_error(node, f"attribute {attribute.name!r} has invalid type")
        result[attribute.name] = attribute
    return result


def _require_arity(
    node: NodeProto,
    inputs: Sequence[_Constant],
    counts: set[int],
    *,
    outputs: int = 1,
) -> None:
    if len(inputs) not in counts or len(node.output) != outputs or not all(node.output):
        expected = "/".join(str(count) for count in sorted(counts))
        raise _node_error(node, f"expected {expected} inputs and {outputs} output")


def _same_dtype(node: NodeProto, inputs: Sequence[_Constant]) -> int:
    data_types = {value.data_type for value in inputs}
    if len(data_types) != 1:
        raise _node_error(node, "input dtypes must match")
    return next(iter(data_types))


def _axis(node: NodeProto, axis: int, rank: int, *, insertion: bool = False) -> int:
    upper = rank + (1 if insertion else 0)
    lower = -upper
    if axis < lower or axis >= upper:
        raise _node_error(node, f"axis {axis} is out of range for rank {rank}")
    return axis + upper if axis < 0 else axis


def _int_values(node: NodeProto, value: _Constant, role: str) -> tuple[int, ...]:
    if value.data_type not in _CONTROL_DTYPES or value.array.ndim != 1:
        raise _node_error(node, f"{role} must be a rank-1 integer tensor")
    return tuple(int(item) for item in value.array.tolist())


def _tensor_nbytes(shape: Sequence[int], data_type: int) -> int:
    try:
        itemsize = int(np.dtype(helper.tensor_dtype_to_np_dtype(data_type)).itemsize)
    except (TypeError, ValueError) as error:
        raise OnnxExportError(f"Unsupported ONNX tensor dtype {data_type}") from error
    return math.prod(shape) * itemsize


def _identity(
    node: NodeProto,
    inputs: tuple[_Constant, ...],
    opset: int,
) -> _Evaluation:
    del opset
    _require_arity(node, inputs, {1})
    _attributes(node, {})
    value = inputs[0]
    return _Evaluation(value.array.shape, value.data_type, lambda: value.array.copy())


def _cast(
    node: NodeProto,
    inputs: tuple[_Constant, ...],
    opset: int,
) -> _Evaluation:
    _require_arity(node, inputs, {1})
    allowed_attributes = {"to": AttributeProto.INT}
    if opset >= 19:
        allowed_attributes["saturate"] = AttributeProto.INT
    attributes = _attributes(
        node,
        allowed_attributes,
    )
    target = attributes.get("to")
    if target is None:
        raise _node_error(node, "required attribute 'to' is missing")
    data_type = int(target.i)
    if data_type not in _SUPPORTED_DTYPES:
        raise _node_error(node, f"Cast target dtype {data_type} is unsupported")
    saturate = int(attributes.get("saturate", helper.make_attribute("saturate", 1)).i)
    if saturate != 1:
        raise _node_error(node, "saturate=0 is unsupported")
    value = inputs[0]
    dtype = helper.tensor_dtype_to_np_dtype(data_type)
    return _Evaluation(
        value.array.shape,
        data_type,
        lambda: value.array.astype(dtype, copy=True),
    )


def _transpose(
    node: NodeProto,
    inputs: tuple[_Constant, ...],
    opset: int,
) -> _Evaluation:
    del opset
    _require_arity(node, inputs, {1})
    attributes = _attributes(node, {"perm": AttributeProto.INTS})
    value = inputs[0]
    permutation = tuple(attributes["perm"].ints) if "perm" in attributes else tuple(
        reversed(range(value.array.ndim))
    )
    if sorted(permutation) != list(range(value.array.ndim)):
        raise _node_error(node, "perm must be a permutation of input axes")
    shape = tuple(int(value.array.shape[index]) for index in permutation)
    return _Evaluation(
        shape,
        value.data_type,
        lambda: np.transpose(value.array, permutation),
    )


def _reshape(
    node: NodeProto,
    inputs: tuple[_Constant, ...],
    opset: int,
) -> _Evaluation:
    _require_arity(node, inputs, {2})
    allowed_attributes = {"allowzero": AttributeProto.INT} if opset >= 14 else {}
    attributes = _attributes(node, allowed_attributes)
    allowzero = int(attributes.get("allowzero", helper.make_attribute("allowzero", 0)).i)
    if allowzero not in {0, 1}:
        raise _node_error(node, "allowzero must be 0 or 1")
    value, shape_value = inputs
    requested = list(_int_values(node, shape_value, "shape"))
    if sum(item == -1 for item in requested) > 1 or any(item < -1 for item in requested):
        raise _node_error(node, "shape contains invalid dimensions")
    if allowzero and 0 in requested and -1 in requested:
        raise _node_error(node, "allowzero=1 cannot combine zero and -1 dimensions")
    resolved = [
        int(value.array.shape[index]) if item == 0 and not allowzero else item
        for index, item in enumerate(requested)
    ]
    if not allowzero and any(
        item == 0 and index >= value.array.ndim
        for index, item in enumerate(requested)
    ):
        raise _node_error(node, "zero dimension has no corresponding input dimension")
    input_size = int(value.array.size)
    known_size = math.prod(item for item in resolved if item != -1)
    if -1 in resolved:
        if known_size == 0 or input_size % known_size:
            raise _node_error(node, "shape cannot infer the -1 dimension")
        resolved[resolved.index(-1)] = input_size // known_size
    if math.prod(resolved) != input_size:
        raise _node_error(node, "shape changes tensor element count")
    shape = tuple(resolved)
    return _Evaluation(shape, value.data_type, lambda: np.reshape(value.array, shape))


def _squeeze(
    node: NodeProto,
    inputs: tuple[_Constant, ...],
    opset: int,
) -> _Evaluation:
    allowed_inputs = {1, 2} if opset >= 13 else {1}
    _require_arity(node, inputs, allowed_inputs)
    allowed_attributes = {} if opset >= 13 else {"axes": AttributeProto.INTS}
    attributes = _attributes(node, allowed_attributes)
    value = inputs[0]
    if len(inputs) == 2:
        raw_axes = _int_values(node, inputs[1], "axes")
    elif "axes" in attributes:
        raw_axes = tuple(int(item) for item in attributes["axes"].ints)
    else:
        raw_axes = tuple(
            index for index, dimension in enumerate(value.array.shape) if dimension == 1
        )
    axes = tuple(_axis(node, item, value.array.ndim) for item in raw_axes)
    if len(set(axes)) != len(axes):
        raise _node_error(node, "axes contain duplicates")
    if any(value.array.shape[item] != 1 for item in axes):
        raise _node_error(node, "only dimensions of size 1 can be squeezed")
    shape = tuple(
        dimension
        for index, dimension in enumerate(value.array.shape)
        if index not in set(axes)
    )
    return _Evaluation(
        shape,
        value.data_type,
        lambda: np.squeeze(value.array, axis=axes),
    )


def _unsqueeze(
    node: NodeProto,
    inputs: tuple[_Constant, ...],
    opset: int,
) -> _Evaluation:
    allowed_inputs = {2} if opset >= 13 else {1}
    _require_arity(node, inputs, allowed_inputs)
    allowed_attributes = {} if opset >= 13 else {"axes": AttributeProto.INTS}
    attributes = _attributes(node, allowed_attributes)
    value = inputs[0]
    raw_axes = (
        _int_values(node, inputs[1], "axes")
        if len(inputs) == 2
        else tuple(int(item) for item in attributes.get("axes", AttributeProto()).ints)
    )
    if not raw_axes:
        raise _node_error(node, "axes must not be empty")
    output_rank = value.array.ndim + len(raw_axes)
    axes = tuple(_axis(node, item, output_rank - 1, insertion=True) for item in raw_axes)
    if len(set(axes)) != len(axes):
        raise _node_error(node, "axes contain duplicates")
    shape = list(value.array.shape)
    for axis in sorted(axes):
        shape.insert(axis, 1)

    def evaluate() -> np.ndarray[Any, Any]:
        result = value.array
        for axis in sorted(axes):
            result = np.expand_dims(result, axis)
        return result

    return _Evaluation(tuple(shape), value.data_type, evaluate)


def _slice(
    node: NodeProto,
    inputs: tuple[_Constant, ...],
    opset: int,
) -> _Evaluation:
    if opset < 10:
        _require_arity(node, inputs, {1})
        attributes = _attributes(
            node,
            {
                "starts": AttributeProto.INTS,
                "ends": AttributeProto.INTS,
                "axes": AttributeProto.INTS,
            },
        )
        if "starts" not in attributes or "ends" not in attributes:
            raise _node_error(node, "starts and ends attributes are required")
        starts = tuple(int(item) for item in attributes["starts"].ints)
        ends = tuple(int(item) for item in attributes["ends"].ints)
        axes = tuple(
            int(item)
            for item in attributes.get(
                "axes",
                helper.make_attribute("axes", list(range(len(starts)))),
            ).ints
        )
        steps = (1,) * len(starts)
        value = inputs[0]
    else:
        if len(node.input) not in {3, 4, 5} or len(node.output) != 1 or not node.output[0]:
            raise _node_error(node, "expected 3/4/5 inputs and 1 output")
        _attributes(node, {})
        present = iter(inputs)
        slots = tuple(next(present) if name else None for name in node.input)
        data_value, starts_value, ends_value = slots[:3]
        if data_value is None or starts_value is None or ends_value is None:
            raise _node_error(node, "data, starts, and ends inputs are required")
        value = data_value
        starts = _int_values(node, starts_value, "starts")
        ends = _int_values(node, ends_value, "ends")
        axes = (
            _int_values(node, slots[3], "axes")
            if len(slots) >= 4 and slots[3] is not None
            else tuple(range(len(starts)))
        )
        steps = (
            _int_values(node, slots[4], "steps")
            if len(slots) == 5 and slots[4] is not None
            else (1,) * len(starts)
        )
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        raise _node_error(node, "slice controls must have equal lengths")
    normalized_axes = tuple(_axis(node, item, value.array.ndim) for item in axes)
    if len(set(normalized_axes)) != len(normalized_axes):
        raise _node_error(node, "axes contain duplicates")
    if any(step == 0 for step in steps):
        raise _node_error(node, "steps must not contain zero")
    slices: list[slice] = [slice(None)] * value.array.ndim
    shape = list(value.array.shape)
    for start, end, axis, step in zip(
        starts,
        ends,
        normalized_axes,
        steps,
        strict=True,
    ):
        item = slice(start, end, step)
        normalized = item.indices(value.array.shape[axis])
        shape[axis] = len(range(*normalized))
        slices[axis] = item
    key = tuple(slices)
    return _Evaluation(
        tuple(shape),
        value.data_type,
        lambda: value.array[key],
    )


def _concat(
    node: NodeProto,
    inputs: tuple[_Constant, ...],
    opset: int,
) -> _Evaluation:
    del opset
    if not inputs:
        raise _node_error(node, "expected at least one input")
    _require_arity(node, inputs, {len(inputs)})
    attributes = _attributes(node, {"axis": AttributeProto.INT})
    if "axis" not in attributes:
        raise _node_error(node, "required attribute 'axis' is missing")
    data_type = _same_dtype(node, inputs)
    rank = inputs[0].array.ndim
    if any(value.array.ndim != rank for value in inputs):
        raise _node_error(node, "all inputs must have the same rank")
    axis = _axis(node, int(attributes["axis"].i), rank)
    shape = list(inputs[0].array.shape)
    for value in inputs[1:]:
        if any(
            left != right
            for index, (left, right) in enumerate(
                zip(shape, value.array.shape, strict=True)
            )
            if index != axis
        ):
            raise _node_error(node, "non-concatenated dimensions must match")
        shape[axis] += int(value.array.shape[axis])
    return _Evaluation(
        tuple(shape),
        data_type,
        lambda: np.concatenate([value.array for value in inputs], axis=axis),
    )


def _gather(
    node: NodeProto,
    inputs: tuple[_Constant, ...],
    opset: int,
) -> _Evaluation:
    del opset
    _require_arity(node, inputs, {2})
    attributes = _attributes(node, {"axis": AttributeProto.INT})
    data, indices = inputs
    if indices.data_type not in {int(TensorProto.INT32), int(TensorProto.INT64)}:
        raise _node_error(node, "indices must use INT32 or INT64")
    axis = _axis(
        node,
        int(attributes.get("axis", helper.make_attribute("axis", 0)).i),
        data.array.ndim,
    )
    dimension = int(data.array.shape[axis])
    if np.any(indices.array < -dimension) or np.any(indices.array >= dimension):
        raise _node_error(node, "index is out of bounds")
    shape = (
        tuple(data.array.shape[:axis])
        + tuple(indices.array.shape)
        + tuple(data.array.shape[axis + 1 :])
    )
    return _Evaluation(
        shape,
        data.data_type,
        lambda: np.take(data.array, indices.array, axis=axis),
    )


def _binary(
    operation: Callable[
        [np.ndarray[Any, Any], np.ndarray[Any, Any]],
        np.ndarray[Any, Any],
    ],
) -> Callable[[NodeProto, tuple[_Constant, ...], int], _Evaluation]:
    def evaluator(
        node: NodeProto,
        inputs: tuple[_Constant, ...],
        opset: int,
    ) -> _Evaluation:
        del opset
        _require_arity(node, inputs, {2})
        _attributes(node, {})
        data_type = _same_dtype(node, inputs)
        if data_type == int(TensorProto.BOOL):
            raise _node_error(node, f"{node.op_type} does not accept BOOL tensors")
        try:
            shape = np.broadcast_shapes(inputs[0].array.shape, inputs[1].array.shape)
        except ValueError as error:
            raise _node_error(node, "input shapes are not broadcastable") from error
        return _Evaluation(
            tuple(int(item) for item in shape),
            data_type,
            lambda: operation(inputs[0].array, inputs[1].array),
        )

    return evaluator


def _divide(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    if np.any(right == 0):
        raise ValueError("division by zero")
    if np.issubdtype(left.dtype, np.integer):
        quotient = np.floor_divide(left, right)
        if np.issubdtype(left.dtype, np.signedinteger):
            remainder = np.remainder(left, right)
            correction = (remainder != 0) & ((left < 0) != (right < 0))
            quotient = quotient + correction.astype(left.dtype)
        return np.asarray(quotient, dtype=left.dtype)
    return np.asarray(np.divide(left, right))


def _neg(
    node: NodeProto,
    inputs: tuple[_Constant, ...],
    opset: int,
) -> _Evaluation:
    del opset
    _require_arity(node, inputs, {1})
    _attributes(node, {})
    value = inputs[0]
    if value.data_type in {
        int(TensorProto.BOOL),
        int(TensorProto.UINT8),
        int(TensorProto.UINT16),
        int(TensorProto.UINT32),
        int(TensorProto.UINT64),
    }:
        raise _node_error(node, "Neg input dtype is unsupported")
    return _Evaluation(value.array.shape, value.data_type, lambda: np.negative(value.array))


_EVALUATORS: Mapping[
    str,
    Callable[[NodeProto, tuple[_Constant, ...], int], _Evaluation],
] = {
    "Identity": _identity,
    "Cast": _cast,
    "Transpose": _transpose,
    "Reshape": _reshape,
    "Squeeze": _squeeze,
    "Unsqueeze": _unsqueeze,
    "Slice": _slice,
    "Concat": _concat,
    "Gather": _gather,
    "Add": _binary(np.add),
    "Sub": _binary(np.subtract),
    "Mul": _binary(np.multiply),
    "Div": _binary(_divide),
    "Neg": _neg,
}
