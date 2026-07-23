"""Lower supported static W8A8 MatMul QDQ graphs to MC62 deployment operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper, numpy_helper

from ._graph import (
    GraphIndex,
    TensorInfo,
    attribute_int,
    attribute_ints,
    clone_model,
    constant_array,
    graph_names,
    remove_unused_initializers,
    remove_value_info,
    unique_name,
)
from .schemas import ASCEND_DEQUANT_OP, ASCEND_QUANT_OP

_QDQ_OPS: Final = frozenset({"QuantizeLinear", "DequantizeLinear"})
_FLOAT_DTYPES: Final = frozenset({TensorProto.FLOAT16, TensorProto.FLOAT})


@dataclass(frozen=True, slots=True)
class _QDQPair:
    q: NodeProto
    dq: NodeProto
    float_name: str
    scale_name: str
    zero_point_name: str | None
    scale: np.ndarray
    zero_point: np.ndarray | None
    axis: int


@dataclass(frozen=True, slots=True)
class _LoweringPlan:
    matmul: NodeProto
    activation: _QDQPair
    weight: _QDQPair
    transpose: NodeProto | None
    quantized_weight: np.ndarray
    weight_scale: np.ndarray
    activation_info: TensorInfo
    output_info: TensorInfo
    per_token: bool
    removed_node_ids: frozenset[int]


def _node_label(node: NodeProto) -> str:
    return node.name or node.op_type


def _normal_axis(axis: int, rank: int, *, where: str) -> int:
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"{where}: axis {axis} is invalid for rank {rank}")
    return normalized


def _zero_point_name(node: NodeProto) -> str | None:
    if len(node.input) < 3 or not node.input[2]:
        return None
    return str(node.input[2])


def _require_int8(pair_node: NodeProto, zero_point: np.ndarray | None, *, where: str) -> None:
    if zero_point is not None:
        if zero_point.dtype != np.dtype(np.int8):
            raise ValueError(f"{where}: zero point must be INT8, got {zero_point.dtype}")
        return
    output_dtype = attribute_int(pair_node, "output_dtype")
    if output_dtype != TensorProto.INT8:
        raise ValueError(f"{where}: Q without zero point must declare output_dtype=INT8")


def _read_qdq_pair(dq: NodeProto, index: GraphIndex, *, where: str) -> _QDQPair:
    if dq.domain not in ("", "ai.onnx") or dq.op_type != "DequantizeLinear":
        raise ValueError(f"{where}: expected default-domain DequantizeLinear")
    if len(dq.input) < 2:
        raise ValueError(f"{where}: DequantizeLinear is missing scale")
    q = index.producer(dq.input[0])
    if q is None or q.domain not in ("", "ai.onnx") or q.op_type != "QuantizeLinear":
        raise ValueError(f"{where}: expected DequantizeLinear(QuantizeLinear(...))")
    if len(q.input) < 2:
        raise ValueError(f"{where}: QuantizeLinear is missing scale")
    if q.input[1] != dq.input[1] or _zero_point_name(q) != _zero_point_name(dq):
        raise ValueError(f"{where}: QuantizeLinear and DequantizeLinear parameters must match")
    if attribute_int(q, "block_size", 0) != 0 or attribute_int(dq, "block_size", 0) != 0:
        raise ValueError(f"{where}: blocked quantization is not supported")
    q_axis_value = attribute_int(q, "axis", 1)
    dq_axis_value = attribute_int(dq, "axis", 1)
    assert q_axis_value is not None and dq_axis_value is not None
    q_axis = int(q_axis_value)
    dq_axis = int(dq_axis_value)
    if q_axis != dq_axis:
        raise ValueError(f"{where}: QuantizeLinear and DequantizeLinear axis values must match")

    scale = constant_array(index, q.input[1])
    if scale is None:
        raise ValueError(f"{where}: scale must be constant")
    scale = np.asarray(scale)
    if scale.ndim > 1:
        raise ValueError(f"{where}: scale must be scalar or one-dimensional")
    if not np.issubdtype(scale.dtype, np.floating):
        raise ValueError(f"{where}: scale must be floating point")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"{where}: scale must contain finite positive values")

    zero_point_name = _zero_point_name(q)
    zero_point = None
    if zero_point_name is not None:
        zero_point = constant_array(index, zero_point_name)
        if zero_point is None:
            raise ValueError(f"{where}: zero point must be constant")
        zero_point = np.asarray(zero_point)
        if zero_point.shape != scale.shape:
            raise ValueError(f"{where}: zero point shape must match scale shape")
    _require_int8(q, zero_point, where=where)

    q_users = index.users(q.output[0])
    if len(q_users) != 1 or q_users[0] is not dq:
        raise ValueError(f"{where}: QuantizeLinear output must be consumed only by its paired DQ")
    return _QDQPair(
        q=q,
        dq=dq,
        float_name=q.input[0],
        scale_name=q.input[1],
        zero_point_name=zero_point_name,
        scale=scale,
        zero_point=zero_point,
        axis=q_axis,
    )


def _float_info(index: GraphIndex, pair: _QDQPair, *, where: str) -> TensorInfo:
    info = index.tensor_info.get(pair.float_name)
    if info is None:
        raise ValueError(f"{where}: floating input rank and dtype must be present in ONNX metadata")
    if info.elem_type not in _FLOAT_DTYPES:
        raise ValueError(f"{where}: only FLOAT16 and FLOAT inputs are supported")
    if len(info.shape) < 2:
        raise ValueError(f"{where}: activation rank must be at least 2")
    return info


def _output_info(index: GraphIndex, matmul: NodeProto, activation: TensorInfo, n_out: int) -> TensorInfo:
    info = index.tensor_info.get(matmul.output[0])
    if info is None:
        return TensorInfo(activation.elem_type, (*activation.shape[:-1], n_out))
    if info.elem_type not in _FLOAT_DTYPES:
        raise ValueError(f"MatMul '{_node_label(matmul)}': output must be FLOAT16 or FLOAT")
    return info


def _validate_activation(
    pair: _QDQPair,
    info: TensorInfo,
    *,
    where: str,
) -> bool:
    if pair.scale.ndim == 0:
        return False
    axis = _normal_axis(pair.axis, len(info.shape), where=where)
    if axis != len(info.shape) - 2:
        raise ValueError(f"{where}: per-token activation axis must normalize to -2")
    token_dimension = info.shape[axis]
    if isinstance(token_dimension, int) and token_dimension != pair.scale.size:
        raise ValueError(
            f"{where}: token scale length {pair.scale.size} does not match dimension {token_dimension}"
        )
    return True


def _weight_source(
    matmul: NodeProto,
    index: GraphIndex,
) -> tuple[NodeProto, NodeProto | None]:
    producer = index.producer(matmul.input[1])
    if producer is None:
        raise ValueError(f"MatMul '{_node_label(matmul)}': weight must be produced by QDQ")
    if producer.op_type == "DequantizeLinear":
        return producer, None
    if producer.op_type != "Transpose":
        raise ValueError(
            f"MatMul '{_node_label(matmul)}': only direct QDQ or one weight Transpose is supported"
        )
    dq = index.producer(producer.input[0])
    if dq is None or dq.op_type != "DequantizeLinear":
        raise ValueError(
            f"MatMul '{_node_label(matmul)}': Transpose input must be produced by weight QDQ"
        )
    return dq, producer


def _is_quantized_weight(producer: NodeProto | None, index: GraphIndex) -> bool:
    if producer is None:
        return False
    if producer.op_type == "DequantizeLinear":
        return True
    if producer.op_type != "Transpose" or not producer.input:
        return False
    transpose_input = index.producer(producer.input[0])
    return transpose_input is not None and transpose_input.op_type == "DequantizeLinear"


def _broadcast_parameter(parameter: np.ndarray, shape: tuple[int, ...], axis: int) -> np.ndarray:
    if parameter.size == 1:
        return parameter.reshape(())
    broadcast_shape = [1] * len(shape)
    broadcast_shape[axis] = int(parameter.size)
    return parameter.reshape(broadcast_shape)


def _quantize_weight(
    pair: _QDQPair,
    weight: np.ndarray,
    transpose: NodeProto | None,
    *,
    where: str,
) -> tuple[np.ndarray, np.ndarray]:
    if weight.ndim != 2:
        raise ValueError(f"{where}: weight must be rank 2")
    if pair.zero_point is not None and np.any(pair.zero_point != 0):
        raise ValueError(f"{where}: weight quantization must be symmetric")

    if pair.scale.size == 1:
        axis = 0
    else:
        axis = _normal_axis(pair.axis, weight.ndim, where=where)
        expected_axis = 0 if transpose is not None else 1
        if axis != expected_axis:
            raise ValueError(f"{where}: per-channel scale must follow output channels")
        if weight.shape[axis] != pair.scale.size:
            raise ValueError(f"{where}: weight scale length does not match output channels")

    scaled = weight.astype(np.float64) / _broadcast_parameter(
        pair.scale.astype(np.float64),
        tuple(int(value) for value in weight.shape),
        axis,
    )
    quantized = np.clip(np.rint(scaled), -128, 127).astype(np.int8)

    if transpose is not None:
        permutation = attribute_ints(transpose, "perm") or (1, 0)
        if permutation != (1, 0):
            raise ValueError(f"{where}: only a two-dimensional swap Transpose is supported")
        quantized = np.ascontiguousarray(np.transpose(quantized, permutation))
    return quantized, pair.scale.astype(np.float32).reshape(-1)


def _require_exclusive_path(
    matmul: NodeProto,
    pair: _QDQPair,
    index: GraphIndex,
    *,
    transpose: NodeProto | None = None,
    where: str,
) -> None:
    expected = transpose if transpose is not None else matmul
    dq_users = index.users(pair.dq.output[0])
    if len(dq_users) != 1 or dq_users[0] is not expected:
        raise ValueError(f"{where}: DequantizeLinear output must have one supported consumer")
    if transpose is not None:
        users = index.users(transpose.output[0])
        if len(users) != 1 or users[0] is not matmul:
            raise ValueError(f"{where}: weight Transpose output must be consumed only by MatMul")


def _build_plan(matmul: NodeProto, index: GraphIndex) -> _LoweringPlan | None:
    if len(matmul.input) < 2 or not matmul.input[0] or not matmul.input[1]:
        raise ValueError(f"MatMul '{_node_label(matmul)}': expected two inputs")
    activation_dq = index.producer(matmul.input[0])
    weight_candidate = index.producer(matmul.input[1])
    activation_quantized = activation_dq is not None and activation_dq.op_type == "DequantizeLinear"
    weight_quantized = _is_quantized_weight(weight_candidate, index)
    if not activation_quantized and not weight_quantized:
        return None
    if not activation_quantized or not weight_quantized:
        raise ValueError(
            f"MatMul '{_node_label(matmul)}': activation and weight must both use supported QDQ"
        )

    where = f"MatMul '{_node_label(matmul)}'"
    assert activation_dq is not None
    activation = _read_qdq_pair(activation_dq, index, where=f"{where} activation")
    weight_dq, transpose = _weight_source(matmul, index)
    weight = _read_qdq_pair(weight_dq, index, where=f"{where} weight")
    _require_exclusive_path(matmul, activation, index, where=f"{where} activation")
    _require_exclusive_path(
        matmul,
        weight,
        index,
        transpose=transpose,
        where=f"{where} weight",
    )

    activation_info = _float_info(index, activation, where=f"{where} activation")
    per_token = _validate_activation(activation, activation_info, where=f"{where} activation")
    weight_array = constant_array(index, weight.float_name)
    if weight_array is None:
        raise ValueError(f"{where} weight: floating weight must be constant")
    quantized_weight, weight_scale = _quantize_weight(
        weight,
        np.asarray(weight_array),
        transpose,
        where=f"{where} weight",
    )
    activation_k = activation_info.shape[-1]
    if isinstance(activation_k, int) and activation_k != quantized_weight.shape[0]:
        raise ValueError(
            f"{where}: activation K={activation_k} does not match weight K={quantized_weight.shape[0]}"
        )
    output_info = _output_info(index, matmul, activation_info, quantized_weight.shape[1])
    removed = {id(activation.q), id(activation.dq), id(weight.q), id(weight.dq)}
    if transpose is not None:
        removed.add(id(transpose))
    return _LoweringPlan(
        matmul=matmul,
        activation=activation,
        weight=weight,
        transpose=transpose,
        quantized_weight=quantized_weight,
        weight_scale=weight_scale,
        activation_info=activation_info,
        output_info=output_info,
        per_token=per_token,
        removed_node_ids=frozenset(removed),
    )


def _numpy_dtype(elem_type: int) -> np.dtype[np.generic]:
    if elem_type == TensorProto.FLOAT16:
        return np.dtype(np.float16)
    if elem_type == TensorProto.FLOAT:
        return np.dtype(np.float32)
    raise ValueError(f"unsupported output elem_type: {elem_type}")


def _pack_fp32_scale(scale: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(scale, dtype=np.float32)
    return contiguous.view(np.uint32).astype(np.uint64)


def _activation_zero_point_correction(plan: _LoweringPlan) -> np.ndarray | None:
    zero_point = plan.activation.zero_point
    if zero_point is None or not np.any(zero_point):
        return None

    weight_sum = plan.quantized_weight.astype(np.int64).sum(axis=0)
    if plan.per_token:
        correction = zero_point.astype(np.int64).reshape(-1, 1) * weight_sum.reshape(1, -1)
        broadcast_shape = [1] * (len(plan.output_info.shape) - 2)
        correction = correction.reshape(*broadcast_shape, *correction.shape)
    else:
        correction = int(zero_point.reshape(())) * weight_sum

    int32 = np.iinfo(np.int32)
    if np.any(correction < int32.min) or np.any(correction > int32.max):
        raise ValueError(
            f"MatMul '{_node_label(plan.matmul)}': activation zero-point compensation "
            "does not fit INT32"
        )
    return correction.astype(np.int32)


def _emit_plan(
    model: onnx.ModelProto,
    plan: _LoweringPlan,
    names: set[str],
) -> tuple[list[NodeProto], list[NodeProto], set[str]]:
    graph = model.graph
    matmul = plan.matmul
    prefix = matmul.name or matmul.output[0] or "matmul"
    stale_values = {
        *plan.activation.q.output,
        *plan.activation.dq.output,
        *plan.weight.q.output,
        *plan.weight.dq.output,
    }
    if plan.transpose is not None:
        stale_values.update(plan.transpose.output)

    float_dtype = _numpy_dtype(plan.activation_info.elem_type)
    inverse = (1.0 / plan.activation.scale.astype(np.float64)).astype(float_dtype)
    if not np.all(np.isfinite(inverse)):
        raise ValueError(f"MatMul '{_node_label(matmul)}': inverse activation scale is not finite")
    inverse_name = unique_name(names, f"{prefix}_inverse_scale")
    graph.initializer.append(numpy_helper.from_array(inverse, inverse_name))

    quant_inputs = [plan.activation.float_name, inverse_name]
    if plan.activation.zero_point is not None:
        offset = plan.activation.zero_point.astype(float_dtype)
        offset_name = unique_name(names, f"{prefix}_offset")
        graph.initializer.append(numpy_helper.from_array(offset, offset_name))
        quant_inputs.append(offset_name)
    quant_output = unique_name(names, f"{prefix}_quantized")
    if plan.per_token:
        quant_node = helper.make_node(
            ASCEND_QUANT_OP,
            quant_inputs,
            [quant_output],
            name=unique_name(names, f"{prefix}_{ASCEND_QUANT_OP}"),
            axis=-2,
            dtype=2,
        )
    else:
        quant_node = helper.make_node(
            ASCEND_QUANT_OP,
            quant_inputs,
            [quant_output],
            name=unique_name(names, f"{prefix}_{ASCEND_QUANT_OP}"),
            dtype=2,
        )

    weight_name = unique_name(names, f"{prefix}_weight_int8")
    graph.initializer.append(numpy_helper.from_array(plan.quantized_weight, weight_name))
    original_output = matmul.output[0]
    matmul_accumulator = unique_name(names, f"{original_output}_int32")
    matmul.input[0] = quant_output
    matmul.input[1] = weight_name
    matmul.output[0] = matmul_accumulator

    accumulator = matmul_accumulator
    trailing: list[NodeProto] = []
    correction = _activation_zero_point_correction(plan)
    if correction is not None:
        correction_name = unique_name(names, f"{prefix}_zero_point_correction")
        graph.initializer.append(numpy_helper.from_array(correction, correction_name))
        accumulator = unique_name(names, f"{original_output}_corrected_int32")
        trailing.append(
            helper.make_node(
                "Sub",
                [matmul_accumulator, correction_name],
                [accumulator],
                name=unique_name(names, f"{prefix}_zero_point_correction"),
            )
        )

    output_dtype_attribute = 1 if plan.output_info.elem_type == TensorProto.FLOAT16 else 0
    if plan.per_token:
        dequant_scale = plan.weight_scale
    else:
        activation_scale = np.asarray(plan.activation.scale, dtype=np.float32).reshape(())
        dequant_scale = activation_scale * plan.weight_scale
    if dequant_scale.size == 1:
        dequant_scale = dequant_scale.reshape(())
    dequant_scale_name = unique_name(names, f"{prefix}_dequant_scale")
    graph.initializer.append(
        numpy_helper.from_array(_pack_fp32_scale(dequant_scale), dequant_scale_name)
    )

    dequant_output = (
        unique_name(names, f"{original_output}_dequant") if plan.per_token else original_output
    )
    dequant_node = helper.make_node(
        ASCEND_DEQUANT_OP,
        [accumulator, dequant_scale_name],
        [dequant_output],
        name=unique_name(names, f"{prefix}_{ASCEND_DEQUANT_OP}"),
        dtype=output_dtype_attribute,
    )
    trailing.append(dequant_node)

    if plan.per_token:
        rank = len(plan.output_info.shape)
        scale_shape = [1] * rank
        scale_shape[-2] = int(plan.activation.scale.size)
        token_scale = plan.activation.scale.astype(_numpy_dtype(plan.output_info.elem_type)).reshape(
            scale_shape
        )
        token_scale_name = unique_name(names, f"{prefix}_token_scale")
        graph.initializer.append(numpy_helper.from_array(token_scale, token_scale_name))
        trailing.append(
            helper.make_node(
                "Mul",
                [dequant_output, token_scale_name],
                [original_output],
                name=unique_name(names, f"{prefix}_token_scale_mul"),
            )
        )

    value_names = (
        (matmul_accumulator,)
        if accumulator == matmul_accumulator
        else (matmul_accumulator, accumulator)
    )
    for value_name in value_names:
        graph.value_info.append(
            helper.make_tensor_value_info(
                value_name,
                TensorProto.INT32,
                list(plan.output_info.shape),
            )
        )
    return [quant_node], trailing, stale_values


def _reject_quantized_gemm(model: onnx.ModelProto, index: GraphIndex) -> None:
    for node in model.graph.node:
        if node.op_type != "Gemm":
            continue
        for input_name in node.input[:2]:
            producer = index.producer(input_name)
            if producer is not None and producer.op_type in {"DequantizeLinear", "Transpose"}:
                raise ValueError(f"Gemm '{_node_label(node)}': quantized Gemm is not supported")


def lower_qdq_core(model: onnx.ModelProto) -> onnx.ModelProto:
    """Mutate a working ModelProto by lowering all supported main-graph QDQ MatMuls."""
    index = GraphIndex(model)
    _reject_quantized_gemm(model, index)
    plans = [
        plan
        for node in list(model.graph.node)
        if node.op_type == "MatMul"
        if (plan := _build_plan(node, index)) is not None
    ]

    owned_nodes: set[int] = set()
    for plan in plans:
        overlap = owned_nodes.intersection(plan.removed_node_ids)
        if overlap:
            raise ValueError(
                f"MatMul '{_node_label(plan.matmul)}': shared QDQ paths are not supported"
            )
        owned_nodes.update(plan.removed_node_ids)

    names = graph_names(model)
    before: dict[int, list[NodeProto]] = {}
    after: dict[int, list[NodeProto]] = {}
    stale_values: set[str] = set()
    for plan in plans:
        leading, trailing, stale = _emit_plan(model, plan, names)
        before[id(plan.matmul)] = leading
        after[id(plan.matmul)] = trailing
        stale_values.update(stale)

    rebuilt: list[NodeProto] = []
    for node in model.graph.node:
        if id(node) in owned_nodes:
            continue
        rebuilt.extend(before.get(id(node), []))
        rebuilt.append(node)
        rebuilt.extend(after.get(id(node), []))
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    remove_unused_initializers(model)
    remove_value_info(model, stale_values)

    residual = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.domain in ("", "ai.onnx") and node.op_type in _QDQ_OPS
        }
    )
    if residual:
        raise ValueError(f"main graph still contains residual QDQ operators: {residual}")
    return model


def lower_qdq(model: onnx.ModelProto) -> onnx.ModelProto:
    """Atomically lower supported main-graph QDQ MatMuls in place and return the same model."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an onnx.ModelProto")
    working = clone_model(model)
    lower_qdq_core(working)
    model.CopyFrom(working)
    return model


__all__ = ["lower_qdq"]
