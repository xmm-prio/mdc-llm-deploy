"""Narrow five-input MDC ONNX contract for MoeExpert."""

from __future__ import annotations

from typing import Any

from onnxscript.values import Opset

from ...onnx.schemas import create_moe_expert_schema

_OPSET18 = Opset("", 18)


def _static_metadata(value: Any, name: str) -> tuple[tuple[int, ...], str]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        raise ValueError(f"MoeExpert ONNX export requires known {name} rank and dtype")
    try:
        static_shape = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"MoeExpert ONNX export requires static {name} shape"
        ) from error
    return static_shape, str(dtype).upper()


def _has_dtype(dtype: str, expected: str) -> bool:
    return dtype == expected or dtype.endswith(f".{expected}")


def validate_onnx_contract(
    x: Any,
    topk_ids: Any,
    topk_weight: Any,
    expert_weights: Any,
    quant_scales: Any,
) -> None:
    """Validate only the fully quantized MDC direct-export subset."""
    x_shape, x_dtype = _static_metadata(x, "x")
    if len(x_shape) != 2 or not _has_dtype(x_dtype, "INT8"):
        raise ValueError("MoeExpert ONNX x must be INT8 [T,H]")

    ids_shape, ids_dtype = _static_metadata(topk_ids, "topk_ids")
    routing_shape, routing_dtype = _static_metadata(topk_weight, "topk_weight")
    weights_shape, weights_dtype = _static_metadata(expert_weights, "expert_weights")
    scales_shape, scales_dtype = _static_metadata(quant_scales, "quant_scales")

    if len(ids_shape) != 2 or not _has_dtype(ids_dtype, "INT16"):
        raise ValueError("MoeExpert ONNX topk_ids must be INT16 [T,K]")
    if routing_shape != ids_shape or not _has_dtype(routing_dtype, "FLOAT16"):
        raise ValueError("MoeExpert ONNX topk_weight must be FLOAT16 [T,K]")
    if ids_shape[0] != x_shape[0] or ids_shape[1] <= 0:
        raise ValueError("MoeExpert ONNX routing shape must match T with positive K")
    if len(weights_shape) != 2 or not _has_dtype(weights_dtype, "INT8"):
        raise ValueError("MoeExpert ONNX expert_weights must be INT8 [3*E*I,H]")
    if len(scales_shape) != 1 or not _has_dtype(scales_dtype, "FLOAT"):
        raise ValueError("MoeExpert ONNX quant_scales must be FLOAT32 [1+4E]")
    if scales_shape[0] < 5 or (scales_shape[0] - 1) % 4:
        raise ValueError("MoeExpert ONNX quant_scales length must equal 1 + 4E")

    expert_count = (scales_shape[0] - 1) // 4
    if weights_shape[1] != x_shape[1] or weights_shape[0] % (3 * expert_count):
        raise ValueError("MoeExpert ONNX expert_weights shape must be [3*E*I,H]")
    intermediate_size = weights_shape[0] // (3 * expert_count)
    if x_shape[1] <= 0 or x_shape[1] % 256:
        raise ValueError("MoeExpert ONNX H must be a positive multiple of 256")
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("MoeExpert ONNX I must be a positive multiple of 128")


def translate(
    x: Any,
    topk_ids: Any,
    topk_weight: Any,
    expert_weights: Any,
    quant_scales: Any = None,
) -> Any:
    """Emit default-domain MoeExpert with five actual inputs."""
    validate_onnx_contract(
        x, topk_ids, topk_weight, expert_weights, quant_scales
    )
    return _OPSET18.MoeExpert(
        x, topk_ids, topk_weight, expert_weights, quant_scales
    )


create_schema = create_moe_expert_schema
