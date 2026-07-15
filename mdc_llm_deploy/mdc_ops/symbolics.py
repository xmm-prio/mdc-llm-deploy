"""Opset 18 ONNX symbolics sourced from MDC operator schemas."""
# mypy: disable-error-code="no-any-return"

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.onnx.symbolic_helper import parse_args

from .schema import OPERATOR_SCHEMAS

Symbolic = Callable[..., Any]


def _schema(name: str) -> Any:
    return OPERATOR_SCHEMAS[name]


@parse_args("v", "v", "f")
def _rms_norm_symbolic(
    graph: Any, x: Any, gamma: Any, epsilon: float
) -> tuple[Any, Any]:
    return graph.op(
        _schema("RmsNorm").onnx_name,
        x,
        gamma,
        epsilon_f=epsilon,
        outputs=2,
    )


@parse_args("v", "v", "v", "v", "i", "s")
def _rope_symbolic(
    graph: Any,
    query: Any,
    key: Any,
    cos: Any,
    sin: Any,
    layout: int,
    rotary_mode: str,
) -> tuple[Any, Any]:
    return graph.op(
        _schema("ApplyRotaryPosEmb").onnx_name,
        query,
        key,
        cos,
        sin,
        layout_i=layout,
        rotary_mode_s=rotary_mode,
        outputs=2,
    )


@parse_args(
    "v",
    "v",
    "v",
    "v",
    "f",
    "i",
    "i",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "b",
)
def _attention_symbolic(
    graph: Any,
    query: Any,
    key: Any,
    value: Any,
    atten_mask: Any,
    scale: float,
    num_heads: int | None,
    num_key_value_heads: int | None,
    key_antiquant_scale: Any,
    key_antiquant_offset: Any,
    value_antiquant_scale: Any,
    value_antiquant_offset: Any,
    dequant_scale_query: Any,
    quant_scale1: Any,
    softmax_lse_flag: bool,
) -> tuple[Any, Any]:
    del dequant_scale_query
    inputs = [
        query,
        key,
        value,
        None,
        atten_mask,
        None,
        None,
        None,
        quant_scale1,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        key_antiquant_scale,
        key_antiquant_offset,
        value_antiquant_scale,
        value_antiquant_offset,
    ]
    return graph.op(
        _schema("FusedInferAttentionScore").onnx_name,
        *inputs,
        num_heads_i=1 if num_heads is None else num_heads,
        scale_f=scale,
        input_layout_s="BNSD",
        num_key_value_heads_i=0
        if num_key_value_heads is None
        else num_key_value_heads,
        sparse_mode_i=0,
        softmax_lse_flag_i=int(softmax_lse_flag),
        outputs=2,
    )


@parse_args("v", "v", "v", "i", "i")
def _quant_symbolic(
    graph: Any,
    x: Any,
    scale: Any,
    offset: Any,
    axis: int,
    dtype: int,
) -> Any:
    inputs = [x, scale] if offset is None else [x, scale, offset]
    return graph.op(
        _schema("AscendQuantV2").onnx_name,
        *inputs,
        axis_i=axis,
        dtype_i=dtype,
    )


@parse_args("v", "v", "b", "b", "i")
def _dequant_symbolic(
    graph: Any,
    x: Any,
    deq_scale: Any,
    sqrt_mode: bool,
    relu_flag: bool,
    dtype: int,
) -> Any:
    return graph.op(
        _schema("AscendDequant").onnx_name,
        x,
        deq_scale,
        sqrt_mode_i=int(sqrt_mode),
        relu_flag_i=int(relu_flag),
        dtype_i=dtype,
    )


@parse_args("v", "v", "v", "v", "v", "v")
def _moe_symbolic(
    graph: Any,
    x: Any,
    topk_ids: Any,
    topk_weight: Any,
    expert_weights: Any,
    quant_scales: Any,
    quant_offsets: Any,
) -> Any:
    inputs = [x, topk_ids, topk_weight, expert_weights, quant_scales]
    if quant_offsets is not None:
        inputs.append(quant_offsets)
    return graph.op(_schema("MoeExpert").onnx_name, *inputs)


_SYMBOLICS: dict[str, Symbolic] = {
    "RmsNorm": _rms_norm_symbolic,
    "ApplyRotaryPosEmb": _rope_symbolic,
    "FusedInferAttentionScore": _attention_symbolic,
    "AscendQuantV2": _quant_symbolic,
    "AscendDequant": _dequant_symbolic,
    "MoeExpert": _moe_symbolic,
}


def register_onnx_symbolics() -> None:
    """Register all legacy ONNX symbolics at their schema-declared opset."""
    for name, schema in OPERATOR_SCHEMAS.items():
        torch.onnx.register_custom_op_symbolic(
            schema.qualified_torch_name,
            _SYMBOLICS[name],
            schema.opset,
        )


register_onnx_symbolics()
