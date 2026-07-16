"""Opset 18 ONNX symbolics sourced from MDC operator schemas."""
# mypy: disable-error-code="no-any-return,no-untyped-call"

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch.onnx.symbolic_helper import (
    _optional_input_placeholder_tensor,
    parse_args,
)

from ..contracts.attention import (
    ATTENTION_INPUT_COUNT,
    AttentionInput,
)
from ..contracts.schema import OPERATOR_SCHEMAS

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
    resolved_num_heads = 1 if num_heads is None else num_heads
    resolved_kv_heads = (
        resolved_num_heads
        if num_key_value_heads is None
        else num_key_value_heads
    )
    inputs = [
        _optional_input_placeholder_tensor(graph)
        for _ in range(ATTENTION_INPUT_COUNT)
    ]
    inputs[AttentionInput.QUERY] = query
    inputs[AttentionInput.KEY] = key
    inputs[AttentionInput.VALUE] = value
    for slot, value in (
        (AttentionInput.ATTEN_MASK, atten_mask),
        (AttentionInput.QUANT_SCALE1, quant_scale1),
        (AttentionInput.KEY_ANTIQUANT_SCALE, key_antiquant_scale),
        (AttentionInput.KEY_ANTIQUANT_OFFSET, key_antiquant_offset),
        (AttentionInput.VALUE_ANTIQUANT_SCALE, value_antiquant_scale),
        (AttentionInput.VALUE_ANTIQUANT_OFFSET, value_antiquant_offset),
        (AttentionInput.DEQUANT_SCALE_QUERY, dequant_scale_query),
    ):
        if value is not None:
            inputs[slot] = value
    schema = _schema("FusedInferAttentionScore")
    attributes = schema.attributes
    return graph.op(
        schema.onnx_name,
        *inputs,
        num_heads_i=resolved_num_heads,
        scale_f=scale,
        input_layout_s=attributes["input_layout"],
        num_key_value_heads_i=resolved_kv_heads,
        sparse_mode_i=attributes["sparse_mode"],
        pre_tokens_i=attributes["pre_tokens"],
        next_tokens_i=attributes["next_tokens"],
        inner_precise_i=attributes["inner_precise"],
        block_size_i=attributes["block_size"],
        antiquant_mode_i=attributes["antiquant_mode"],
        softmax_lse_flag_i=int(softmax_lse_flag),
        key_antiquant_mode_i=attributes["key_antiquant_mode"],
        value_antiquant_mode_i=attributes[
            "value_antiquant_mode"
        ],
        query_quant_mode_i=attributes["query_quant_mode"],
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
    inputs = [x, topk_ids, topk_weight, expert_weights]
    if quant_scales is not None:
        inputs.append(quant_scales)
    if quant_offsets is not None:
        if quant_scales is None:
            inputs.append(_optional_input_placeholder_tensor(graph))
        inputs.append(quant_offsets)
    return graph.op(_schema("MoeExpert").onnx_name, *inputs)


_SYMBOLICS: Mapping[str, Symbolic] = MappingProxyType({
    "RmsNorm": _rms_norm_symbolic,
    "ApplyRotaryPosEmb": _rope_symbolic,
    "FusedInferAttentionScore": _attention_symbolic,
    "AscendQuantV2": _quant_symbolic,
    "AscendDequant": _dequant_symbolic,
    "MoeExpert": _moe_symbolic,
})
if set(_SYMBOLICS) != set(OPERATOR_SCHEMAS):
    raise RuntimeError(
        "Every MDC operator schema requires one ONNX symbolic"
    )
_REGISTERED = False


def register_onnx_symbolics() -> None:
    """Register all ONNX symbolics exactly once."""
    global _REGISTERED
    if _REGISTERED:
        return
    for name, schema in OPERATOR_SCHEMAS.items():
        torch.onnx.register_custom_op_symbolic(
            schema.qualified_torch_name,
            _SYMBOLICS[name],
            schema.opset,
        )
    _REGISTERED = True
