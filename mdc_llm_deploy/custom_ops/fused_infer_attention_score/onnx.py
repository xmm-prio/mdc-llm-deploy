"""Narrow MC62 float-decode ONNX contract and Dynamo translation."""

from __future__ import annotations

import math
from typing import Any, Final

import onnx
from onnx.defs import OpSchema
from onnxscript import opset18 as op
from onnxscript.values import Opset

from .contract import MAX_TOKENS, TORCH_INPUT_SLOTS

ONNX_OP_NAME: Final = "FusedInferAttentionScore"
ONNX_ATTRIBUTE_NAMES: Final = frozenset(
    {"num_heads", "scale", "input_layout", "num_key_value_heads"}
)
_LOCAL_OPSET = Opset("", 18)


def create_schema() -> OpSchema:
    """Create the process-local default-domain FIA schema."""
    parameter = OpSchema.FormalParameter
    attribute = OpSchema.Attribute
    attr_type = OpSchema.AttrType
    return OpSchema(
        ONNX_OP_NAME,
        "",
        18,
        doc="MC62 float decode fused inference attention.",
        inputs=[
            parameter("query", "T"),
            parameter("key", "T"),
            parameter("value", "T"),
        ],
        outputs=[parameter("attention_out", "T")],
        type_constraints=[
            (
                "T",
                ["tensor(float16)", "tensor(bfloat16)"],
                "Supported MC62 float decode tensor types.",
            )
        ],
        attributes=[
            attribute("num_heads", attr_type.INT, "Query head count."),
            attribute("scale", attr_type.FLOAT, "Attention score scale."),
            attribute("input_layout", attr_type.STRING, "Input tensor layout."),
            attribute("num_key_value_heads", attr_type.INT, "Key/value head count."),
        ],
    )


def _shape(value: Any, name: str) -> tuple[int | None, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise RuntimeError(f"MC62 float decode ONNX requires known {name} rank")
    result: list[int | None] = []
    for dimension in shape:
        if isinstance(dimension, int):
            result.append(dimension)
        else:
            result.append(None)
    return tuple(result)


def _dtype_name(value: Any, name: str) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise RuntimeError(f"MC62 float decode ONNX requires known {name} dtype")
    return str(dtype).lower()


def _validate_qkv(query: Any, key: Any, value: Any, num_heads: int, kv_heads: int) -> None:
    shapes = {
        "query": _shape(query, "query"),
        "key": _shape(key, "key"),
        "value": _shape(value, "value"),
    }
    if any(len(shape) != 4 for shape in shapes.values()):
        raise RuntimeError("MC62 float decode ONNX requires rank-4 BNSD Q/K/V")
    query_shape = shapes["query"]
    key_shape = shapes["key"]
    value_shape = shapes["value"]
    if any(dimension is None for shape in shapes.values() for dimension in shape):
        raise RuntimeError("MC62 float decode ONNX requires static Q/K/V shapes")
    if query_shape[2] != 1:
        raise RuntimeError(
            "MC62 float decode ONNX requires query sequence length S=1; "
            "float prefill must use small ops or fully-int8 FIA"
        )
    if key_shape != value_shape:
        raise RuntimeError("MC62 float decode ONNX requires matching key/value shapes")
    if query_shape[0] != key_shape[0] or query_shape[3] != key_shape[3]:
        raise RuntimeError("MC62 float decode ONNX Q/K/V batch and head dimensions must match")
    if query_shape[1] != num_heads:
        raise RuntimeError("MC62 float decode ONNX num_heads must match query heads")
    effective_kv_heads = key_shape[1] if kv_heads == 0 else kv_heads
    if effective_kv_heads is None:
        raise RuntimeError("MC62 float decode ONNX requires static K/V head counts")
    if key_shape[1] != effective_kv_heads:
        raise RuntimeError("MC62 float decode ONNX num_key_value_heads must match K/V heads")
    if num_heads % effective_kv_heads:
        raise RuntimeError("MC62 float decode ONNX requires valid GQA head divisibility")

    dtypes = [
        _dtype_name(tensor, name)
        for name, tensor in zip(shapes, (query, key, value), strict=True)
    ]
    if not any(token in dtypes[0] for token in ("float16", "bfloat16")):
        raise RuntimeError(
            "MC62 float decode ONNX FusedInferAttentionScore supports only "
            "FLOAT16 and BFLOAT16 Q/K/V"
        )
    if dtypes[1:] != dtypes[:1] * 2:
        raise RuntimeError("ONNX FusedInferAttentionScore Q/K/V dtypes must match")


def translate(
    query: Any,
    key: Any,
    value: Any,
    pse_shift: Any = None,
    atten_mask: Any = None,
    actual_seq_lengths: Any = None,
    actual_seq_lengths_kv: Any = None,
    dequant_scale1: Any = None,
    quant_scale1: Any = None,
    dequant_scale2: Any = None,
    quant_scale2: Any = None,
    quant_offset2: Any = None,
    antiquant_scale: Any = None,
    antiquant_offset: Any = None,
    block_table: Any = None,
    query_padding_size: Any = None,
    kv_padding_size: Any = None,
    key_antiquant_scale: Any = None,
    key_antiquant_offset: Any = None,
    value_antiquant_scale: Any = None,
    value_antiquant_offset: Any = None,
    key_shared_prefix: Any = None,
    value_shared_prefix: Any = None,
    actual_shared_prefix_len: Any = None,
    query_rope: Any = None,
    key_rope: Any = None,
    key_rope_antiquant_scale: Any = None,
    dequant_scale_query: Any = None,
    learnable_sink: Any = None,
    num_heads: int = 1,
    scale: float = 1.0,
    pre_tokens: int = MAX_TOKENS,
    next_tokens: int = MAX_TOKENS,
    input_layout: str = "BNSD",
    num_key_value_heads: int = 0,
    sparse_mode: int = 0,
    inner_precise: int = 0,
    block_size: int = 0,
    antiquant_mode: int = 0,
    softmax_lse_flag: bool = False,
    key_antiquant_mode: int = 0,
    value_antiquant_mode: int = 0,
    query_quant_mode: int = 0,
) -> tuple[Any, Any]:
    """Lower broad Torch FIA calls to the narrow three-input decode ABI."""
    optional = (
        pse_shift,
        atten_mask,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        dequant_scale1,
        quant_scale1,
        dequant_scale2,
        quant_scale2,
        quant_offset2,
        antiquant_scale,
        antiquant_offset,
        block_table,
        query_padding_size,
        kv_padding_size,
        key_antiquant_scale,
        key_antiquant_offset,
        value_antiquant_scale,
        value_antiquant_offset,
        key_shared_prefix,
        value_shared_prefix,
        actual_shared_prefix_len,
        query_rope,
        key_rope,
        key_rope_antiquant_scale,
        dequant_scale_query,
        learnable_sink,
    )
    unsupported = [
        TORCH_INPUT_SLOTS[index + 3]
        for index, tensor in enumerate(optional)
        if tensor is not None
    ]
    if unsupported:
        raise RuntimeError(
            f"MC62 float decode ONNX does not support optional inputs: {', '.join(unsupported)}"
        )
    if input_layout != "BNSD":
        raise RuntimeError("MC62 ONNX FusedInferAttentionScore supports only BNSD layout")
    if num_heads <= 0 or num_key_value_heads < 0:
        raise RuntimeError("MC62 float decode ONNX requires positive valid head counts")
    if not math.isfinite(scale):
        raise RuntimeError("MC62 float decode ONNX requires finite scale")
    if softmax_lse_flag:
        raise RuntimeError("MC62 ONNX FusedInferAttentionScore does not support softmax LSE")
    non_default = {
        "pre_tokens": pre_tokens != MAX_TOKENS,
        "next_tokens": next_tokens != MAX_TOKENS,
        "sparse_mode": sparse_mode != 0,
        "inner_precise": inner_precise != 0,
        "block_size": block_size != 0,
        "antiquant_mode": antiquant_mode != 0,
        "key_antiquant_mode": key_antiquant_mode != 0,
        "value_antiquant_mode": value_antiquant_mode != 0,
        "query_quant_mode": query_quant_mode != 0,
    }
    unsupported_attributes = [name for name, enabled in non_default.items() if enabled]
    if unsupported_attributes:
        raise RuntimeError(
            "MC62 float decode ONNX does not support attributes: "
            f"{', '.join(unsupported_attributes)}"
        )
    _validate_qkv(query, key, value, num_heads, num_key_value_heads)

    attention_out = _LOCAL_OPSET.FusedInferAttentionScore(
        query,
        key,
        value,
        num_heads=num_heads,
        scale=scale,
        input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
    )
    softmax_lse = op.Constant(
        value=onnx.helper.make_tensor(
            "softmax_lse",
            onnx.TensorProto.FLOAT,
            [1],
            [0.0],
        )
    )
    return attention_out, softmax_lse
