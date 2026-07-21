"""Narrow Dynamo ONNX contract for ApplyRotaryPosEmb."""

from __future__ import annotations

from typing import Any, cast

import onnx
import onnxscript.ir as ir
from onnx.defs import OpSchema
from onnxscript import values

from .contract import HEAD_AXIS, LAYOUT_RANK, ROTARY_MODES

ONNX_NAME = "ApplyRotaryPosEmb"
ONNX_OPSET = 18
_MAX_HEAD_DIM = 1024
_ONNX_DTYPES = frozenset(
    {ir.DataType.FLOAT16, ir.DataType.BFLOAT16, ir.DataType.FLOAT}
)
_LOCAL_OPSET = values.Opset("", ONNX_OPSET)


def create_schema() -> OpSchema:
    """Create the process-local default-domain schema."""
    parameter = OpSchema.FormalParameter
    attribute = OpSchema.Attribute
    return OpSchema(
        ONNX_NAME,
        "",
        ONNX_OPSET,
        doc="Apply rotary position embeddings to query and key tensors.",
        inputs=[
            parameter("query", "T"),
            parameter("key", "T"),
            parameter("cos", "T"),
            parameter("sin", "T"),
        ],
        outputs=[parameter("query_out", "T"), parameter("key_out", "T")],
        type_constraints=[
            (
                "T",
                ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"],
                "Supported MDC floating-point tensor types.",
            )
        ],
        attributes=[
            attribute(
                "layout",
                onnx.helper.make_attribute("layout", 1),
                "Tensor layout: 1=BSND, 2=SBND, 3=BNSD, 4=TND.",
            ),
            attribute(
                "rotary_mode",
                onnx.helper.make_attribute("rotary_mode", "half"),
                "Rotation pairing mode.",
            ),
        ],
    )


def translate(
    query: ir.Value,
    key: ir.Value,
    cos: ir.Value,
    sin: ir.Value,
    layout: int = 1,
    rotary_mode: str = "half",
) -> tuple[Any, Any]:
    """Validate the board subset and emit one default-domain ONNX node."""
    validate_onnx_inputs(query, key, cos, sin, layout, rotary_mode)
    return cast(
        tuple[Any, Any],
        _LOCAL_OPSET.ApplyRotaryPosEmb(
            query,
            key,
            cos,
            sin,
            layout=layout,
            rotary_mode=rotary_mode,
            _outputs=2,
        ),
    )


def validate_onnx_inputs(
    query: ir.Value,
    key: ir.Value,
    cos: ir.Value,
    sin: ir.Value,
    layout: int,
    rotary_mode: str,
) -> None:
    """Reject inputs outside the static MC62 direct-export subset."""
    if layout not in LAYOUT_RANK:
        raise ValueError("ONNX ApplyRotaryPosEmb requires layout in {1, 2, 3, 4}")
    if rotary_mode not in ROTARY_MODES:
        raise ValueError(
            "ONNX ApplyRotaryPosEmb requires rotary_mode in "
            "{'half', 'interleave', 'quarter'}"
        )

    values_ = (query, key, cos, sin)
    if any(value.dtype not in _ONNX_DTYPES for value in values_):
        raise TypeError(
            "ONNX ApplyRotaryPosEmb requires FLOAT16, BFLOAT16, or FLOAT inputs"
        )
    if any(value.dtype != query.dtype for value in values_[1:]):
        raise TypeError("ONNX ApplyRotaryPosEmb requires all inputs to have the same dtype")

    shapes = tuple(_static_shape(value, name) for value, name in zip(
        values_, ("query", "key", "cos", "sin"), strict=True
    ))
    query_shape, key_shape, cos_shape, sin_shape = shapes
    rank = LAYOUT_RANK[layout]
    if any(len(shape) != rank for shape in shapes):
        raise ValueError(
            f"ONNX ApplyRotaryPosEmb layout {layout} requires rank-{rank} inputs"
        )
    if cos_shape != sin_shape:
        raise ValueError("ONNX ApplyRotaryPosEmb requires cos and sin shapes to match")

    head_axis = HEAD_AXIS[layout]
    for axis in range(rank):
        if axis != head_axis and query_shape[axis] != key_shape[axis]:
            raise ValueError(
                "ONNX ApplyRotaryPosEmb query and key may differ only in head count"
            )
    if cos_shape[head_axis] != 1:
        raise ValueError("ONNX ApplyRotaryPosEmb requires cos/sin head dimension 1")
    for axis in range(rank - 1):
        if axis != head_axis and cos_shape[axis] not in (1, query_shape[axis]):
            raise ValueError(
                "ONNX ApplyRotaryPosEmb cos/sin cannot broadcast to query shape"
            )

    head_dim = query_shape[-1]
    rotary_dim = cos_shape[-1]
    if key_shape[-1] != head_dim:
        raise ValueError("ONNX ApplyRotaryPosEmb requires equal query/key head dimensions")
    if head_dim > _MAX_HEAD_DIM:
        raise ValueError("ONNX ApplyRotaryPosEmb requires head dimension <= 1024")
    if not 0 < rotary_dim <= head_dim:
        raise ValueError("ONNX ApplyRotaryPosEmb requires 0 < rotary dimension <= head dimension")
    divisor = 4 if rotary_mode == "quarter" else 2
    if rotary_dim % divisor:
        raise ValueError(
            f"ONNX ApplyRotaryPosEmb {rotary_mode} rotary dimension must be "
            f"divisible by {divisor}"
        )


def _static_shape(value: ir.Value, name: str) -> tuple[int, ...]:
    shape = value.shape
    if shape is None or not shape.is_static():
        raise ValueError(
            f"ONNX ApplyRotaryPosEmb requires a static {name} shape"
        )
    dimensions = tuple(shape)
    if not all(isinstance(dimension, int) for dimension in dimensions):
        raise ValueError(
            f"ONNX ApplyRotaryPosEmb requires a static {name} shape"
        )
    return cast(tuple[int, ...], dimensions)
