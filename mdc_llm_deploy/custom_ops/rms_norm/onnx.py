"""Narrow Dynamo ONNX direct-export contract for RmsNorm."""

from __future__ import annotations

import math
from typing import cast

from onnx import helper
from onnx.defs import OpSchema
from onnxscript import ir, values

_LOCAL_OPSET = values.Opset("", 18)
_SUPPORTED_DTYPES = frozenset(
    {ir.DataType.FLOAT16, ir.DataType.BFLOAT16, ir.DataType.FLOAT}
)


def create_schema() -> OpSchema:
    """Create the process-local default-domain NPURmsNorm schema."""
    parameter = OpSchema.FormalParameter
    epsilon = OpSchema.Attribute(
        "epsilon",
        helper.make_attribute("epsilon", 1e-6),
        "Positive normalization epsilon.",
    )
    return OpSchema(
        "NPURmsNorm",
        "",
        18,
        doc="Apply RMS normalization over the trailing dimensions described by gamma.",
        inputs=[
            parameter("x", "T"),
            parameter("gamma", "T"),
        ],
        outputs=[
            parameter("y", "T"),
            parameter("rstd", "tensor(float)"),
        ],
        type_constraints=[
            (
                "T",
                ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"],
                "Supported floating-point tensor types.",
            )
        ],
        attributes=[epsilon],
    )


def _static_shape(value: ir.Value, name: str) -> tuple[int, ...]:
    shape = value.shape
    if shape is None:
        raise RuntimeError(f"RmsNorm ONNX export requires known {name} rank")
    if not shape.is_static():
        raise RuntimeError("RmsNorm ONNX export requires static input shapes")
    return tuple(cast(int, dimension) for dimension in shape)


def validate_onnx_inputs(
    x: ir.Value,
    gamma: ir.Value,
    epsilon: float,
) -> None:
    """Validate only the static MC62 direct-export subset."""
    if not isinstance(epsilon, (float, int)) or isinstance(epsilon, bool):
        raise TypeError("RmsNorm ONNX epsilon must be a real number")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("RmsNorm ONNX epsilon must be finite and positive")

    x_shape = _static_shape(x, "x")
    gamma_shape = _static_shape(gamma, "gamma")
    if not 1 <= len(x_shape) <= 8:
        raise RuntimeError("RmsNorm ONNX x rank must be between 1 and 8")
    if not 1 <= len(gamma_shape) <= len(x_shape):
        raise RuntimeError("RmsNorm ONNX gamma rank must be between 1 and x rank")
    if any(dimension == 0 for dimension in gamma_shape):
        raise RuntimeError("RmsNorm ONNX gamma dimensions must be non-empty")
    if x_shape[-len(gamma_shape) :] != gamma_shape:
        raise RuntimeError("RmsNorm ONNX gamma shape must match trailing x dimensions")
    if x.dtype not in _SUPPORTED_DTYPES or gamma.dtype != x.dtype:
        raise RuntimeError(
            "RmsNorm ONNX inputs must have the same supported floating dtype"
        )


def translate(
    x: ir.Value,
    gamma: ir.Value,
    epsilon: float = 1e-6,
) -> tuple[ir.Value, ir.Value]:
    """Emit one default-domain NPURmsNorm node with two outputs."""
    validate_onnx_inputs(x, gamma, epsilon)
    outputs = _LOCAL_OPSET.NPURmsNorm(x, gamma, epsilon=float(epsilon))
    return cast(tuple[ir.Value, ir.Value], outputs)


ONNX_SCHEMA = create_schema()
