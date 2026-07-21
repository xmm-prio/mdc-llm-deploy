"""Process-local ONNX schemas for MDC deployment operators."""

from __future__ import annotations

from typing import Final

from onnx import helper
from onnx.defs import OpSchema

from .._onnx_schema_registry import ensure_onnx_schemas

ASCEND_QUANT_OP: Final = "NPUAscendQuantV2"
ASCEND_DEQUANT_OP: Final = "AscendDequant"
MDC_ONNX_OPSET: Final = 18


def _quant_schema() -> OpSchema:
    parameter = OpSchema.FormalParameter
    option = OpSchema.FormalParameterOption
    attribute = OpSchema.Attribute
    return OpSchema(
        ASCEND_QUANT_OP,
        "",
        MDC_ONNX_OPSET,
        doc="MC62 activation quantization.",
        inputs=[
            parameter("x", "T"),
            parameter("scale", "T"),
            parameter("offset", "T", param_option=option.Optional),
        ],
        outputs=[parameter("y", "TQ")],
        type_constraints=[
            (
                "T",
                ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
                "Supported floating-point input types.",
            ),
            ("TQ", ["tensor(int8)"], "Supported quantized output type."),
        ],
        attributes=[
            attribute("axis", helper.make_attribute("axis", -1), "Scale axis."),
            attribute("dtype", helper.make_attribute("dtype", 2), "GE destination dtype."),
        ],
    )


def _dequant_schema() -> OpSchema:
    parameter = OpSchema.FormalParameter
    attribute = OpSchema.Attribute
    attr_type = OpSchema.AttrType
    return OpSchema(
        ASCEND_DEQUANT_OP,
        "",
        MDC_ONNX_OPSET,
        doc="MC62 INT32 accumulator dequantization.",
        inputs=[
            parameter("x", "TI"),
            parameter("deq_scale", "TS"),
        ],
        outputs=[parameter("y", "TO")],
        type_constraints=[
            ("TI", ["tensor(int32)"], "Accumulator type."),
            ("TS", ["tensor(uint64)"], "Packed FP32 dequant scale."),
            ("TO", ["tensor(float16)", "tensor(float)"], "Supported output types."),
        ],
        attributes=[
            attribute("dtype", attr_type.INT, "GE output dtype."),
        ],
    )


def register_schemas() -> None:
    """Register MDC schemas idempotently in the process-local ONNX registry."""
    ensure_onnx_schemas((_quant_schema(), _dequant_schema()))


__all__ = [
    "ASCEND_DEQUANT_OP",
    "ASCEND_QUANT_OP",
    "MDC_ONNX_OPSET",
    "register_schemas",
]
