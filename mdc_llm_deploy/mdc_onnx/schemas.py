"""Process-local ONNX schemas for MDC deployment operators."""

from __future__ import annotations

from typing import Final

from onnx import defs, helper
from onnx.defs import OpSchema

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


def _matches(schema: OpSchema, expected: OpSchema) -> bool:
    return (
        schema.name == expected.name
        and schema.domain == expected.domain
        and schema.since_version == expected.since_version
        and [item.name for item in schema.inputs] == [item.name for item in expected.inputs]
        and [item.name for item in schema.outputs] == [item.name for item in expected.outputs]
        and set(schema.attributes) == set(expected.attributes)
    )


def register_schemas() -> None:
    """Register MDC schemas idempotently in the process-local ONNX registry."""
    for schema in (_quant_schema(), _dequant_schema()):
        try:
            current = defs.get_schema(
                schema.name,
                schema.since_version,
                schema.domain,
            )
        except defs.SchemaError:
            defs.register_schema(schema)
        else:
            if not _matches(current, schema):
                raise RuntimeError(
                    f"conflicting ONNX schema already registered for "
                    f"{schema.domain!r}::{schema.name}@{schema.since_version}"
                )


__all__ = [
    "ASCEND_DEQUANT_OP",
    "ASCEND_QUANT_OP",
    "MDC_ONNX_OPSET",
    "register_schemas",
]
