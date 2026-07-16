"""Framework-independent operator contracts."""

from .attention import (
    ATTENTION_INPUT_COUNT,
    ATTENTION_OUTPUT_COUNT,
    RELEASE_ATTENTION_ATTRIBUTES,
    AttentionInput,
)
from .moe import (
    MoeExpertLayout,
    Projection,
    QuantSlot,
    WeightSegment,
    infer_moe_layout,
)
from .onnx import MDC_ONNX_DOMAIN, MDC_ONNX_OPSET
from .schema import (
    OPERATOR_SCHEMAS,
    TORCH_NAMESPACE,
    AttributeValue,
    OperatorSchema,
    schema_for_onnx_name,
    schema_for_torch_name,
)

__all__ = [
    "ATTENTION_INPUT_COUNT",
    "ATTENTION_OUTPUT_COUNT",
    "MDC_ONNX_DOMAIN",
    "MDC_ONNX_OPSET",
    "OPERATOR_SCHEMAS",
    "RELEASE_ATTENTION_ATTRIBUTES",
    "TORCH_NAMESPACE",
    "AttentionInput",
    "AttributeValue",
    "MoeExpertLayout",
    "OperatorSchema",
    "Projection",
    "QuantSlot",
    "WeightSegment",
    "infer_moe_layout",
    "schema_for_onnx_name",
    "schema_for_torch_name",
]
