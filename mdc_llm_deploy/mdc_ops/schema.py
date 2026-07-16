"""Compatibility exports for framework-independent operator schemas."""

from ..onnx_protocol import (
    MDC_ONNX_DOMAIN as MDC_ONNX_DOMAIN,
)
from ..onnx_protocol import (
    MDC_ONNX_OPSET as MDC_ONNX_OPSET,
)
from ..operator_schema import (
    OPERATOR_SCHEMAS as OPERATOR_SCHEMAS,
)
from ..operator_schema import (
    TORCH_NAMESPACE as TORCH_NAMESPACE,
)
from ..operator_schema import (
    AttributeValue as AttributeValue,
)
from ..operator_schema import (
    OperatorSchema as OperatorSchema,
)
from ..operator_schema import (
    schema_for_onnx_name as schema_for_onnx_name,
)
from ..operator_schema import (
    schema_for_torch_name as schema_for_torch_name,
)

__all__ = [
    "MDC_ONNX_DOMAIN",
    "MDC_ONNX_OPSET",
    "OPERATOR_SCHEMAS",
    "TORCH_NAMESPACE",
    "AttributeValue",
    "OperatorSchema",
    "schema_for_onnx_name",
    "schema_for_torch_name",
]
