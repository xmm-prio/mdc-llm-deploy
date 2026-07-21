"""MDC-specific ONNX graph transformations."""

from .opset_downgrade import downgrade_opset
from .pipeline import process_onnx
from .quant_lowering import lower_qdq
from .schemas import register_schemas

register_schemas()

__all__ = ["downgrade_opset", "lower_qdq", "process_onnx"]
