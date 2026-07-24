"""Ordered ONNX deployment pipeline stages."""

from .adapter import AdapterConfig, OnnxAdapter
from .compatibility import lower_opset_compatibility
from .normalization import normalize_graph
from .opset import downgrade_opset
from .qdq import lower_qdq

__all__ = [
    "AdapterConfig",
    "OnnxAdapter",
    "downgrade_opset",
    "lower_opset_compatibility",
    "lower_qdq",
    "normalize_graph",
]
