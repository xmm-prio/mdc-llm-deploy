"""MDC-specific ONNX graph transformations."""

from .fusion import FusionPassResult, FusionReport, run_fusion_passes
from .pipeline import (
    AdapterConfig,
    OnnxAdapter,
    downgrade_opset,
    lower_opset_compatibility,
    lower_qdq,
    normalize_graph,
)
from .schema import register_schemas

__all__ = [
    "AdapterConfig",
    "FusionPassResult",
    "FusionReport",
    "OnnxAdapter",
    "downgrade_opset",
    "lower_opset_compatibility",
    "lower_qdq",
    "normalize_graph",
    "register_schemas",
    "run_fusion_passes",
]
