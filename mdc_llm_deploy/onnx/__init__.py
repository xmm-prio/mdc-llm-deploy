"""MDC-specific ONNX graph transformations."""

from .adapter import AdapterConfig, OnnxAdapter
from .compatibility_lowering import lower_opset_compatibility
from .fusion_pass import FusionPassResult, FusionReport, run_fusion_passes
from .normalization import normalize_graph
from .opset_downgrade import downgrade_opset
from .quant_lowering import lower_qdq
from .schemas import register_schemas

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
