"""MDC-specific ONNX graph transformations."""

from .compatibility_lowering import lower_opset_compatibility
from .fusion_pass import FusionPassResult, FusionReport, run_fusion_passes
from .normalization import normalize_graph
from .opset_downgrade import downgrade_opset
from .pipeline import process_onnx
from .quant_lowering import lower_qdq
from .schemas import register_schemas

__all__ = [
    "FusionPassResult",
    "FusionReport",
    "downgrade_opset",
    "lower_opset_compatibility",
    "lower_qdq",
    "normalize_graph",
    "process_onnx",
    "register_schemas",
    "run_fusion_passes",
]
