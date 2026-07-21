"""Central declarations and lazy registration for MDC ONNX schemas."""

from .declarations import (
    ALL_SCHEMA_NAMES,
    ASCEND_DEQUANT_OP,
    ASCEND_QUANT_OP,
    CANN_FIA_SOURCE_COMMIT,
    CANN_FIA_SOURCE_URL,
    FUSED_INFER_ATTENTION_SCORE_OP,
    MDC_ONNX_OPSET,
    QUANTIZATION_SCHEMA_NAMES,
    RMS_NORM_OP,
    ROTARY_POSITION_EMBEDDING_OP,
    create_ascend_dequant_schema,
    create_ascend_quant_schema,
    create_fused_infer_attention_score_schema,
    create_rms_norm_schema,
    create_rotary_position_embedding_schema,
)
from .registry import (
    OnnxSchemaConflictError,
    register_schema_objects,
    register_schemas,
    schemas_for_names,
)

__all__ = [
    "ALL_SCHEMA_NAMES",
    "ASCEND_DEQUANT_OP",
    "ASCEND_QUANT_OP",
    "CANN_FIA_SOURCE_COMMIT",
    "CANN_FIA_SOURCE_URL",
    "FUSED_INFER_ATTENTION_SCORE_OP",
    "MDC_ONNX_OPSET",
    "QUANTIZATION_SCHEMA_NAMES",
    "RMS_NORM_OP",
    "ROTARY_POSITION_EMBEDDING_OP",
    "OnnxSchemaConflictError",
    "create_ascend_dequant_schema",
    "create_ascend_quant_schema",
    "create_fused_infer_attention_score_schema",
    "create_rms_norm_schema",
    "create_rotary_position_embedding_schema",
    "register_schema_objects",
    "register_schemas",
    "schemas_for_names",
]
