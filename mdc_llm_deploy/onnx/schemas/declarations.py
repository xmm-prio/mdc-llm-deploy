"""Pure declarations for MDC default-domain ONNX schemas."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Final

from onnx import helper
from onnx.defs import OpSchema

MDC_ONNX_OPSET: Final = 18

ASCEND_QUANT_OP: Final = "NPUAscendQuantV2"
ASCEND_DEQUANT_OP: Final = "AscendDequant"
RMS_NORM_OP: Final = "NPURmsNorm"
ROTARY_POSITION_EMBEDDING_OP: Final = "ApplyRotaryPosEmb"
FUSED_INFER_ATTENTION_SCORE_OP: Final = "FusedInferAttentionScore"
MOE_EXPERT_OP: Final = "MoeExpert"

CANN_FIA_SOURCE_COMMIT: Final = "606a5ddb67c67d93c137a7b474fa7a5edd05f7c9"
CANN_FIA_SOURCE_URL: Final = (
    "https://gitcode.com/cann/ops-transformer/blob/"
    f"{CANN_FIA_SOURCE_COMMIT}/attention/fused_infer_attention_score/"
    "op_host/fused_infer_attention_score_def.cpp"
)

SchemaFactory = Callable[[], OpSchema]


def _parameter(
    name: str,
    type_name: str,
    *,
    optional: bool = False,
) -> OpSchema.FormalParameter:
    option = OpSchema.FormalParameterOption
    return OpSchema.FormalParameter(
        name,
        type_name,
        param_option=option.Optional if optional else option.Single,
    )


def _default_attribute(name: str, value: object, description: str) -> OpSchema.Attribute:
    return OpSchema.Attribute(name, helper.make_attribute(name, value), description)


def _required_attribute(
    name: str,
    attribute_type: OpSchema.AttrType,
    description: str,
) -> OpSchema.Attribute:
    return OpSchema.Attribute(name, attribute_type, description, required=True)


def create_ascend_quant_schema() -> OpSchema:
    """Create the activation-quantization schema."""
    return OpSchema(
        ASCEND_QUANT_OP,
        "",
        MDC_ONNX_OPSET,
        doc="MC62 activation quantization.",
        inputs=[
            _parameter("x", "T"),
            _parameter("scale", "T"),
            _parameter("offset", "T", optional=True),
        ],
        outputs=[_parameter("y", "TQ")],
        type_constraints=[
            (
                "T",
                ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
                "Supported floating-point input types.",
            ),
            ("TQ", ["tensor(int8)"], "Supported quantized output type."),
        ],
        attributes=[
            _default_attribute("axis", -1, "Scale axis."),
            _default_attribute("dtype", 2, "GE destination dtype."),
        ],
    )


def create_ascend_dequant_schema() -> OpSchema:
    """Create the accumulator-dequantization schema."""
    return OpSchema(
        ASCEND_DEQUANT_OP,
        "",
        MDC_ONNX_OPSET,
        doc="MC62 INT32 accumulator dequantization.",
        inputs=[
            _parameter("x", "TI"),
            _parameter("deq_scale", "TS"),
        ],
        outputs=[_parameter("y", "TO")],
        type_constraints=[
            ("TI", ["tensor(int32)"], "Accumulator type."),
            ("TS", ["tensor(uint64)"], "Packed FP32 dequant scale."),
            ("TO", ["tensor(float16)", "tensor(float)"], "Supported output types."),
        ],
        attributes=[
            _required_attribute("dtype", OpSchema.AttrType.INT, "GE output dtype."),
        ],
    )


def create_rms_norm_schema() -> OpSchema:
    """Create the RMS normalization schema."""
    return OpSchema(
        RMS_NORM_OP,
        "",
        MDC_ONNX_OPSET,
        doc="Apply RMS normalization over the trailing dimensions described by gamma.",
        inputs=[_parameter("x", "T"), _parameter("gamma", "T")],
        outputs=[
            _parameter("y", "T"),
            _parameter("rstd", "tensor(float)"),
        ],
        type_constraints=[
            (
                "T",
                ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"],
                "Supported floating-point tensor types.",
            )
        ],
        attributes=[
            _default_attribute(
                "epsilon",
                1e-6,
                "Positive normalization epsilon.",
            )
        ],
    )


def create_rotary_position_embedding_schema() -> OpSchema:
    """Create the rotary-position-embedding schema."""
    return OpSchema(
        ROTARY_POSITION_EMBEDDING_OP,
        "",
        MDC_ONNX_OPSET,
        doc="Apply rotary position embeddings to query and key tensors.",
        inputs=[
            _parameter("query", "T"),
            _parameter("key", "T"),
            _parameter("cos", "T"),
            _parameter("sin", "T"),
        ],
        outputs=[
            _parameter("query_out", "T"),
            _parameter("key_out", "T"),
        ],
        type_constraints=[
            (
                "T",
                ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"],
                "Supported MDC floating-point tensor types.",
            )
        ],
        attributes=[
            _default_attribute(
                "layout",
                1,
                "Tensor layout: 1=BSND, 2=SBND, 3=BNSD, 4=TND.",
            ),
            _default_attribute(
                "rotary_mode",
                "half",
                "Rotation pairing mode.",
            ),
        ],
    )


def _fia_inputs() -> list[OpSchema.FormalParameter]:
    optional = True
    return [
        _parameter("query", "T_QUERY"),
        _parameter("key", "T_KEY"),
        _parameter("value", "T_VALUE"),
        _parameter("pse_shift", "T_PSE", optional=optional),
        _parameter("atten_mask", "T_MASK", optional=optional),
        _parameter("actual_seq_lengths", "T_INT64", optional=optional),
        _parameter("actual_seq_lengths_kv", "T_INT64", optional=optional),
        _parameter("dequant_scale1", "T_DEQUANT", optional=optional),
        _parameter("quant_scale1", "T_FLOAT32", optional=optional),
        _parameter("dequant_scale2", "T_DEQUANT", optional=optional),
        _parameter("quant_scale2", "T_POST_QUANT", optional=optional),
        _parameter("quant_offset2", "T_POST_QUANT", optional=optional),
        _parameter("antiquant_scale", "T_ANTIQUANT", optional=optional),
        _parameter("antiquant_offset", "T_ANTIQUANT", optional=optional),
        _parameter("block_table", "T_INT32", optional=optional),
        _parameter("query_padding_size", "T_INT64", optional=optional),
        _parameter("kv_padding_size", "T_INT64", optional=optional),
        _parameter("key_antiquant_scale", "T_ANTIQUANT", optional=optional),
        _parameter("key_antiquant_offset", "T_ANTIQUANT", optional=optional),
        _parameter("value_antiquant_scale", "T_ANTIQUANT", optional=optional),
        _parameter("value_antiquant_offset", "T_ANTIQUANT", optional=optional),
        _parameter("key_shared_prefix", "T_PREFIX", optional=optional),
        _parameter("value_shared_prefix", "T_PREFIX", optional=optional),
        _parameter("actual_shared_prefix_len", "T_INT64", optional=optional),
        _parameter("query_rope", "T_QUERY_ROPE", optional=optional),
        _parameter("key_rope", "T_KEY_ROPE", optional=optional),
        _parameter("key_rope_antiquant_scale", "T_ANTIQUANT", optional=optional),
        _parameter("dequant_scale_query", "T_FLOAT32", optional=optional),
        _parameter("learnable_sink", "T_SINK", optional=optional),
        _parameter("q_start_idx", "T_INT64", optional=optional),
        _parameter("kv_start_idx", "T_INT64", optional=optional),
    ]


def _fia_type_constraints() -> list[tuple[str, list[str], str]]:
    float_types = ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"]
    return [
        (
            "T_QUERY",
            ["tensor(float16)", "tensor(bfloat16)", "tensor(int8)"],
            "CANN query tensor types.",
        ),
        (
            "T_KEY",
            ["tensor(float16)", "tensor(bfloat16)", "tensor(int8)", "tensor(int4)"],
            "CANN key tensor types.",
        ),
        (
            "T_VALUE",
            ["tensor(float16)", "tensor(bfloat16)", "tensor(int8)", "tensor(int4)"],
            "CANN value tensor types.",
        ),
        ("T_PSE", ["tensor(float16)", "tensor(bfloat16)"], "PSE tensor types."),
        (
            "T_MASK",
            ["tensor(float16)", "tensor(bool)", "tensor(uint8)", "tensor(int8)"],
            "Attention mask tensor types.",
        ),
        ("T_INT64", ["tensor(int64)"], "INT64 metadata tensors."),
        (
            "T_DEQUANT",
            ["tensor(uint64)", "tensor(float)"],
            "Dequantization scale tensor types.",
        ),
        ("T_FLOAT32", ["tensor(float)"], "FLOAT32 scale tensors."),
        (
            "T_POST_QUANT",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "Post-quantization tensor types.",
        ),
        ("T_ANTIQUANT", float_types, "Antiquantization tensor types."),
        ("T_INT32", ["tensor(int32)"], "INT32 metadata tensors."),
        (
            "T_PREFIX",
            [
                "tensor(float16)",
                "tensor(bfloat16)",
                "tensor(float)",
                "tensor(int8)",
                "tensor(int4)",
            ],
            "Shared-prefix tensor types.",
        ),
        (
            "T_QUERY_ROPE",
            ["tensor(float16)", "tensor(bfloat16)", "tensor(int8)"],
            "Query RoPE tensor types.",
        ),
        (
            "T_KEY_ROPE",
            ["tensor(float16)", "tensor(bfloat16)", "tensor(int8)", "tensor(int4)"],
            "Key RoPE tensor types.",
        ),
        (
            "T_SINK",
            ["tensor(float16)", "tensor(bfloat16)"],
            "Learnable sink tensor types.",
        ),
        (
            "T_ATTENTION_OUT",
            ["tensor(float16)", "tensor(bfloat16)", "tensor(int8)"],
            "Attention output tensor types.",
        ),
    ]


def _fia_attributes() -> list[OpSchema.Attribute]:
    return [
        _required_attribute("num_heads", OpSchema.AttrType.INT, "Query head count."),
        _default_attribute("scale", 1.0, "Attention score scale."),
        _default_attribute("pre_tokens", 2147483647, "Previous-token window."),
        _default_attribute("next_tokens", 2147483647, "Next-token window."),
        _default_attribute("input_layout", "BSH", "Input and output tensor layout."),
        _default_attribute("num_key_value_heads", 0, "Key/value head count."),
        _default_attribute("sparse_mode", 0, "Sparse attention mode."),
        _default_attribute("inner_precise", 1, "Inner precision mode."),
        _default_attribute("block_size", 0, "Paged-attention block size."),
        _default_attribute("antiquant_mode", 0, "Antiquantization mode."),
        _default_attribute("softmax_lse_flag", False, "Whether to produce softmax LSE."),
        _default_attribute("key_antiquant_mode", 0, "Key antiquantization mode."),
        _default_attribute("value_antiquant_mode", 0, "Value antiquantization mode."),
        _default_attribute("query_quant_mode", 0, "Query quantization mode."),
        _default_attribute("pse_type", 0, "PSE representation type."),
        _default_attribute("out_dtype", 0, "Output dtype selector."),
    ]


def create_fused_infer_attention_score_schema() -> OpSchema:
    """Create the FIA schema frozen from the recorded CANN master source."""
    return OpSchema(
        FUSED_INFER_ATTENTION_SCORE_OP,
        "",
        MDC_ONNX_OPSET,
        doc=(
            "CANN FusedInferAttentionScore ABI frozen from "
            f"{CANN_FIA_SOURCE_COMMIT}."
        ),
        inputs=_fia_inputs(),
        outputs=[
            _parameter("attention_out", "T_ATTENTION_OUT"),
            _parameter("softmax_lse", "tensor(float)"),
        ],
        type_constraints=_fia_type_constraints(),
        attributes=_fia_attributes(),
    )


def create_moe_expert_schema() -> OpSchema:
    """Create the fully quantized routed-expert schema."""
    return OpSchema(
        MOE_EXPERT_OP,
        "",
        MDC_ONNX_OPSET,
        doc="Fully quantized MDC routed SwiGLU expert operator.",
        inputs=[
            _parameter("x", "T_INT8"),
            _parameter("topk_ids", "T_INT16"),
            _parameter("topk_weight", "T_FLOAT16"),
            _parameter("expert_weights", "T_INT8"),
            _parameter("quant_scales", "T_FLOAT32"),
        ],
        outputs=[_parameter("out", "T_FLOAT16")],
        type_constraints=[
            ("T_INT8", ["tensor(int8)"], "INT8 tensors."),
            ("T_INT16", ["tensor(int16)"], "INT16 tensors."),
            ("T_FLOAT16", ["tensor(float16)"], "FLOAT16 tensors."),
            ("T_FLOAT32", ["tensor(float)"], "FLOAT32 tensors."),
        ],
    )


SCHEMA_FACTORIES = MappingProxyType(
    {
        ASCEND_QUANT_OP: create_ascend_quant_schema,
        ASCEND_DEQUANT_OP: create_ascend_dequant_schema,
        RMS_NORM_OP: create_rms_norm_schema,
        ROTARY_POSITION_EMBEDDING_OP: create_rotary_position_embedding_schema,
        FUSED_INFER_ATTENTION_SCORE_OP: create_fused_infer_attention_score_schema,
        MOE_EXPERT_OP: create_moe_expert_schema,
    }
)

ALL_SCHEMA_NAMES: Final = tuple(SCHEMA_FACTORIES)
QUANTIZATION_SCHEMA_NAMES: Final = (ASCEND_QUANT_OP, ASCEND_DEQUANT_OP)


__all__ = [
    "ALL_SCHEMA_NAMES",
    "ASCEND_DEQUANT_OP",
    "ASCEND_QUANT_OP",
    "CANN_FIA_SOURCE_COMMIT",
    "CANN_FIA_SOURCE_URL",
    "FUSED_INFER_ATTENTION_SCORE_OP",
    "MDC_ONNX_OPSET",
    "MOE_EXPERT_OP",
    "QUANTIZATION_SCHEMA_NAMES",
    "RMS_NORM_OP",
    "ROTARY_POSITION_EMBEDDING_OP",
    "SCHEMA_FACTORIES",
    "create_ascend_dequant_schema",
    "create_ascend_quant_schema",
    "create_fused_infer_attention_score_schema",
    "create_moe_expert_schema",
    "create_rms_norm_schema",
    "create_rotary_position_embedding_schema",
]
