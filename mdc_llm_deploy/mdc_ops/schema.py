"""Single source of truth for MDC operator schemas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

type AttributeValue = bool | float | int | str

MDC_ONNX_DOMAIN = "ai.onnx"
MDC_ONNX_OPSET = 18
TORCH_NAMESPACE = "mdc_llm_deploy"


@dataclass(frozen=True, slots=True)
class OperatorSchema:
    """Describe one MDC operator across Torch and ONNX."""

    ge_name: str
    onnx_name: str
    torch_name: str
    torch_schema: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: Mapping[str, AttributeValue]
    domain: str = MDC_ONNX_DOMAIN
    opset: int = MDC_ONNX_OPSET

    @property
    def qualified_torch_name(self) -> str:
        """Return namespace-qualified Torch operator name."""
        return f"{TORCH_NAMESPACE}::{self.torch_name}"

    @property
    def attribute_defaults(self) -> dict[str, AttributeValue]:
        """Return a mutable copy of ONNX attribute defaults."""
        return dict(self.attributes)


OPERATOR_SCHEMAS: dict[str, OperatorSchema] = {
    "RmsNorm": OperatorSchema(
        ge_name="RmsNorm",
        onnx_name="NPURmsNorm",
        torch_name="rms_norm",
        torch_schema=(
            "rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) -> (Tensor, Tensor)"
        ),
        inputs=("x", "gamma"),
        outputs=("y", "rstd"),
        attributes={"epsilon": 1e-6},
    ),
    "ApplyRotaryPosEmb": OperatorSchema(
        ge_name="ApplyRotaryPosEmb",
        onnx_name="ApplyRoPE",
        torch_name="apply_rotary_pos_emb",
        torch_schema=(
            "apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, "
            'int layout=1, str rotary_mode="half") -> (Tensor, Tensor)'
        ),
        inputs=("query", "key", "cos", "sin"),
        outputs=("query_out", "key_out"),
        attributes={"layout": 1, "rotary_mode": "half"},
    ),
    "FusedInferAttentionScore": OperatorSchema(
        ge_name="FusedInferAttentionScore",
        onnx_name="FusedInferAttentionScore",
        torch_name="fused_infer_attention_score",
        torch_schema=(
            "fused_infer_attention_score("
            "Tensor query, Tensor key, Tensor value, Tensor? atten_mask=None, "
            "float scale=1.0, int? num_heads=None, int? num_key_value_heads=None, "
            "Tensor? key_antiquant_scale=None, Tensor? key_antiquant_offset=None, "
            "Tensor? value_antiquant_scale=None, Tensor? value_antiquant_offset=None, "
            "Tensor? dequant_scale_query=None, Tensor? quant_scale1=None, "
            "bool softmax_lse_flag=False) -> (Tensor, Tensor)"
        ),
        inputs=(
            "query",
            "key",
            "value",
            "atten_mask",
            "key_antiquant_scale",
            "key_antiquant_offset",
            "value_antiquant_scale",
            "value_antiquant_offset",
            "dequant_scale_query",
            "quant_scale1",
        ),
        outputs=("attention_out", "softmax_lse"),
        attributes={
            "num_heads": 1,
            "scale": 1.0,
            "input_layout": "BNSD",
            "num_key_value_heads": 0,
            "sparse_mode": 0,
            "softmax_lse_flag": False,
        },
    ),
    "AscendQuantV2": OperatorSchema(
        ge_name="AscendQuantV2",
        onnx_name="NPUAscendQuantV2",
        torch_name="ascend_quant_v2",
        torch_schema=(
            "ascend_quant_v2(Tensor x, Tensor scale, Tensor? offset=None, "
            "int axis=-1, int dtype=2) -> Tensor"
        ),
        inputs=("x", "scale", "offset"),
        outputs=("y",),
        attributes={"axis": -1, "dtype": 2},
    ),
    "AscendDequant": OperatorSchema(
        ge_name="AscendDequant",
        onnx_name="AscendDequant",
        torch_name="ascend_dequant",
        torch_schema=(
            "ascend_dequant(Tensor x, Tensor deq_scale, bool sqrt_mode=False, "
            "bool relu_flag=False, int dtype=0) -> Tensor"
        ),
        inputs=("x", "deq_scale"),
        outputs=("y",),
        attributes={"sqrt_mode": False, "relu_flag": False, "dtype": 0},
    ),
    "MoeExpert": OperatorSchema(
        ge_name="MoeExpert",
        onnx_name="MoeExpert",
        torch_name="moe_expert",
        torch_schema=(
            "moe_expert(Tensor x, Tensor topk_ids, Tensor topk_weight, "
            "Tensor expert_weights, Tensor quant_scales, "
            "Tensor? quant_offsets=None) -> Tensor"
        ),
        inputs=(
            "x",
            "topk_ids",
            "topk_weight",
            "expert_weights",
            "quant_scales",
            "quant_offsets",
        ),
        outputs=("out",),
        attributes={},
    ),
}


def schema_for_torch_name(torch_name: str) -> OperatorSchema:
    """Return schema matching an unqualified Torch operator name."""
    for schema in OPERATOR_SCHEMAS.values():
        if schema.torch_name == torch_name:
            return schema
    raise KeyError(torch_name)
