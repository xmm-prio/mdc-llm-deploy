"""Framework-independent MDC operator schema contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .attention_layout import (
    RELEASE_ATTENTION_ATTRIBUTES,
    AttentionInput,
)
from .onnx_protocol import MDC_ONNX_DOMAIN, MDC_ONNX_OPSET

AttributeValue = bool | float | int | str
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

    def __post_init__(self) -> None:
        """Freeze attribute defaults as an immutable snapshot."""
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType(dict(self.attributes)),
        )

    @property
    def qualified_torch_name(self) -> str:
        """Return namespace-qualified Torch operator name."""
        return f"{TORCH_NAMESPACE}::{self.torch_name}"

    @property
    def attribute_defaults(self) -> dict[str, AttributeValue]:
        """Return a mutable copy of ONNX attribute defaults."""
        return dict(self.attributes)


OPERATOR_SCHEMAS: Mapping[str, OperatorSchema] = MappingProxyType({
    "RmsNorm": OperatorSchema(
        ge_name="RmsNorm",
        onnx_name="NPURmsNorm",
        torch_name="rms_norm",
        torch_schema=(
            "rms_norm(Tensor x, Tensor gamma, "
            "float epsilon=1e-6) -> (Tensor, Tensor)"
        ),
        inputs=("x", "gamma"),
        outputs=("y", "rstd"),
        attributes={"epsilon": 1e-6},
    ),
    "ApplyRotaryPosEmb": OperatorSchema(
        ge_name="ApplyRotaryPosEmb",
        onnx_name="ApplyRotaryPosEmb",
        torch_name="apply_rotary_pos_emb",
        torch_schema=(
            "apply_rotary_pos_emb(Tensor query, Tensor key, "
            "Tensor cos, Tensor sin, int layout=1, "
            'str rotary_mode="half") -> (Tensor, Tensor)'
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
            "Tensor query, Tensor key, Tensor value, "
            "Tensor? atten_mask=None, float scale=1.0, "
            "int? num_heads=None, "
            "int? num_key_value_heads=None, "
            "Tensor? key_antiquant_scale=None, "
            "Tensor? key_antiquant_offset=None, "
            "Tensor? value_antiquant_scale=None, "
            "Tensor? value_antiquant_offset=None, "
            "Tensor? dequant_scale_query=None, "
            "Tensor? quant_scale1=None, "
            "bool softmax_lse_flag=False) -> (Tensor, Tensor)"
        ),
        inputs=tuple(
            slot.name.lower() for slot in AttentionInput
        ),
        outputs=("attention_out", "softmax_lse"),
        attributes={
            **RELEASE_ATTENTION_ATTRIBUTES,
            "num_heads": 1,
            "scale": 1.0,
            "num_key_value_heads": 0,
        },
    ),
    "AscendQuantV2": OperatorSchema(
        ge_name="AscendQuantV2",
        onnx_name="NPUAscendQuantV2",
        torch_name="ascend_quant_v2",
        torch_schema=(
            "ascend_quant_v2(Tensor x, Tensor scale, "
            "Tensor? offset=None, int axis=-1, int dtype=2) "
            "-> Tensor"
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
            "ascend_dequant(Tensor x, Tensor deq_scale, "
            "bool sqrt_mode=False, bool relu_flag=False, "
            "int dtype=0) -> Tensor"
        ),
        inputs=("x", "deq_scale"),
        outputs=("y",),
        attributes={
            "sqrt_mode": False,
            "relu_flag": False,
            "dtype": 0,
        },
    ),
    "MoeExpert": OperatorSchema(
        ge_name="MoeExpert",
        onnx_name="MoeExpert",
        torch_name="moe_expert",
        torch_schema=(
            "moe_expert(Tensor x, Tensor topk_ids, "
            "Tensor topk_weight, Tensor expert_weights, "
            "Tensor quant_scales, Tensor? quant_offsets=None) "
            "-> Tensor"
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
})

_ONNX_SCHEMA_INDEX = MappingProxyType(
    {
        schema.onnx_name: schema
        for schema in OPERATOR_SCHEMAS.values()
    }
)
_TORCH_SCHEMA_INDEX = MappingProxyType(
    {
        schema.torch_name: schema
        for schema in OPERATOR_SCHEMAS.values()
    }
)
if (
    len(_ONNX_SCHEMA_INDEX) != len(OPERATOR_SCHEMAS)
    or len(_TORCH_SCHEMA_INDEX) != len(OPERATOR_SCHEMAS)
):
    raise RuntimeError(
        "Operator schemas contain duplicate ONNX or Torch names"
    )


def schema_for_torch_name(torch_name: str) -> OperatorSchema:
    """Return schema matching an unqualified Torch operator name."""
    return _TORCH_SCHEMA_INDEX[torch_name]


def schema_for_onnx_name(onnx_name: str) -> OperatorSchema:
    """Return schema matching an ONNX operator name."""
    return _ONNX_SCHEMA_INDEX[onnx_name]


__all__ = [
    "OPERATOR_SCHEMAS",
    "TORCH_NAMESPACE",
    "AttributeValue",
    "OperatorSchema",
    "schema_for_onnx_name",
    "schema_for_torch_name",
]
