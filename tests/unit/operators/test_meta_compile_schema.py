"""Meta, FakeTensor, compile, export, and schema tests."""

from __future__ import annotations

import operator

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from mdc_llm_deploy.operators import (
    MDC_ONNX_DOMAIN,
    MDC_ONNX_OPSET,
    OPERATOR_SCHEMAS,
    apply_rotary_pos_emb,
    ascend_dequant,
    ascend_quant_v2,
    fused_infer_attention_score,
    moe_expert,
    rms_norm,
)
from mdc_llm_deploy.operators.contracts.attention import (
    RELEASE_ATTENTION_ATTRIBUTES,
    AttentionInput,
)
from mdc_llm_deploy.operators.contracts.schema import (
    OPERATOR_SCHEMAS as PROTOCOL_OPERATOR_SCHEMAS,
)
from mdc_llm_deploy.operators.contracts.schema import (
    schema_for_onnx_name,
    schema_for_torch_name,
)


def test_schema_is_single_opset_18_source() -> None:
    assert OPERATOR_SCHEMAS is PROTOCOL_OPERATOR_SCHEMAS
    assert set(OPERATOR_SCHEMAS) == {
        "RmsNorm",
        "ApplyRotaryPosEmb",
        "FusedInferAttentionScore",
        "AscendQuantV2",
        "AscendDequant",
        "MoeExpert",
    }
    assert {schema.opset for schema in OPERATOR_SCHEMAS.values()} == {18}
    assert {schema.domain for schema in OPERATOR_SCHEMAS.values()} == {"ai.onnx"}
    assert MDC_ONNX_OPSET == 18
    assert MDC_ONNX_DOMAIN == "ai.onnx"
    assert OPERATOR_SCHEMAS["AscendQuantV2"].attributes["axis"] == -1
    assert (
        schema_for_onnx_name("NPUAscendQuantV2")
        is schema_for_torch_name("ascend_quant_v2")
        is OPERATOR_SCHEMAS["AscendQuantV2"]
    )
    attention_schema = OPERATOR_SCHEMAS[
        "FusedInferAttentionScore"
    ]
    assert attention_schema.inputs == tuple(
        slot.name.lower() for slot in AttentionInput
    )
    assert all(
        attention_schema.attributes[name] == value
        for name, value in RELEASE_ATTENTION_ATTRIBUTES.items()
    )
    assert len(
        {schema.qualified_torch_name for schema in OPERATOR_SCHEMAS.values()}
    ) == 6
    assert len(
        {schema.onnx_name for schema in OPERATOR_SCHEMAS.values()}
    ) == 6
    with pytest.raises(KeyError):
        schema_for_onnx_name("Unknown")
    with pytest.raises(KeyError):
        schema_for_torch_name("unknown")


def test_operator_schema_registry_is_deeply_immutable() -> None:
    schema = OPERATOR_SCHEMAS["RmsNorm"]

    with pytest.raises(TypeError):
        operator.setitem(OPERATOR_SCHEMAS, "future", schema)
    with pytest.raises(TypeError):
        operator.setitem(schema.attributes, "epsilon", 0.0)

    copy = schema.attribute_defaults
    copy["epsilon"] = 0.0
    assert schema.attributes["epsilon"] == 1e-6


def test_meta_outputs_cover_all_operators() -> None:
    x = torch.empty(2, 4, device="meta")
    gamma = torch.empty(4, device="meta")
    normalized, rstd = rms_norm(x, gamma)
    assert normalized.shape == (2, 4)
    assert rstd.shape == (2,)
    assert rstd.dtype == torch.float32

    query = torch.empty(1, 2, 4, 8, device="meta")
    key = torch.empty(1, 2, 2, 8, device="meta")
    cos = torch.empty(1, 2, 1, 8, device="meta")
    rope_query, rope_key = apply_rotary_pos_emb(query, key, cos, cos)
    assert rope_query.shape == query.shape
    assert rope_key.shape == key.shape

    attention_query = torch.empty(1, 4, 3, 8, device="meta")
    attention_key = torch.empty(1, 2, 5, 8, device="meta")
    attention, lse = fused_infer_attention_score(
        attention_query,
        attention_key,
        torch.empty_like(attention_key),
        softmax_lse_flag=True,
    )
    assert attention.shape == (1, 4, 3, 8)
    assert lse.shape == (1, 4, 3, 1)
    assert lse.dtype == torch.float32

    quantized = ascend_quant_v2(x, torch.empty(4, device="meta"))
    assert quantized.shape == x.shape
    assert quantized.dtype == torch.int8

    dequantized = ascend_dequant(
        torch.empty(2, 4, dtype=torch.int32, device="meta"),
        torch.empty(4, dtype=torch.uint64, device="meta"),
        dtype=1,
    )
    assert dequantized.shape == (2, 4)
    assert dequantized.dtype == torch.float16

    moe = moe_expert(
        torch.empty(2, 4, device="meta"),
        torch.empty(2, 3, dtype=torch.int64, device="meta"),
        torch.empty(2, 3, device="meta"),
        torch.empty(5, 3 * 4 * 8, device="meta"),
    )
    assert moe.shape == (2, 4)
    assert moe.dtype == torch.float32


def test_fake_tensor_outputs_have_expected_metadata() -> None:
    mode = FakeTensorMode()
    with mode:
        x = torch.empty(2, 4)
        gamma = torch.empty(4)
        output, rstd = rms_norm(x, gamma)
        assert type(output).__name__ == "FakeTensor"
        assert output.shape == (2, 4)
        assert rstd.shape == (2,)


def test_torch_compile_executes_custom_operator() -> None:
    def function(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, gamma)[0]

    compiled = torch.compile(function, backend="eager", fullgraph=True)
    x = torch.randn(2, 4)
    gamma = torch.randn(4)
    torch.testing.assert_close(compiled(x, gamma), function(x, gamma))


class _RmsNormModule(nn.Module):
    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, gamma)[0]


def test_torch_export_preserves_custom_operator() -> None:
    exported = torch.export.export(_RmsNormModule(), (torch.randn(2, 4), torch.ones(4)))
    targets = {str(node.target) for node in exported.graph.nodes}
    assert "mdc_llm_deploy.rms_norm.default" in targets
