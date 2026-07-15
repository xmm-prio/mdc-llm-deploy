"""Boundary and failure tests for MDC operator capsules."""

from __future__ import annotations

import pytest
import torch

from mdc_llm_deploy.mdc_ops import (
    apply_rotary_pos_emb,
    ascend_dequant,
    ascend_quant_v2,
    fused_infer_attention_score,
    moe_expert,
    rms_norm,
)


def _valid_moe_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.ones((1, 2), dtype=torch.int8),
        torch.tensor([[0, 1, 4]], dtype=torch.int16),
        torch.tensor([[0.5, 0.5, 1.0]], dtype=torch.float16),
        torch.ones(5 * 3 * 2 * 2, dtype=torch.int8),
        torch.ones(21, dtype=torch.float32),
        torch.zeros(21, dtype=torch.int32),
    )


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf")])
def test_rms_norm_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        rms_norm(torch.ones(2, 4), torch.ones(4), epsilon)


def test_rms_norm_rejects_shape_dtype_nonfinite_and_autograd() -> None:
    with pytest.raises(ValueError, match="gamma shape"):
        rms_norm(torch.ones(2, 4), torch.ones(3))
    with pytest.raises(TypeError, match="one dtype"):
        rms_norm(torch.ones(2, 4), torch.ones(4, dtype=torch.float16))
    with pytest.raises(ValueError, match="NaN or Inf"):
        rms_norm(torch.tensor([[float("nan")]]), torch.ones(1))
    with pytest.raises(RuntimeError, match="do not support autograd"):
        rms_norm(torch.ones(2, 4, requires_grad=True), torch.ones(4))


def test_rope_rejects_layout_mode_shape_and_nonfinite() -> None:
    query = torch.ones(1, 2, 2, 4)
    key = torch.ones(1, 2, 1, 4)
    cos = torch.ones(1, 2, 1, 4)
    sin = torch.zeros_like(cos)
    with pytest.raises(ValueError, match="layout"):
        apply_rotary_pos_emb(query, key, cos, sin, layout=0)
    with pytest.raises(ValueError, match="unsupported rotary_mode"):
        apply_rotary_pos_emb(query, key, cos, sin, rotary_mode="bad")
    with pytest.raises(ValueError, match="head dim"):
        apply_rotary_pos_emb(query[..., :3], key[..., :3], cos[..., :3], sin[..., :3])
    with pytest.raises(ValueError, match="broadcastable"):
        apply_rotary_pos_emb(query, key, torch.ones(1, 3, 1, 4), torch.ones(1, 3, 1, 4))


def test_attention_rejects_fully_masked_row_and_bad_heads() -> None:
    query = torch.ones(1, 4, 1, 2)
    key = torch.ones(1, 2, 2, 2)
    value = torch.ones_like(key)
    with pytest.raises(ValueError, match="fully masked"):
        fused_infer_attention_score(
            query, key, value, atten_mask=torch.ones(1, 1, 1, 2, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="Head attributes"):
        fused_infer_attention_score(query, key, value, num_heads=8)
    with pytest.raises(TypeError, match="Attention mask"):
        fused_infer_attention_score(
            query, key, value, atten_mask=torch.zeros(1, 1, 1, 2)
        )


def test_attention_rejects_missing_or_invalid_quant_parameters() -> None:
    query = torch.ones(1, 1, 1, 2, dtype=torch.int8)
    key = torch.ones(1, 1, 2, 2, dtype=torch.int8)
    value = torch.ones_like(key)
    with pytest.raises(ValueError, match="query requires scale"):
        fused_infer_attention_score(
            query,
            key,
            value,
            key_antiquant_scale=torch.tensor(1.0),
            value_antiquant_scale=torch.tensor(1.0),
        )
    with pytest.raises(ValueError, match="quant_scale1 must be positive"):
        fused_infer_attention_score(
            query,
            key,
            value,
            dequant_scale_query=torch.tensor(1.0),
            key_antiquant_scale=torch.tensor(1.0),
            value_antiquant_scale=torch.tensor(1.0),
            quant_scale1=torch.tensor(0.0),
        )


def test_quant_rejects_bad_scale_axis_dtype_and_autograd() -> None:
    x = torch.ones(2, 3)
    with pytest.raises(ValueError, match="positive"):
        ascend_quant_v2(x, torch.tensor(0.0))
    with pytest.raises(ValueError, match="axis"):
        ascend_quant_v2(x, torch.tensor(1.0), axis=2)
    with pytest.raises(ValueError, match="quantization axis"):
        ascend_quant_v2(x, torch.ones(2), axis=1)
    with pytest.raises(ValueError, match="dtype=2"):
        ascend_quant_v2(x, torch.tensor(1.0), dtype=29)
    with pytest.raises(RuntimeError, match="do not support autograd"):
        ascend_quant_v2(x.requires_grad_(), torch.tensor(1.0))


def test_dequant_rejects_high_bits_nonfinite_and_shape() -> None:
    x = torch.ones(2, 3, dtype=torch.int32)
    with pytest.raises(ValueError, match="high 32 bits"):
        ascend_dequant(x, torch.tensor([1 << 32], dtype=torch.uint64))
    nan_bits = torch.tensor([0x7FC00000], dtype=torch.uint64)
    with pytest.raises(ValueError, match="NaN or Inf"):
        ascend_dequant(x, nan_bits)
    with pytest.raises(ValueError, match="output channels"):
        ascend_dequant(x, torch.ones(2, dtype=torch.uint64))


@pytest.mark.parametrize(
    ("argument_index", "replacement", "message"),
    [
        (4, torch.ones(20), "21 ordered scales"),
        (1, torch.tensor([[1, 1, 4]], dtype=torch.int16), "unique"),
        (1, torch.tensor([[0, 1, 3]], dtype=torch.int16), "shared id 4"),
        (
            2,
            torch.tensor([[0.25, 0.5, 1.0]], dtype=torch.float16),
            "sum to one",
        ),
        (5, torch.zeros(20, dtype=torch.int32), "21-scale order"),
    ],
    ids=(
        "scale-count",
        "duplicate-expert-ids",
        "shared-expert-id",
        "route-weight-sum",
        "offset-count",
    ),
)
def test_moe_rejects_wrong_contract_and_route_values(
    argument_index: int,
    replacement: torch.Tensor,
    message: str,
) -> None:
    valid = list(_valid_moe_inputs())
    valid[argument_index] = replacement

    with pytest.raises(ValueError, match=message):
        moe_expert(*valid)
