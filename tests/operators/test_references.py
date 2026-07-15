"""Independent numerical references for MDC operators."""

from __future__ import annotations

import torch
from torch.nn import functional

from mdc_llm_deploy.mdc_ops import (
    apply_rotary_pos_emb,
    ascend_dequant,
    ascend_quant_v2,
    fused_infer_attention_score,
    moe_expert,
    rms_norm,
)


def _encode_scale(values: list[float]) -> torch.Tensor:
    scale = torch.tensor(values, dtype=torch.float32)
    return (scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).to(torch.uint64)


def test_rms_norm_matches_independent_fp32_reference() -> None:
    generator = torch.Generator().manual_seed(20260714)
    base = torch.randn(2, 3, 8, generator=generator)
    x = base[..., ::2]
    gamma = torch.randn(4, generator=generator)

    actual, actual_rstd = rms_norm(x, gamma)
    expected_rstd = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6)
    expected = (x.float() * expected_rstd.unsqueeze(-1) * gamma.float()).to(x.dtype)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_rstd, expected_rstd)
    assert actual_rstd.dtype == torch.float32
    assert actual_rstd.shape == (2, 3)


def test_rope_half_matches_independent_fp32_reference() -> None:
    generator = torch.Generator().manual_seed(20260714)
    query = torch.rand(1, 3, 4, 8, generator=generator) * 4 - 2
    key = torch.rand(1, 3, 2, 8, generator=generator) * 4 - 2
    angle = torch.rand(1, 3, 1, 8, generator=generator)
    cos, sin = angle.cos(), angle.sin()

    query_out, key_out = apply_rotary_pos_emb(query, key, cos, sin)

    def reference(value: torch.Tensor) -> torch.Tensor:
        first, second = value.float().chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return (value.float() * cos.float() + rotated * sin.float()).to(value.dtype)

    torch.testing.assert_close(query_out, reference(query))
    torch.testing.assert_close(key_out, reference(key))


def test_attention_true_mask_means_blocked_and_gqa_matches_reference() -> None:
    query = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[1.0, -1.0]]]]
    )
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]]])
    value = torch.tensor([[[[2.0, 3.0], [7.0, 11.0]], [[13.0, 17.0], [19.0, 23.0]]]])
    mask = torch.tensor([[[[False, True]]]])

    actual, lse = fused_infer_attention_score(
        query,
        key,
        value,
        atten_mask=mask,
        scale=0.5,
        num_heads=4,
        num_key_value_heads=2,
        softmax_lse_flag=True,
    )

    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    scores = query.float() @ expanded_key.float().transpose(-2, -1) * 0.5
    scores = scores.masked_fill(mask, float("-inf"))
    expected = torch.softmax(scores, dim=-1) @ expanded_value.float()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(lse, torch.logsumexp(scores, dim=-1, keepdim=True))
    torch.testing.assert_close(actual[..., 0, :], expanded_value[..., 0, :])


def test_attention_int8_antiquant_and_score_quantization() -> None:
    query = torch.tensor([[[[2, -2]]]], dtype=torch.int8)
    key = torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.int8)
    value = torch.tensor([[[[4, 8], [12, 16]]]], dtype=torch.int8)
    query_scale = torch.tensor(0.5)
    key_scale = torch.tensor(0.25)
    value_scale = torch.tensor(0.5)
    score_scale = torch.tensor(16.0)

    actual, _ = fused_infer_attention_score(
        query,
        key,
        value,
        scale=1.0,
        dequant_scale_query=query_scale,
        key_antiquant_scale=key_scale,
        value_antiquant_scale=value_scale,
        quant_scale1=score_scale,
    )

    scores = (query.float() * query_scale) @ (key.float() * key_scale).transpose(-2, -1)
    probability = torch.softmax(scores, dim=-1)
    probability = torch.round(probability * score_scale).clamp(-128, 127) / score_scale
    expected = (probability @ (value.float() * value_scale)).to(torch.float16)
    torch.testing.assert_close(actual, expected)


def test_quant_uses_multiplication_scale_ties_to_even_and_axis() -> None:
    x = torch.tensor(
        [[0.25, 0.75], [-0.25, -0.75], [100.0, -100.0]], dtype=torch.float32
    )
    scale = torch.tensor([2.0, 2.0, 4.0], dtype=torch.float32)
    offset = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    actual = ascend_quant_v2(x, scale, offset, axis=0)
    expected = torch.round(x * scale[:, None] + offset[:, None]).clamp(-128, 127)

    torch.testing.assert_close(actual, expected.to(torch.int8))
    torch.testing.assert_close(
        ascend_quant_v2(torch.tensor([-0.75, -0.25, 0.25, 0.75]), torch.tensor(2.0)),
        torch.tensor([-2, 0, 0, 2], dtype=torch.int8),
    )


def test_dequant_decodes_low_fp32_bits_and_attributes() -> None:
    x = torch.tensor([[-4, 8], [12, -16]], dtype=torch.int32)
    encoded = _encode_scale([0.5, 0.25])

    actual = ascend_dequant(x, encoded, relu_flag=True, dtype=1)
    expected = (x.float() * torch.tensor([0.5, 0.25])).relu().half()

    torch.testing.assert_close(actual, expected)


def _independent_moe_reference(
    x: torch.Tensor,
    ids: torch.Tensor,
    route_weights: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    intermediate_size: int,
) -> torch.Tensor:
    hidden_size = x.shape[1]
    hidden = (x.float() - offsets[0]) * scales[0]
    result = torch.zeros_like(hidden)
    cursor = 0
    for expert_id in range(5):
        base = 1 + expert_id * 4
        matrices = []
        for rows, columns, parameter_index in (
            (intermediate_size, hidden_size, base),
            (intermediate_size, hidden_size, base + 1),
            (hidden_size, intermediate_size, base + 3),
        ):
            length = rows * columns
            quantized = packed_weights[cursor : cursor + length].reshape(rows, columns)
            matrices.append((quantized.float() - offsets[parameter_index]) * scales[parameter_index])
            cursor += length
        gate, up, down = matrices
        activation = functional.silu(hidden @ gate.T) * (hidden @ up.T)
        activation_quantized = torch.round(
            activation / scales[base + 2] + offsets[base + 2]
        ).clamp(-128, 127)
        activation = (
            activation_quantized - offsets[base + 2]
        ) * scales[base + 2]
        expert_output = activation @ down.T
        selected_weight = (
            (ids == expert_id).to(route_weights.dtype) * route_weights
        ).sum(dim=1, keepdim=True)
        result += expert_output * selected_weight
    return result.half()


def test_moe_five_experts_and_21_parameter_order_match_reference() -> None:
    generator = torch.Generator().manual_seed(20260714)
    hidden_size, intermediate_size = 2, 3
    x = torch.randint(-4, 5, (3, hidden_size), dtype=torch.int8, generator=generator)
    ids = torch.tensor([[0, 1, 4], [2, 3, 4], [1, 3, 4]], dtype=torch.int16)
    route_weights = torch.tensor(
        [[0.25, 0.75, 1.0], [0.5, 0.5, 1.0], [0.75, 0.25, 1.0]],
        dtype=torch.float16,
    )
    packed_count = 5 * 3 * hidden_size * intermediate_size
    packed = torch.randint(-3, 4, (packed_count,), dtype=torch.int8, generator=generator)
    scales = torch.linspace(0.05, 0.25, 21)
    offsets = torch.arange(21, dtype=torch.int32) % 3 - 1

    actual = moe_expert(x, ids, route_weights, packed, scales, offsets)
    expected = _independent_moe_reference(
        x, ids, route_weights, packed, scales, offsets, intermediate_size
    )

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
