from __future__ import annotations

import math

import pytest
import torch

from mdc_llm_deploy.custom_ops.fused_infer_attention_score import (
    FusedInferAttentionScore,
)


def _manual_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    repeats = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(repeats, dim=1)
    expanded_value = value.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(query.float(), expanded_key.float().transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores.masked_fill(torch.broadcast_to(mask, scores.shape), -torch.inf)
    return (
        torch.matmul(torch.softmax(scores, dim=-1), expanded_value.float()).to(query.dtype),
        torch.logsumexp(scores, dim=-1, keepdim=True),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpu_bnsd_prefill_supports_gqa_broadcast_mask_and_lse(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(7)
    query = torch.randn(2, 4, 3, 8, dtype=dtype)
    key = torch.randn(2, 2, 5, 8, dtype=dtype)
    value = torch.randn(2, 2, 5, 8, dtype=dtype)
    mask = torch.zeros(2, 1, 3, 5, dtype=torch.bool)
    mask[:, :, :, -1] = True
    scale = 1.0 / math.sqrt(8)

    output, lse = FusedInferAttentionScore.cpu(
        query,
        key,
        value,
        atten_mask=mask,
        num_heads=4,
        num_key_value_heads=2,
        scale=scale,
        input_layout="BNSD",
        softmax_lse_flag=True,
    )
    expected_output, expected_lse = _manual_attention(query, key, value, mask, scale)

    tolerance = 2e-2 if dtype != torch.float32 else 1e-5
    torch.testing.assert_close(output, expected_output, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(lse, expected_lse, atol=1e-5, rtol=1e-5)
    assert lse.dtype == torch.float32


def test_cpu_bsnd_decode_honors_actual_sequence_lengths() -> None:
    query = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 1.0], [1.0, -1.0]]],
        ]
    )
    key = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]],
        ]
    )
    value = key * 2

    output, lse = FusedInferAttentionScore.cpu(
        query,
        key,
        value,
        actual_seq_lengths=torch.tensor([1, 0], dtype=torch.int64),
        actual_seq_lengths_kv=torch.tensor([1, 2], dtype=torch.int64),
        num_heads=2,
        num_key_value_heads=2,
        input_layout="BSND",
        softmax_lse_flag=True,
    )

    torch.testing.assert_close(output[0, 0], value[0, 0])
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    assert torch.isfinite(lse[0]).all()
    assert torch.isinf(lse[1]).all()


def test_cpu_without_lse_returns_documented_placeholder() -> None:
    query = torch.ones(1, 1, 1, 4)
    output, lse = FusedInferAttentionScore.cpu(
        query,
        query,
        query,
        num_heads=1,
        input_layout="BNSD",
    )

    assert output.shape == query.shape
    assert lse.shape == (1,)
    assert lse.dtype == torch.float32
    assert lse.item() == 0.0


def test_cpu_rejects_fully_masked_active_row() -> None:
    tensor = torch.ones(1, 1, 2, 4)

    with pytest.raises(ValueError, match="visible key"):
        FusedInferAttentionScore.cpu(
            tensor,
            tensor,
            tensor,
            atten_mask=torch.ones(1, 1, 2, 2, dtype=torch.bool),
            num_heads=1,
            input_layout="BNSD",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "block_table": torch.zeros(1, dtype=torch.int32),
                "input_layout": "BNSD",
            },
            "block_table",
        ),
        ({"input_layout": "TND"}, "BNSD and BSND"),
        ({"sparse_mode": 2, "input_layout": "BNSD"}, "sparse_mode"),
    ],
)
def test_cpu_rejects_out_of_scope_features(
    kwargs: dict[str, object],
    message: str,
) -> None:
    tensor = torch.ones(1, 1, 1, 4)

    with pytest.raises(NotImplementedError, match=message):
        FusedInferAttentionScore.cpu(
            tensor,
            tensor,
            tensor,
            num_heads=1,
            **kwargs,
        )


def test_fake_reports_layout_independent_output_metadata() -> None:
    query = torch.empty(2, 3, 4, 8, device="meta", dtype=torch.bfloat16)
    key = torch.empty(2, 5, 2, 8, device="meta", dtype=torch.bfloat16)
    value = torch.empty_like(key)

    output, lse = FusedInferAttentionScore.fake(
        query,
        key,
        value,
        num_heads=4,
        num_key_value_heads=2,
        input_layout="BSND",
        softmax_lse_flag=True,
    )

    assert output.shape == query.shape
    assert output.dtype == query.dtype
    assert output.device.type == "meta"
    assert lse.shape == (2, 4, 3, 1)
    assert lse.dtype == torch.float32


def test_class_preserves_complete_onnx_contract() -> None:
    assert len(FusedInferAttentionScore.onnx_input_slots) == 29
    assert FusedInferAttentionScore.onnx_input_slots[:7] == (
        "query",
        "key",
        "value",
        "pse_shift",
        "atten_mask",
        "actual_seq_lengths",
        "actual_seq_lengths_kv",
    )
    assert set(FusedInferAttentionScore.onnx_attribute_defaults) == {
        "num_heads",
        "scale",
        "pre_tokens",
        "next_tokens",
        "input_layout",
        "num_key_value_heads",
        "sparse_mode",
        "inner_precise",
        "block_size",
        "antiquant_mode",
        "softmax_lse_flag",
        "key_antiquant_mode",
        "value_antiquant_mode",
        "query_quant_mode",
    }


def test_onnx_rejects_torch_only_bsnd_and_lse() -> None:
    arguments = [object()] * 29

    with pytest.raises(RuntimeError, match="only BNSD"):
        FusedInferAttentionScore.onnx.__wrapped__(
            object(),
            *arguments,
            4,
            0.5,
            2_147_483_647,
            2_147_483_647,
            "BSND",
            2,
            0,
            0,
            0,
            0,
            False,
            0,
            0,
            0,
        )

    with pytest.raises(RuntimeError, match="softmax LSE"):
        FusedInferAttentionScore.onnx.__wrapped__(
            object(),
            *arguments,
            4,
            0.5,
            2_147_483_647,
            2_147_483_647,
            "BNSD",
            2,
            0,
            0,
            0,
            0,
            True,
            0,
            0,
            0,
        )


def test_onnx_rejects_unsupported_optional_slots() -> None:
    arguments: list[object | None] = [object(), object(), object()] + [None] * 26
    arguments[14] = object()

    with pytest.raises(RuntimeError, match="block_table"):
        FusedInferAttentionScore.onnx.__wrapped__(
            object(),
            *arguments,
            4,
            0.5,
            2_147_483_647,
            2_147_483_647,
            "BNSD",
            2,
            0,
            0,
            0,
            0,
            False,
            0,
            0,
            0,
        )
