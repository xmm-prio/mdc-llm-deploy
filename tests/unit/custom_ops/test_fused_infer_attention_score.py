from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from mdc_llm_deploy.custom_ops.fused_infer_attention_score import (
    ONNX_ATTRIBUTE_NAMES,
    PLUGIN,
    REGISTERED_OPERATOR,
    TORCH_INPUT_SLOTS,
    attention_kernel,
    fake_attention,
    fused_infer_attention_score,
)
from mdc_llm_deploy.custom_ops.fused_infer_attention_score.onnx import translate


def _inputs(
    *,
    dtype: torch.dtype = torch.float32,
    query_sequence: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(2, 4, query_sequence, 8, dtype=dtype),
        torch.randn(2, 2, 5, 8, dtype=dtype),
        torch.randn(2, 2, 5, 8, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_torch_kernel_supports_prefill_gqa_mask_and_lse(dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    query, key, value = _inputs(dtype=dtype)
    mask = torch.zeros(2, 1, 3, 5, dtype=torch.bool)
    mask[..., -1] = True
    scale = 1.0 / math.sqrt(8)

    output, lse = fused_infer_attention_score(
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
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    scores = torch.matmul(query.float(), expanded_key.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(torch.broadcast_to(mask, scores.shape), -torch.inf)
    expected = torch.matmul(torch.softmax(scores, dim=-1), expanded_value.float()).to(dtype)

    tolerance = 2e-2 if dtype != torch.float32 else 1e-5
    torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(lse, torch.logsumexp(scores, dim=-1, keepdim=True))
    assert lse.dtype == torch.float32


def test_torch_kernel_supports_bsnd_and_actual_lengths() -> None:
    query = torch.randn(2, 1, 4, 8)
    key = torch.randn(2, 5, 2, 8)
    value = torch.randn(2, 5, 2, 8)

    output, lse = attention_kernel(
        query,
        key,
        value,
        actual_seq_lengths=torch.tensor([1, 0]),
        actual_seq_lengths_kv=torch.tensor([5, 3]),
        num_heads=4,
        num_key_value_heads=2,
        input_layout="BSND",
        softmax_lse_flag=True,
    )

    assert output.shape == query.shape
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    assert torch.isinf(lse[1]).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block_table": torch.zeros(1, dtype=torch.int32)}, "block_table"),
        ({"input_layout": "TND"}, "BNSD and BSND"),
        ({"sparse_mode": 2}, "sparse_mode"),
        ({"num_heads": 3, "num_key_value_heads": 2}, "query head axis"),
    ],
)
def test_torch_kernel_rejects_invalid_broad_contract(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    query, key, value = _inputs(query_sequence=1)
    arguments: dict[str, Any] = {"num_heads": 4}
    arguments.update(kwargs)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        attention_kernel(query, key, value, **arguments)


def test_fake_and_opcheck_cover_registered_contract() -> None:
    query, key, value = _inputs(dtype=torch.float16, query_sequence=1)
    with FakeTensorMode() as mode:
        fake_inputs = tuple(mode.from_tensor(tensor) for tensor in (query, key, value))
        output, lse = fake_attention(
            *fake_inputs,
            num_heads=4,
            num_key_value_heads=2,
            softmax_lse_flag=True,
        )
    assert output.shape == query.shape
    assert lse.shape == (2, 4, 1, 1)

    torch.library.opcheck(
        fused_infer_attention_score,
        (query, key, value),
        {"num_heads": 4, "num_key_value_heads": 2},
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )


def test_plugin_keeps_torch_and_onnx_contracts_separate() -> None:
    assert REGISTERED_OPERATOR.plugin is PLUGIN
    assert len(TORCH_INPUT_SLOTS) == 29
    assert PLUGIN.onnx.schema.name == "FusedInferAttentionScore"
    assert len(PLUGIN.onnx.schema.inputs) == 31
    assert len(PLUGIN.onnx.schema.outputs) == 2
    assert set(PLUGIN.onnx.schema.attributes) == ONNX_ATTRIBUTE_NAMES


class _OnnxValue:
    def __init__(self, shape: tuple[int | None, ...], dtype: str = "FLOAT16") -> None:
        self.shape = shape
        self.dtype = dtype


def test_onnx_contract_rejects_torch_legal_prefill_float32_and_mask() -> None:
    fp16_prefill = _OnnxValue((1, 4, 3, 8))
    fp16_kv = _OnnxValue((1, 2, 5, 8))
    with pytest.raises(RuntimeError, match="query sequence length S=1"):
        translate(
            fp16_prefill,
            fp16_kv,
            fp16_kv,
            num_heads=4,
            num_key_value_heads=2,
        )

    fp32_query = _OnnxValue((1, 4, 1, 8), "FLOAT")
    fp32_kv = _OnnxValue((1, 2, 5, 8), "FLOAT")
    with pytest.raises(RuntimeError, match="only FLOAT16 and BFLOAT16"):
        translate(fp32_query, fp32_kv, fp32_kv, num_heads=4, num_key_value_heads=2)

    fp16_query = _OnnxValue((1, 4, 1, 8))
    with pytest.raises(RuntimeError, match="optional inputs: atten_mask"):
        translate(
            fp16_query,
            fp16_kv,
            fp16_kv,
            atten_mask=object(),
            num_heads=4,
            num_key_value_heads=2,
        )


@pytest.mark.parametrize(
    ("key_shape", "value_shape", "num_heads", "kv_heads", "message"),
    [
        ((1, 2, 5, 8), (1, 2, 6, 8), 4, 2, "matching key/value"),
        ((1, 2, 5, 16), (1, 2, 5, 16), 4, 2, "head dimensions"),
        ((1, 2, 5, 8), (1, 2, 5, 8), 8, 2, "match query heads"),
        ((1, 3, 5, 8), (1, 3, 5, 8), 4, 3, "GQA"),
    ],
)
def test_onnx_contract_rejects_invalid_kv_and_head_metadata(
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    num_heads: int,
    kv_heads: int,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        translate(
            _OnnxValue((1, 4, 1, 8)),
            _OnnxValue(key_shape),
            _OnnxValue(value_shape),
            num_heads=num_heads,
            num_key_value_heads=kv_heads,
        )
