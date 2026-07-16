"""Conditional CUDA and NPU device-dispatch tests."""

from __future__ import annotations

import pytest
import torch

from mdc_llm_deploy.mdc_ops import (
    OPERATOR_SCHEMAS,
    apply_rotary_pos_emb,
    ascend_dequant,
    ascend_quant_v2,
    fused_infer_attention_score,
    moe_expert,
    operator_backend_status,
    registered_device_dispatches,
    rms_norm,
)


def _assert_on_device(value: torch.Tensor, device: torch.device) -> None:
    assert value.device.type == device.type
    if device.index is not None:
        assert value.device.index == device.index


def _device_smoke(device: torch.device) -> None:
    normalized, _ = rms_norm(
        torch.ones(2, 4, device=device), torch.ones(4, device=device)
    )
    _assert_on_device(normalized, device)

    query = torch.ones(1, 2, 2, 4, device=device)
    key = torch.ones(1, 2, 1, 4, device=device)
    cos = torch.ones(1, 2, 1, 4, device=device)
    rope_query, _ = apply_rotary_pos_emb(query, key, cos, torch.zeros_like(cos))
    _assert_on_device(rope_query, device)

    attention, _ = fused_infer_attention_score(
        query.transpose(1, 2),
        key.transpose(1, 2),
        key.transpose(1, 2),
        scale=0.5,
    )
    _assert_on_device(attention, device)

    quantized = ascend_quant_v2(query, torch.tensor(2.0, device=device))
    _assert_on_device(quantized, device)

    scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    encoded = (scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).to(torch.uint64)
    dequantized = ascend_dequant(
        torch.ones(1, 4, dtype=torch.int32, device=device), encoded
    )
    _assert_on_device(dequantized, device)

    moe_output = moe_expert(
        torch.ones(1, 2, dtype=torch.int8, device=device),
        torch.tensor([[0, 1, 4]], dtype=torch.int16, device=device),
        torch.tensor([[0.5, 0.5, 1.0]], dtype=torch.float16, device=device),
        torch.ones(5 * 3 * 2 * 2, dtype=torch.int8, device=device),
        torch.ones(21, dtype=torch.float32, device=device),
        torch.zeros(21, dtype=torch.int32, device=device),
    )
    _assert_on_device(moe_output, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_dispatch_for_all_six_operators() -> None:
    assert "CUDA" in registered_device_dispatches()
    _device_smoke(torch.device("cuda"))


def _npu_available() -> bool:
    backend = getattr(torch, "npu", None)
    return bool(backend is not None and backend.is_available())


@pytest.mark.skipif(not _npu_available(), reason="NPU is unavailable")
def test_npu_dispatch_for_all_six_operators() -> None:
    assert "PrivateUse1" in registered_device_dispatches()
    _device_smoke(torch.device("npu"))


def test_cpu_and_meta_dispatch_are_always_registered() -> None:
    assert {"CPU", "Meta"} <= set(registered_device_dispatches())


def test_backend_status_never_mislabels_reference_as_accelerated() -> None:
    registered = set(registered_device_dispatches())
    for schema in OPERATOR_SCHEMAS.values():
        statuses = operator_backend_status(schema.torch_name)
        assert [item.dispatch_key for item in statuses] == [
            "CPU",
            "CUDA",
            "PrivateUse1",
        ]
        assert all(item.operator == schema.torch_name for item in statuses)
        for item in statuses:
            assert item.registered == (item.dispatch_key in registered)
            expected = "reference" if item.registered else "unavailable"
            assert item.implementation == expected


def test_backend_status_rejects_unknown_operator() -> None:
    with pytest.raises(KeyError, match="Unknown MDC Torch operator"):
        operator_backend_status("FusedInferAttentionScore")
