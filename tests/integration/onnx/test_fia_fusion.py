from __future__ import annotations

import onnx
import pytest
import torch
from onnx import TensorProto, helper

from mdc_llm_deploy.onnx._graph import GraphIndex
from mdc_llm_deploy.onnx.fusion_pass import fuse_fused_infer_attention_score
from mdc_llm_deploy.onnx.schemas import (
    FUSED_INFER_ATTENTION_SCORE_OP,
    register_schemas,
)

from .qwen3_export_fixtures import (
    AttentionBackend,
    Qwen3ExportCase,
    Qwen3Family,
    export_static_generation,
    export_static_prefill,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_CASES = tuple(
    Qwen3ExportCase(family, backend, dtype)
    for family in Qwen3Family
    for backend in AttentionBackend
    for dtype in (*_SUPPORTED_DTYPES, torch.float32)
)


@pytest.fixture(autouse=True)
def offline_export_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.id)
def test_real_qwen3_prefill_and_cache_decode_fia_contract(case: Qwen3ExportCase) -> None:
    prefill = export_static_prefill(case)
    decode = export_static_generation(case)["decode"]
    original = {
        "prefill": prefill.SerializeToString(),
        "decode": decode.SerializeToString(),
    }

    results = {
        "prefill": fuse_fused_infer_attention_score(prefill),
        "decode": fuse_fused_infer_attention_score(decode),
    }

    if case.dtype is torch.float32:
        assert all(result.fused_count == 0 for result in results.values())
        assert prefill.SerializeToString() == original["prefill"]
        assert decode.SerializeToString() == original["decode"]
        return

    assert {name: result.fused_count for name, result in results.items()} == {
        "prefill": 1,
        "decode": 1,
    }
    _assert_real_fia(prefill, case, query_length=3, key_length=3)
    _assert_real_fia(decode, case, query_length=1, key_length=4)


def _assert_real_fia(
    model: onnx.ModelProto,
    case: Qwen3ExportCase,
    *,
    query_length: int,
    key_length: int,
) -> None:
    index = GraphIndex(model)
    fused = next(
        node for node in model.graph.node if node.op_type == FUSED_INFER_ATTENTION_SCORE_OP
    )
    attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in fused.attribute
    }
    expected_elem_type = (
        TensorProto.FLOAT16 if case.dtype is torch.float16 else TensorProto.BFLOAT16
    )

    assert len(fused.input) == 31
    assert len(fused.output) == 2
    assert index.tensor_info[fused.input[0]].shape == (1, 4, query_length, 8)
    assert index.tensor_info[fused.input[1]].shape == (1, 2, key_length, 8)
    assert index.tensor_info[fused.input[2]].shape == (1, 2, key_length, 8)
    assert all(
        index.tensor_info[name].elem_type == expected_elem_type
        for name in fused.input[:3]
    )
    assert fused.input[3] == ""
    assert fused.input[4]
    mask_producer = index.producer(fused.input[4])
    assert mask_producer is not None
    assert mask_producer.op_type == "Not"
    assert all(not name for name in fused.input[5:])
    assert attributes["input_layout"] == b"BNSD"
    assert attributes["num_heads"] == 4
    assert attributes["num_key_value_heads"] == 2
    assert attributes["scale"] == pytest.approx(_expected_scale(case), rel=1e-6)
    assert len(attributes) == 16
    if query_length == 1:
        assert "past_key_values.layers.0" in fused.input[1]
        assert "past_key_values.layers.0" in fused.input[2]

    register_schemas(FUSED_INFER_ATTENTION_SCORE_OP)
    onnx.checker.check_model(model, full_check=True)


def _expected_scale(case: Qwen3ExportCase) -> float:
    if case.attention_backend is AttentionBackend.EAGER:
        return 0.353515625
    if case.dtype is torch.float16:
        return 0.3536996841430664
    return 0.3525390625
