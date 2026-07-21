from __future__ import annotations

from collections import Counter
from importlib.metadata import version

import onnx
import pytest
import torch
from onnx import TensorProto, ValueInfoProto

from .qwen3_export_fixtures import (
    EXPORT_CASES,
    AttentionBackend,
    Qwen3ExportCase,
    Qwen3Family,
    export_static_generation,
    export_static_prefill,
    onnx_export_config,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_ONNX_DTYPES = {
    torch.float16: TensorProto.FLOAT16,
    torch.bfloat16: TensorProto.BFLOAT16,
    torch.float32: TensorProto.FLOAT,
}
_COMMON_QWEN3_OPERATORS = frozenset(
    {
        "Cos",
        "MatMul",
        "ReduceMean",
        "Sin",
        "Softmax",
    }
)
_EXPORT_DEPENDENCIES = {
    "onnx": "1.22.0",
    "onnxscript": "0.7.1",
    "torch": "2.12.0",
    "transformers": "5.14.0",
}


@pytest.fixture(autouse=True)
def offline_export_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.mark.parametrize("case", EXPORT_CASES, ids=lambda case: case.id)
def test_static_prefill_export_contract(case: Qwen3ExportCase) -> None:
    model = export_static_prefill(case)

    _assert_standard_static_onnx(model)
    _assert_qwen3_graph_pattern(model, case)
    assert _value_names(model.graph.input) == {
        "input_ids",
        "attention_mask",
        "position_ids",
    }
    assert _value_names(model.graph.output) == {"logits"}
    assert _shape(_value(model.graph.input, "input_ids")) == (1, 3)
    assert _shape(_value(model.graph.output, "logits")) == (1, 3, 32)
    assert _elem_type(_value(model.graph.output, "logits")) == _ONNX_DTYPES[case.dtype]


@pytest.mark.parametrize("case", EXPORT_CASES, ids=lambda case: case.id)
def test_static_generation_export_has_real_kv_cache(case: Qwen3ExportCase) -> None:
    components = export_static_generation(case)

    assert set(components) == {"prefill", "decode"}
    prefill = components["prefill"]
    decode = components["decode"]
    for model in components.values():
        _assert_standard_static_onnx(model)
        _assert_qwen3_graph_pattern(model, case)

    prefill_cache_outputs = _cache_values(prefill.graph.output)
    decode_cache_inputs = _cache_values(decode.graph.input)
    decode_cache_outputs = _cache_values(decode.graph.output)
    assert len(prefill_cache_outputs) == 2
    assert len(decode_cache_inputs) == 2
    assert len(decode_cache_outputs) == 2
    assert {_shape(value) for value in prefill_cache_outputs} == {(1, 2, 3, 8)}
    assert {_shape(value) for value in decode_cache_inputs} == {(1, 2, 3, 8)}
    assert {_shape(value) for value in decode_cache_outputs} == {(1, 2, 4, 8)}
    assert _shape(_value(decode.graph.input, "input_ids")) == (1, 1)
    assert _shape(_value(decode.graph.input, "attention_mask")) == (1, 4)
    assert _shape(_value(decode.graph.input, "position_ids")) == (1, 1)
    assert _elem_type(_value(decode.graph.output, "logits")) == _ONNX_DTYPES[case.dtype]


def test_onnx_export_config_is_fixed_static_opset_18() -> None:
    config = onnx_export_config()

    assert config.opset_version == 18
    assert config.optimize is True
    assert config.dynamic is False
    assert config.dynamic_shapes is None


@pytest.mark.parametrize("package,expected_version", _EXPORT_DEPENDENCIES.items())
def test_export_dependency_baseline(package: str, expected_version: str) -> None:
    assert version(package) == expected_version


def _assert_standard_static_onnx(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model)
    default_opsets = [opset.version for opset in model.opset_import if opset.domain == ""]
    assert default_opsets == [18]
    for value in (*model.graph.input, *model.graph.output):
        assert all(not dimension.dim_param for dimension in value.type.tensor_type.shape.dim)


def _assert_qwen3_graph_pattern(model: onnx.ModelProto, case: Qwen3ExportCase) -> None:
    operators = Counter(node.op_type for node in model.graph.node)
    assert operators.keys() >= _COMMON_QWEN3_OPERATORS
    assert operators["MatMul"] >= 7
    if case.attention_backend is AttentionBackend.SDPA:
        assert operators["Where"] >= 1
    if case.family is Qwen3Family.MOE_30B_A3B:
        assert operators["TopK"] >= 1
        assert operators["Sigmoid"] >= 1
        assert operators["MatMul"] >= 15
    else:
        assert operators["TopK"] == 0


def _value_names(values: list[ValueInfoProto]) -> set[str]:
    return {value.name for value in values}


def _value(values: list[ValueInfoProto], name: str) -> ValueInfoProto:
    return next(value for value in values if value.name == name)


def _cache_values(values: list[ValueInfoProto]) -> list[ValueInfoProto]:
    return [value for value in values if "past_key_values.layers.0" in value.name]


def _shape(value: ValueInfoProto) -> tuple[int, ...]:
    return tuple(dimension.dim_value for dimension in value.type.tensor_type.shape.dim)


def _elem_type(value: ValueInfoProto) -> int:
    return value.type.tensor_type.elem_type
