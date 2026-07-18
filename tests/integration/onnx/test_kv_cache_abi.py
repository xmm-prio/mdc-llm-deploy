"""End-to-end integration tests for the public KV cache artifact ABI."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

import onnx
import pytest
import torch
from onnx import TensorProto

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.export import convert_to_decode, export
from mdc_llm_deploy.graph.lifecycle import metadata
from mdc_llm_deploy.graph.metadata import (
    GraphMetadata,
    TensorAbi,
    derive_artifact_io_abi,
)
from mdc_llm_deploy.onnx import onnx_export, standard_onnx_export
from mdc_llm_deploy.onnx.transform.attention import attention_cache_dtype_overrides
from mdc_llm_deploy.onnx.validation.io import validate_io_abi
from mdc_llm_deploy.onnx.validation.model import validate_mdc_model
from mdc_llm_deploy.quantization import oneshot
from tests.support.models.qwen3 import dense_model, moe_model

pytestmark = pytest.mark.integration

_ONNX_DTYPES = {
    "float32": TensorProto.FLOAT,
    "int8": TensorProto.INT8,
    "int64": TensorProto.INT64,
}


@dataclass(frozen=True, slots=True)
class _Scenario:
    model_kind: Literal["dense", "moe"]
    layers: int
    quantized: bool
    save_kv_cache: bool
    decode: bool

    @property
    def id(self) -> str:
        precision = "int8" if self.quantized else "float"
        stage = "decode" if self.decode else "prefill"
        return (
            f"{self.model_kind}-{self.layers}l-{precision}-"
            f"{stage}-save-{self.save_kv_cache}"
        )


_SCENARIOS = (
    _Scenario("dense", 1, False, True, False),
    _Scenario("dense", 2, True, False, True),
    _Scenario("moe", 1, False, False, False),
    _Scenario("moe", 2, True, True, True),
)


class _Named(Protocol):
    @property
    def name(self) -> str: ...


def _cache_names(prefix: str, layers: int) -> tuple[str, ...]:
    return tuple(
        f"{prefix}.{layer_id}.{edge}"
        for layer_id in range(layers)
        for edge in ("key", "value")
    )


def _names(values: Iterable[_Named]) -> tuple[str, ...]:
    return tuple(item.name for item in values)


def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(
        dimension.dim_value
        for dimension in value.type.tensor_type.shape.dim
    )


def _assert_onnx_entries(
    actual: Iterable[onnx.ValueInfoProto],
    expected: tuple[TensorAbi, ...],
    *,
    dtype_overrides: dict[str, str] | None = None,
) -> None:
    entries = tuple(actual)
    assert _names(entries) == tuple(item.name for item in expected)
    overrides = dtype_overrides or {}
    for value_info, abi in zip(entries, expected, strict=True):
        expected_dtype = overrides.get(abi.name, abi.dtype)
        assert value_info.type.tensor_type.elem_type == _ONNX_DTYPES[expected_dtype]
        assert _shape(value_info) == abi.shape


def _assert_strict_decode_pairs(value: GraphMetadata) -> None:
    artifact = derive_artifact_io_abi(value)
    past = value.input_abi[1:]
    present = value.output_abi[1:]
    assert _names(past) == _cache_names("past", artifact.layer_count)
    assert _names(present) == _cache_names("present", artifact.layer_count)
    assert len(past) == len(present) == artifact.layer_count * 2
    for layer_id in range(artifact.layer_count):
        past_key, past_value = past[layer_id * 2 : layer_id * 2 + 2]
        present_key, present_value = present[layer_id * 2 : layer_id * 2 + 2]
        assert past_key.shape == past_value.shape
        assert present_key.shape == present_value.shape
        assert past_key.dtype == past_value.dtype == present_key.dtype == present_value.dtype
        assert past_key.shape[:2] == present_key.shape[:2]
        assert past_key.shape[2] + 1 == present_key.shape[2]
        assert past_key.shape[3] == present_key.shape[3]


def _make_graph(scenario: _Scenario) -> torch.fx.GraphModule:
    constructor = dense_model if scenario.model_kind == "dense" else moe_model
    model = constructor(4, layers=scenario.layers)
    model.export_config = replace(
        model.export_config,
        save_kv_cache=scenario.save_kv_cache,
    )
    inputs = {"input_ids": torch.arange(4).reshape(1, 4)}

    model_outputs = model(**inputs)
    assert len(model_outputs) == 1 + 2 * scenario.layers

    graph = export(model, inputs)
    if scenario.quantized:
        oneshot(
            graph,
            "configs/quantization/minmax-attention-a8.json",
            [inputs],
        )
    if scenario.decode:
        convert_to_decode(graph)
    return graph


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda item: item.id)
def test_model_fx_standard_and_mdc_share_kv_artifact_abi(
    tmp_path: Path,
    scenario: _Scenario,
) -> None:
    graph = _make_graph(scenario)
    value = metadata(graph)
    artifact = derive_artifact_io_abi(value)
    expected_internal_outputs = ("logits", *_cache_names("present", scenario.layers))

    assert _names(value.output_abi) == expected_internal_outputs
    assert artifact.layer_count == scenario.layers
    assert artifact.save_kv_cache is scenario.save_kv_cache
    if scenario.decode:
        _assert_strict_decode_pairs(value)
    else:
        assert _names(value.input_abi) == ("input_ids",)

    standard = standard_onnx_export(
        graph,
        tmp_path / f"{scenario.id}-standard.onnx",
        external_data=False,
    )
    mdc = onnx_export(
        graph,
        tmp_path / f"{scenario.id}-mdc.onnx",
        external_data=False,
    )

    if scenario.model_kind == "dense":
        onnx.checker.check_model(standard, full_check=True)
    validate_mdc_model(
        mdc,
        value,
        output_dtype_overrides=attention_cache_dtype_overrides(value),
    )
    _assert_onnx_entries(standard.graph.input, artifact.inputs)
    _assert_onnx_entries(standard.graph.output, artifact.outputs)
    _assert_onnx_entries(mdc.graph.input, artifact.inputs)
    _assert_onnx_entries(
        mdc.graph.output,
        artifact.outputs,
        dtype_overrides=attention_cache_dtype_overrides(value),
    )
    assert _names(standard.graph.output) == _names(mdc.graph.output)

    for model in (standard, mdc):
        produced = {
            output for node in model.graph.node for output in node.output
        }
        assert set(expected_internal_outputs[1:]) <= produced


def test_twelve_layers_keep_numeric_order_through_both_onnx_paths(
    tmp_path: Path,
) -> None:
    scenario = _Scenario("dense", 12, False, True, False)
    graph = _make_graph(scenario)
    value = metadata(graph)
    expected_outputs = ("logits", *_cache_names("present", 12))
    attention_fqns = tuple(
        boundary.fqn
        for boundary in value.boundaries
        if boundary.kind == "attention"
    )

    assert attention_fqns == tuple(
        f"model.layers.{layer_id}.self_attn" for layer_id in range(12)
    )
    assert attention_fqns.index("model.layers.2.self_attn") < attention_fqns.index(
        "model.layers.10.self_attn"
    )

    standard = standard_onnx_export(
        graph,
        tmp_path / "twelve-standard.onnx",
        external_data=False,
    )
    mdc = onnx_export(
        graph,
        tmp_path / "twelve-mdc.onnx",
        external_data=False,
    )

    assert _names(standard.graph.output) == expected_outputs
    assert _names(mdc.graph.output) == expected_outputs
    assert [
        node.name.removeprefix("mdc.attention.model.layers.").split(".", 1)[0]
        for node in mdc.graph.node
        if node.op_type == "FusedInferAttentionScore"
    ] == [str(layer_id) for layer_id in range(12)]
    onnx.checker.check_model(standard, full_check=True)
    validate_mdc_model(mdc, value)


def test_real_artifact_validation_rejects_broken_cache_pairs(
    tmp_path: Path,
) -> None:
    graph = _make_graph(_Scenario("dense", 2, False, True, True))
    value = metadata(graph)
    exported = standard_onnx_export(
        graph,
        tmp_path / "validation-source.onnx",
        external_data=False,
    )

    def remove_output(model: onnx.ModelProto) -> None:
        model.graph.output.pop()

    def swap_pair(model: onnx.ModelProto) -> None:
        first = onnx.ValueInfoProto()
        first.CopyFrom(model.graph.output[1])
        model.graph.output[1].CopyFrom(model.graph.output[2])
        model.graph.output[2].CopyFrom(first)

    def change_shape(model: onnx.ModelProto) -> None:
        model.graph.output[1].type.tensor_type.shape.dim[2].dim_value = 99

    def change_dtype(model: onnx.ModelProto) -> None:
        model.graph.output[1].type.tensor_type.elem_type = TensorProto.INT8

    for mutation in (remove_output, swap_pair, change_shape, change_dtype):
        candidate = copy.deepcopy(exported)
        mutation(candidate)
        with pytest.raises(OnnxExportError):
            validate_io_abi(candidate, value)


def test_repeated_exports_are_deterministic(tmp_path: Path) -> None:
    graph = _make_graph(_Scenario("dense", 1, False, True, False))

    for exporter in (standard_onnx_export, onnx_export):
        first = exporter(
            graph,
            tmp_path / f"{exporter.__name__}-first.onnx",
            external_data=False,
        )
        second = exporter(
            graph,
            tmp_path / f"{exporter.__name__}-second.onnx",
            external_data=False,
        )
        assert first.SerializeToString(
            deterministic=True
        ) == second.SerializeToString(deterministic=True)
