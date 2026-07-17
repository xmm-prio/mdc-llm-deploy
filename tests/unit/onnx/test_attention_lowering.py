"""Module-boundary tests for ONNX Attention lowering."""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.export import export
from mdc_llm_deploy.graph.metadata import (
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    QuantizedTarget,
)
from mdc_llm_deploy.onnx import api, onnx_export
from mdc_llm_deploy.onnx.transform import (
    attention as attention_lowering,
)
from mdc_llm_deploy.onnx.transform.cleanup import producer_map
from mdc_llm_deploy.onnx.transform.support import OnnxLoweringContext
from mdc_llm_deploy.operators.contracts.attention import AttentionInput
from mdc_llm_deploy.quantization import oneshot
from tests.support.models.qwen3 import dense_model


def _rms_norm_fixture(
    fqns: tuple[str, ...],
    *,
    reverse_boundaries: bool = False,
) -> tuple[onnx.ModelProto, GraphMetadata]:
    shape = (1, 2, 4)
    nodes: list[onnx.NodeProto] = []
    inputs: list[onnx.ValueInfoProto] = []
    outputs: list[onnx.ValueInfoProto] = []
    initializers: list[onnx.TensorProto] = []
    value_info: list[onnx.ValueInfoProto] = []
    boundaries = [FusionBoundary("rms_norm", fqn) for fqn in fqns]
    for index, fqn in enumerate(fqns):
        source = f"source.{index}"
        inverse = f"inverse.{index}"
        normalized = f"normalized.{index}"
        output = f"output.{index}"
        gamma = f"graph.{fqn}.weight"
        inputs.append(helper.make_tensor_value_info(source, TensorProto.FLOAT, shape))
        outputs.append(helper.make_tensor_value_info(output, TensorProto.FLOAT, shape))
        initializers.extend(
            [
                numpy_helper.from_array(np.ones((1,), dtype=np.float32), inverse),
                numpy_helper.from_array(np.ones((4,), dtype=np.float32), gamma),
            ]
        )
        value_info.append(
            helper.make_tensor_value_info(normalized, TensorProto.FLOAT, shape)
        )
        nodes.extend(
            [
                helper.make_node(
                    "Identity",
                    [source],
                    [f"padding.{index}"],
                    name=f"padding.{index}",
                ),
                helper.make_node(
                    "Mul",
                    [source, inverse],
                    [normalized],
                    name=f"normalized.{index}",
                ),
                helper.make_node(
                    "Mul",
                    [normalized, gamma],
                    [output],
                    name=f"terminal.{index}",
                ),
            ]
        )
    graph = helper.make_graph(
        nodes,
        "rms_norm_fixture",
        inputs,
        outputs,
        initializers,
    )
    graph.value_info.extend(value_info)
    metadata = GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="test",
        input_abi=(),
        output_abi=(),
        boundaries=tuple(reversed(boundaries)) if reverse_boundaries else tuple(boundaries),
    )
    return helper.make_model(graph), metadata


def _attention_target(fqn: str) -> QuantizedTarget:
    return QuantizedTarget(
        fqn=fqn,
        target_type="attention",
        algorithm="minmax",
        bits=8,
        granularity="per_tensor",
        symmetric=True,
        scale=(0.5,),
        zero_point=(0,),
    )


@pytest.mark.parametrize(
    ("attention_fqn", "target_fqn", "edge", "matches"),
    [
        ("self_attn", "self_attn.query", "query", True),
        ("self_attn", "layers.0.self_attn.query", "query", True),
        ("self_attn", "layers.0.self_attn2.query", "query", False),
        ("self_attn", "self_attn", "query", False),
        ("self_attn", "self_attn.key", "query", False),
        ("", "layers.0.self_attn.query", "query", False),
    ],
)
def test_attention_target_keeps_path_segment_matching(
    attention_fqn: str,
    target_fqn: str,
    edge: str,
    *,
    matches: bool,
) -> None:
    metadata = GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="test",
        input_abi=(),
        output_abi=(),
        quantized_targets=(_attention_target(target_fqn),),
    )

    result = attention_lowering._target(metadata, edge, attention_fqn)

    assert (result is not None) is matches


@pytest.mark.parametrize(
    ("rope_fqn", "accepted"),
    [
        ("attention.rope", True),
        ("attention2.rope", False),
        ("attention", False),
        ("block.attention.rope", False),
    ],
)
def test_api_lower_selects_only_strict_attention_descendant_rope(
    monkeypatch: pytest.MonkeyPatch,
    rope_fqn: str,
    *,
    accepted: bool,
) -> None:
    lowered: list[tuple[FusionBoundary, ...]] = []
    monkeypatch.setattr(
        api,
        "lower_rope_attention",
        lambda model, value, mask_mode, *, layer_id, context: lowered.append(
            value.boundaries
        ),
    )
    for name in (
        "append_quantized_linears",
        "adapt_quantized_moe",
        "prune_unreachable",
        "topologically_sort",
        "remove_dynamic_value_info",
    ):
        monkeypatch.setattr(api, name, lambda *args: None)
    metadata = GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="test",
        input_abi=(),
        output_abi=(),
        boundaries=(
            FusionBoundary("attention", "attention"),
            FusionBoundary("rope", rope_fqn),
        ),
    )
    model = helper.make_model(helper.make_graph([], "empty", [], []))

    if not accepted:
        with pytest.raises(
            OnnxExportError,
            match="requires one owned RoPE boundary",
        ):
            api._lower(model, metadata, "masked")
        assert lowered == []
        return

    api._lower(model, metadata, "masked")
    assert lowered == [
        (
            FusionBoundary("attention", "attention"),
            FusionBoundary("rope", rope_fqn),
        )
    ]


def test_attention_lowering_exposes_public_stage_entries_without_api_dependency() -> None:
    assert callable(attention_lowering.lower_maskless_attention)
    assert callable(attention_lowering.lower_rms_norms)
    assert callable(attention_lowering.lower_rope_attention)

    tree = ast.parse(inspect.getsource(attention_lowering))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "api" not in imported_modules


def test_api_preserves_attention_lowering_order() -> None:
    source = inspect.getsource(api._lower)

    assert source.index("lower_maskless_attention") < source.index("lower_rms_norms")
    assert source.index("lower_rms_norms") < source.index("lower_rope_attention")
    assert source.index("lower_rope_attention") < source.index("append_quantized_linears")


@pytest.mark.parametrize(
    ("occupied_by", "expected"),
    [
        ("initializer", "value.1"),
        ("input", "value.1"),
        ("node_output", "value.1"),
        ("graph_output", "value"),
        ("value_info", "value"),
    ],
)
def test_attention_uses_shared_context_graph_name_occupancy(
    occupied_by: str,
    expected: str,
) -> None:
    graph = helper.make_graph([], "names", [], [])
    if occupied_by == "initializer":
        graph.initializer.append(
            numpy_helper.from_array(np.ones((1,), dtype=np.float32), "value")
        )
    elif occupied_by == "input":
        graph.input.append(helper.make_tensor_value_info("value", TensorProto.FLOAT, (1,)))
    elif occupied_by == "node_output":
        graph.node.append(helper.make_node("Identity", ["source"], ["value"]))
    elif occupied_by == "graph_output":
        graph.output.append(helper.make_tensor_value_info("value", TensorProto.FLOAT, (1,)))
    else:
        graph.value_info.append(
            helper.make_tensor_value_info("value", TensorProto.FLOAT, (1,))
        )

    context = OnnxLoweringContext.from_model(helper.make_model(graph))

    assert context.unique_name("value") == expected


def test_attention_shared_context_reserves_smallest_suffix() -> None:
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["source"], ["value"]),
                helper.make_node("Identity", ["source"], ["value.1"]),
                helper.make_node("Identity", ["source"], ["value.3"]),
            ],
            "names",
            [],
            [],
        )
    )
    context = OnnxLoweringContext.from_model(model)

    assert context.unique_name("value") == "value.2"
    assert context.unique_name("value") == "value.4"


def _quantized_attention_graph() -> torch.fx.GraphModule:
    inputs = {"input_ids": torch.arange(4).reshape(1, 4)}
    graph = export(dense_model(4, layers=2), inputs)
    oneshot(
        graph,
        "configs/quantization/minmax-attention-a8.json",
        [inputs],
    )
    return graph


def test_quantized_attention_lowering_is_deterministic(
    tmp_path: Path,
) -> None:
    graph = _quantized_attention_graph()
    first = onnx_export(graph, tmp_path / "first.onnx", external_data=False)
    second = onnx_export(graph, tmp_path / "second.onnx", external_data=False)

    assert hashlib.sha256(
        first.SerializeToString(deterministic=True)
    ).digest() == hashlib.sha256(
        second.SerializeToString(deterministic=True)
    ).digest()


def test_quantized_attention_accepts_one_shared_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    graph = _quantized_attention_graph()
    real_lower = attention_lowering.lower_rope_attention
    observed: list[OnnxLoweringContext] = []

    def lower_with_shared_context(
        model: onnx.ModelProto,
        value: GraphMetadata,
        mask_mode: attention_lowering.MaskMode,
        *,
        layer_id: int,
        context: OnnxLoweringContext,
    ) -> None:
        observed.append(context)
        real_lower(
            model,
            value,
            mask_mode,
            layer_id=layer_id,
            context=context,
        )

    monkeypatch.setattr(api, "lower_rope_attention", lower_with_shared_context)

    model = onnx_export(graph, tmp_path / "shared.onnx", external_data=False)

    assert len(observed) == 2
    assert observed[0] is observed[1]
    attention_nodes = [
        node for node in model.graph.node if node.op_type == "FusedInferAttentionScore"
    ]
    assert [node.output[1] for node in attention_nodes] == [
        "mdc.attention.lse",
        "mdc.attention.lse.1",
    ]


def test_quantized_cache_graph_outputs_keep_abi(
    tmp_path: Path,
) -> None:
    model = onnx_export(
        _quantized_attention_graph(),
        tmp_path / "graph_outputs.onnx",
        external_data=False,
    )

    specifications = {
        item.name: item
        for item in (*model.graph.output, *model.graph.value_info)
    }
    quantizers = {
        node.output[0]: node
        for node in model.graph.node
        if node.op_type == "NPUAscendQuantV2"
        and node.output
        and node.output[0].startswith("present.")
    }
    cache_names = {
        f"present.{layer_id}.{edge}"
        for layer_id in range(2)
        for edge in ("key", "value")
    }

    assert quantizers.keys() == cache_names
    for cache_name in cache_names:
        assert (
            specifications[cache_name].type.tensor_type.elem_type
            == TensorProto.INT8
        )
        assert quantizers[cache_name].input[0].startswith(f"{cache_name}.float")
    attention_nodes = [
        node for node in model.graph.node if node.op_type == "FusedInferAttentionScore"
    ]
    assert {
        node.input[index]
        for node in attention_nodes
        for index in (
            AttentionInput.KEY,
            AttentionInput.VALUE,
        )
    } == cache_names


def test_lower_rms_norms_builds_producer_index_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, metadata = _rms_norm_fixture(
        ("layers.0.norm", "layers.1.norm", "layers.2.norm"),
        reverse_boundaries=True,
    )
    original_names = [node.name for node in model.graph.node]
    real_producer_map = producer_map
    calls = 0

    def counted_producer_map(value: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
        nonlocal calls
        calls += 1
        return real_producer_map(value)

    monkeypatch.setattr(
        "mdc_llm_deploy.onnx.transform.attention.producer_map",
        counted_producer_map,
    )

    attention_lowering.lower_rms_norms(model, metadata)

    assert calls == 1
    assert sum(node.op_type == "NPURmsNorm" for node in model.graph.node) == 3
    assert [node.name for node in model.graph.node] == [
        (
            f"mdc.rms_norm.layers.{index // 3}.norm"
            if name.startswith("terminal.")
            else name
        )
        for index, name in enumerate(original_names)
    ]


@pytest.mark.parametrize(
    ("last_producer_valid", "raises"),
    [(False, True), (True, False)],
)
def test_lower_rms_norms_preserves_last_duplicate_producer_semantics(
    last_producer_valid: bool,
    raises: bool,
) -> None:
    model, metadata = _rms_norm_fixture(("norm",))
    duplicate = helper.make_node(
        "Add",
        ["source.0"],
        ["normalized.0"],
        name="duplicate.normalized",
    )
    insert_at = 1 if last_producer_valid else 2
    model.graph.node.insert(insert_at, duplicate)

    if raises:
        with pytest.raises(
            OnnxExportError,
            match=r"RmsNorm boundary 'norm' lacks a standard normalization spine",
        ):
            attention_lowering.lower_rms_norms(model, metadata)
    else:
        attention_lowering.lower_rms_norms(model, metadata)
        assert sum(node.op_type == "NPURmsNorm" for node in model.graph.node) == 1


def test_lower_rms_norms_preserves_missing_terminal_error() -> None:
    model, metadata = _rms_norm_fixture(("norm",))
    del model.graph.node[-1]

    with pytest.raises(
        OnnxExportError,
        match=r"RmsNorm boundary 'norm' maps to 0 terminal nodes",
    ):
        attention_lowering.lower_rms_norms(model, metadata)


def test_lower_rms_norms_replacements_follow_graph_order() -> None:
    model, metadata = _rms_norm_fixture(
        ("layers.0.norm", "layers.1.norm"),
        reverse_boundaries=True,
    )

    attention_lowering.lower_rms_norms(model, metadata)

    assert [node.name for node in model.graph.node] == [
        "padding.0",
        "normalized.0",
        "mdc.rms_norm.layers.0.norm",
        "padding.1",
        "normalized.1",
        "mdc.rms_norm.layers.1.norm",
    ]
