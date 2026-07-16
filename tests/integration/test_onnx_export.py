"""Integration tests for model-independent ONNX persistence contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import onnx
import pytest
import torch
from torch import nn
from torch.fx import Graph, GraphModule

from mdc_llm_deploy.export import convert_to_decode, export
from mdc_llm_deploy.graph import (
    GraphMetadata,
    GraphStage,
    TensorAbi,
    set_metadata,
)
from mdc_llm_deploy.onnx_export import onnx_export
from mdc_llm_deploy.quantization import oneshot
from tests.model_fixtures import dense_model, moe_model

pytestmark = pytest.mark.integration


def test_model_independent_small_operator_graph_exports(
    tmp_path: Path,
) -> None:
    fx_graph = Graph()
    value = fx_graph.placeholder("value")
    result = fx_graph.call_function(torch.ops.aten.relu.default, (value,))
    fx_graph.output(result)
    graph = GraphModule(nn.Module(), fx_graph)
    set_metadata(
        graph,
        GraphMetadata(
            schema_version=1,
            stage=GraphStage.FLOAT_PREFILL,
            model_kind="dense",
            input_abi=(TensorAbi("value", "float32", (1, 2)),),
            output_abi=(TensorAbi("result", "float32", (1, 2)),),
            sequence_length=2,
        ),
    )

    model = onnx_export(graph, tmp_path / "small.onnx")

    assert [item.name for item in model.graph.input] == ["value"]
    assert [item.name for item in model.graph.output] == ["result"]
    assert any(node.op_type == "Relu" for node in model.graph.node)


def _graph(
    *,
    mask_mode: Literal["causal", "none"] = "causal",
) -> torch.fx.GraphModule:
    return export(
        dense_model(4, mask_mode=mask_mode),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )


@pytest.mark.parametrize("mask_mode", ["causal", "none"])
def test_onnx_semantics_come_from_model_config(
    tmp_path: Path,
    mask_mode: Literal["causal", "none"],
) -> None:
    target = tmp_path / f"{mask_mode}.onnx"

    model = onnx_export(_graph(mask_mode=mask_mode), target)

    assert target.is_file()
    assert (tmp_path / f"{mask_mode}.onnx.data").is_file()
    assert [item.name for item in model.graph.input] == ["input_ids"]
    assert [item.name for item in model.graph.output] == ["logits"]
    onnx.load(target, load_external_data=True)


def test_repeated_export_always_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "model.onnx"
    target.write_bytes(b"old")

    onnx_export(_graph(), target, external_data=False)
    first = target.read_bytes()
    onnx_export(_graph(mask_mode="none"), target, external_data=False)

    assert target.read_bytes() != b"old"
    assert target.read_bytes() != first
    assert not (tmp_path / "model.onnx.data").exists()
    assert not any(path.name.startswith(".model") for path in tmp_path.iterdir())


def test_two_layer_onnx_contains_per_layer_custom_operators(
    tmp_path: Path,
) -> None:
    graph = export(
        dense_model(4, layers=2),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )

    model = onnx_export(graph, tmp_path / "two-layer.onnx")

    counts = {
        op_type: sum(node.op_type == op_type for node in model.graph.node)
        for op_type in (
            "NPURmsNorm",
            "FusedInferAttentionScore",
            "ApplyRotaryPosEmb",
        )
    }
    assert counts == {
        "NPURmsNorm": 9,
        "FusedInferAttentionScore": 2,
        "ApplyRotaryPosEmb": 2,
    }
    initializers = {item.name for item in model.graph.initializer}
    rms_nodes = [
        node for node in model.graph.node if node.op_type == "NPURmsNorm"
    ]
    assert all(
        node.input[1] in initializers
        and node.input[1].startswith("graph.")
        and node.input[1].endswith(".weight")
        for node in rms_nodes
    )


def test_float_moe_exports_one_expert_major_custom_node(
    tmp_path: Path,
) -> None:
    graph = export(
        moe_model(4, expert_count=3, top_k=2),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )

    model = onnx_export(graph, tmp_path / "moe.onnx")
    nodes = [node for node in model.graph.node if node.op_type == "MoeExpert"]
    initializers = {item.name: item for item in model.graph.initializer}

    assert len(nodes) == 1
    assert tuple(initializers[nodes[0].input[3]].dims) == (3, 3 * 64 * 32)
    assert not nodes[0].input[4]
    assert not nodes[0].input[5]


def test_int8_moe_exports_quant_parameters(tmp_path: Path) -> None:
    model = moe_model(4, expert_count=3, top_k=2)
    block = model.model.layers[0].mlp
    weights = block.expert_weights.detach().reshape(3, 3, -1)
    scales = weights.abs().amax(dim=-1).clamp_min(1e-8) / 127
    quantized = torch.round(weights / scales.unsqueeze(-1)).clamp(
        -128, 127
    ).to(torch.int8)
    block.set_packed_weights(
        quantized.reshape_as(block.expert_weights),
        scales=scales,
    )
    graph = export(
        model,
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )

    exported = onnx_export(graph, tmp_path / "moe-int8.onnx")
    node = next(
        item for item in exported.graph.node if item.op_type == "MoeExpert"
    )
    initializers = {item.name: item for item in exported.graph.initializer}

    assert tuple(initializers[node.input[3]].dims) == (3, 3 * 64 * 32)
    assert tuple(initializers[node.input[4]].dims) == (3, 3)
    assert not node.input[5]


def test_oneshot_moe_exports_int8_packed_weights(
    tmp_path: Path,
) -> None:
    graph = export(
        moe_model(4, expert_count=3, top_k=2),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )
    oneshot(
        graph,
        "configs/minmax-moe-w8a8.json",
        [{"input_ids": torch.arange(4).reshape(1, 4)}],
    )

    exported = onnx_export(graph, tmp_path / "moe-oneshot.onnx")
    node = next(
        item for item in exported.graph.node if item.op_type == "MoeExpert"
    )
    initializers = {item.name: item for item in exported.graph.initializer}
    producers = {
        output: item
        for item in exported.graph.node
        for output in item.output
    }

    assert initializers[node.input[3]].data_type == onnx.TensorProto.INT8
    assert len(initializers[node.input[3]].dims) == 1
    assert tuple(initializers[node.input[4]].dims) == (13,)
    assert producers[node.input[0]].op_type == "NPUAscendQuantV2"
    assert producers[node.input[1]].op_type == "Cast"


def test_two_layer_decode_accepts_every_cache_and_only_returns_logits(
    tmp_path: Path,
) -> None:
    graph = export(
        dense_model(4, layers=2),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )
    convert_to_decode(graph)

    model = onnx_export(graph, tmp_path / "decode.onnx")

    assert [item.name for item in model.graph.input] == [
        "input_ids",
        "past.0.key",
        "past.0.value",
        "past.1.key",
        "past.1.value",
    ]
    assert [item.name for item in model.graph.output] == ["logits"]
    assert sum(
        node.op_type == "FusedInferAttentionScore"
        for node in model.graph.node
    ) == 2
