"""Integration tests for model-independent ONNX persistence contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from torch import nn
from torch.fx import Graph, GraphModule

from mdc_llm_deploy.export import convert_to_decode, export
from mdc_llm_deploy.graph.lifecycle import (
    GraphMetadata,
    GraphStage,
    QuantizedTarget,
    TensorAbi,
    metadata,
    set_metadata,
)
from mdc_llm_deploy.onnx import onnx_export
from mdc_llm_deploy.onnx.export.standard import build_standard_onnx
from mdc_llm_deploy.onnx.transform.linear import append_quantized_linears
from mdc_llm_deploy.quantization import oneshot
from tests.support.models.qwen3 import dense_model, moe_model

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
            properties={"input_devices": {"value": "cpu"}},
        ),
    )

    model = onnx_export(graph, tmp_path / "small.onnx")

    assert [item.name for item in model.graph.input] == ["value"]
    assert [item.name for item in model.graph.output] == ["result"]
    assert any(node.op_type == "Relu" for node in model.graph.node)


def test_two_linear_weights_keep_content_identity_through_lowering(
    tmp_path: Path,
) -> None:
    root = nn.Module()
    root.add_module("first", nn.Linear(2, 2, bias=False))
    root.add_module("second", nn.Linear(2, 2, bias=False))
    first = torch.tensor([[0.25, -0.5], [0.75, 1.0]])
    second = torch.tensor([[-1.0, 1.5], [2.0, -2.5]])
    root.first.weight = nn.Parameter(first)  # type: ignore[attr-defined]
    root.second.weight = nn.Parameter(second)  # type: ignore[attr-defined]
    fx_graph = Graph()
    value = fx_graph.placeholder("x")
    for name in ("first.weight", "second.weight"):
        weight = fx_graph.get_attr(name)
        value = fx_graph.call_function(
            torch.ops.aten.linear.default,
            (value, weight, None),
        )
    fx_graph.output(value)
    graph = GraphModule(root, fx_graph)
    targets = (
        QuantizedTarget(
            fqn="second",
            target_type="linear",
            algorithm="minmax",
            bits=8,
            granularity="per_tensor",
            symmetric=True,
            scale=(0.5,),
            zero_point=(0,),
        ),
        QuantizedTarget(
            fqn="first",
            target_type="linear",
            algorithm="minmax",
            bits=8,
            granularity="per_tensor",
            symmetric=True,
            scale=(0.25,),
            zero_point=(0,),
        ),
    )
    graph_metadata = GraphMetadata(
        schema_version=1,
        stage=GraphStage.QUANTIZED_PREFILL,
        model_kind="dense",
        input_abi=(TensorAbi("x", "float32", (1, 2)),),
        output_abi=(TensorAbi("output", "float32", (1, 2)),),
        quantized_targets=targets,
        sequence_length=2,
        properties={"input_devices": {"x": "cpu"}},
    )

    standard = build_standard_onnx(graph, graph_metadata, tmp_path)
    append_quantized_linears(standard, graph_metadata)

    packed = {
        item.name: onnx.numpy_helper.to_array(item)
        for item in standard.graph.initializer
        if item.name in {"mdc.linear.first.weight", "mdc.linear.second.weight"}
    }
    np.testing.assert_array_equal(
        packed["mdc.linear.first.weight"],
        np.array([[1, 3], [-2, 4]], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        packed["mdc.linear.second.weight"],
        np.array([[-2, 4], [3, -5]], dtype=np.int8),
    )


def _graph(
    *,
    mask_mode: Literal["causal", "none"] = "causal",
) -> torch.fx.GraphModule:
    return export(
        dense_model(4, mask_mode=mask_mode),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )


def _tensor_devices(graph: GraphModule) -> dict[str, torch.device]:
    return {
        **{name: value.device for name, value in graph.named_parameters()},
        **{name: value.device for name, value in graph.named_buffers()},
    }


def _cuda_linear_graph(device: torch.device) -> GraphModule:
    root = nn.Module()
    root.register_parameter(
        "weight",
        nn.Parameter(
            torch.tensor(
                [[1.0, 2.0], [-3.0, 0.5], [4.0, -1.0]],
                device=device,
            )
        ),
    )
    fx_graph = Graph()
    value = fx_graph.placeholder("value")
    weight = fx_graph.get_attr("weight")
    result = fx_graph.call_function(
        torch.ops.aten.linear.default,
        (value, weight, None),
    )
    fx_graph.output(result)
    graph = GraphModule(root, fx_graph)
    set_metadata(
        graph,
        GraphMetadata(
            schema_version=1,
            stage=GraphStage.FLOAT_PREFILL,
            model_kind="dense",
            input_abi=(TensorAbi("value", "float32", (1, 2)),),
            output_abi=(TensorAbi("result", "float32", (1, 3)),),
            sequence_length=2,
            properties={"input_devices": {"value": str(device)}},
        ),
    )
    return graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_linear_legacy_export_preserves_numerics(
    tmp_path: Path,
) -> None:
    device = torch.device("cuda:0")
    graph = _cuda_linear_graph(device)
    target = tmp_path / "cuda-linear.onnx"
    before_devices = _tensor_devices(graph)

    model = onnx_export(graph, target, external_data=False)

    assert _tensor_devices(graph) == before_devices
    assert before_devices == {"weight": device}
    assert target.is_file()
    initializers = {item.name for item in model.graph.initializer}
    assert "graph.weight" in initializers
    producers = {output: node for node in model.graph.node for output in node.output}
    linear_nodes = [node for node in model.graph.node if node.op_type in {"Gemm", "MatMul"}]
    assert len(linear_nodes) == 1
    assert linear_nodes[0].input[1] == "graph.weight"
    weight_producer = producers.get(linear_nodes[0].input[1])
    assert weight_producer is None or weight_producer.op_type != "Transpose"

    value = torch.tensor([[1.5, -2.0]], device=device)
    expected = graph(value).detach().cpu().numpy()
    actual = ReferenceEvaluator(model).run(
        None,
        {"value": value.cpu().numpy()},
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_qwen3_w8a8_prefill_and_decode_preserve_placement(
    tmp_path: Path,
) -> None:
    device = torch.device("cuda")
    model = dense_model(4).to(device)
    inputs = {
        "input_ids": torch.arange(4, device=device).reshape(1, 4),
    }
    graph = export(model, inputs)
    oneshot(
        graph,
        "configs/quantization/minmax-linear-w8a8.json",
        [inputs],
    )

    before_prefill = _tensor_devices(graph)
    prefill = onnx_export(
        graph,
        tmp_path / "cuda-prefill.onnx",
        external_data=False,
    )

    assert prefill.graph.input[0].name == "input_ids"
    assert _tensor_devices(graph) == before_prefill
    assert metadata(graph).properties["input_devices"] == {"input_ids": "cuda:0"}

    convert_to_decode(graph)
    before_decode = _tensor_devices(graph)
    decode = onnx_export(
        graph,
        tmp_path / "cuda-decode.onnx",
        external_data=False,
    )

    assert [item.name for item in decode.graph.input] == [
        "input_ids",
        "past.0.key",
        "past.0.value",
    ]
    assert _tensor_devices(graph) == before_decode
    assert set(metadata(graph).properties["input_devices"].values()) == {"cuda:0"}


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
    rms_nodes = [node for node in model.graph.node if node.op_type == "NPURmsNorm"]
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
    quantized = torch.round(weights / scales.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    block.set_packed_weights(
        quantized.reshape_as(block.expert_weights),
        scales=scales,
    )
    graph = export(
        model,
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )

    exported = onnx_export(graph, tmp_path / "moe-int8.onnx")
    node = next(item for item in exported.graph.node if item.op_type == "MoeExpert")
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
        "configs/quantization/minmax-moe-w8a8.json",
        [{"input_ids": torch.arange(4).reshape(1, 4)}],
    )

    exported = onnx_export(graph, tmp_path / "moe-oneshot.onnx")
    node = next(item for item in exported.graph.node if item.op_type == "MoeExpert")
    initializers = {item.name: item for item in exported.graph.initializer}
    producers = {output: item for item in exported.graph.node for output in item.output}
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
    assert sum(node.op_type == "FusedInferAttentionScore" for node in model.graph.node) == 2
