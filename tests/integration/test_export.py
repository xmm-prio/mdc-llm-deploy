"""Integration tests for Qwen3 capture and multi-layer graph contracts."""

from __future__ import annotations

import pytest
import torch
from torch.fx import Graph, GraphModule

from mdc_llm_deploy.errors import UnsupportedPatternError
from mdc_llm_deploy.export import convert_to_decode, export
from mdc_llm_deploy.export.decode_cache import insert_cache_quantization
from mdc_llm_deploy.fx_inspection import flatten_nodes
from mdc_llm_deploy.graph import GraphStage, metadata
from mdc_llm_deploy.graph_types import QuantizedTarget
from tests.model_fixtures import dense_model, moe_model

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("layers", [1, 2])
def test_dense_fx_export_preserves_internal_per_layer_kv_outputs(
    layers: int,
) -> None:
    model = dense_model(8, layers=layers)
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    expected = model(**inputs)

    graph = export(model, inputs)
    actual = graph(**inputs)

    assert len(actual) == 1 + 2 * layers
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)
    assert tuple(item.name for item in metadata(graph).output_abi) == (
        "logits",
        *tuple(
            name
            for layer_id in range(layers)
            for name in (
                f"present.{layer_id}.key",
                f"present.{layer_id}.value",
            )
        ),
    )


def test_moe_export_preserves_custom_operator() -> None:
    graph = export(
        moe_model(4, expert_count=3, top_k=2),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )

    targets = {str(node.target) for node in graph.graph.nodes}
    assert any("mdc_llm_deploy.moe_expert" in target for target in targets)
    assert metadata(graph).model_kind == "moe"


def test_decode_rewrite_handles_every_layer_cache() -> None:
    graph = export(
        dense_model(8, layers=2),
        {"input_ids": torch.arange(8).reshape(1, 8)},
    )

    convert_to_decode(graph)

    value = metadata(graph)
    assert value.stage is GraphStage.FLOAT_DECODE
    assert tuple(item.name for item in value.input_abi[1:]) == (
        "past.0.key",
        "past.0.value",
        "past.1.key",
        "past.1.value",
    )
    assert tuple(item.name for item in value.output_abi[1:]) == (
        "present.0.key",
        "present.0.value",
        "present.1.key",
        "present.1.value",
    )
    assert value.properties["cache_devices"] == {
        "past.0.key": "cpu",
        "past.0.value": "cpu",
        "past.1.key": "cpu",
        "past.1.value": "cpu",
    }


def test_export_rejects_training_model() -> None:
    model = dense_model(4)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        export(model, {"input_ids": torch.arange(4).reshape(1, 4)})


def test_decode_rejects_graph_without_attention() -> None:
    class NoAttention(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids.float()

    graph = export(
        NoAttention().eval(),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )
    with pytest.raises(UnsupportedPatternError, match="attention boundary"):
        convert_to_decode(graph)


def test_decode_device_inference_failure_rolls_back_transaction() -> None:
    graph = export(
        dense_model(4),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )
    output = next(node for node in graph.graph.nodes if node.op == "output")
    cache = list(flatten_nodes(output.args[0]))[1]
    cache.meta.pop("val", None)
    original_graph = str(graph.graph)
    original_metadata = metadata(graph)
    original_parameters = dict(
        graph.named_parameters(remove_duplicate=False)
    )

    with pytest.raises(UnsupportedPatternError, match="cannot infer device"):
        convert_to_decode(graph)

    assert str(graph.graph) == original_graph
    assert metadata(graph) is original_metadata
    assert {
        name: id(parameter)
        for name, parameter in graph.named_parameters(remove_duplicate=False)
    } == {name: id(parameter) for name, parameter in original_parameters.items()}


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_decode_cache_constants_follow_each_kv_device() -> None:
    raw_graph = Graph()
    current_key = raw_graph.placeholder("current_key")
    past_key = raw_graph.placeholder("past_key")
    current_value = raw_graph.placeholder("current_value")
    past_value = raw_graph.placeholder("past_value")
    output = raw_graph.output((current_key, current_value))
    current_key.meta["val"] = torch.ones(1, 1, 1, 2, device="cuda:0")
    current_value.meta["val"] = torch.ones(1, 1, 1, 2, device="cuda:1")
    graph = GraphModule(torch.nn.Module(), raw_graph)
    target = QuantizedTarget(
        fqn="attention.key",
        target_type="attention",
        algorithm="minmax",
        bits=8,
        granularity="per_tensor",
        symmetric=True,
        scale=(0.5,),
        zero_point=(0,),
    )

    with graph.graph.inserting_before(output):
        insert_cache_quantization(
            graph, current_key, past_key, target, "0_key", 2
        )
        insert_cache_quantization(
            graph, current_value, past_value, target, "0_value", 2
        )

    assert graph._mdc_0_key_current_scale.device == torch.device("cuda:0")
    assert graph._mdc_0_value_current_scale.device == torch.device("cuda:1")
    assert past_key.meta["val"].device == torch.device("cuda:0")
    assert past_value.meta["val"].device == torch.device("cuda:1")
