from __future__ import annotations

import operator
from collections.abc import Iterable

import pytest
import torch
from torch import Tensor, nn
from torch.fx import Graph, GraphModule

from mdc_llm_deploy.errors import GraphStateError
from mdc_llm_deploy.export.decode.metadata import build_decode_metadata
from mdc_llm_deploy.graph.metadata import (
    GraphMetadata,
    GraphStage,
    TensorAbi,
)


def _metadata(layers: int) -> GraphMetadata:
    return GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="dense",
        input_abi=(TensorAbi("input_ids", "int64", (1, 4)),),
        output_abi=(
            TensorAbi("logits", "float32", (1, 4, 16)),
            *(
                TensorAbi(name, "float32", (1, 2, 4, 8))
                for layer_id in range(layers)
                for name in (
                    f"present.{layer_id}.key",
                    f"present.{layer_id}.value",
                )
            ),
        ),
        sequence_length=4,
        properties={
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "input_devices": {"input_ids": "cpu"},
        },
    )


def _candidate(
    placeholders: Iterable[tuple[str, object]],
    *,
    duplicate_non_placeholder: str | None = None,
) -> GraphModule:
    graph = Graph()
    input_ids = graph.placeholder("input_ids")
    duplicate = (
        graph.call_function(operator.neg, (input_ids,))
        if duplicate_non_placeholder is not None
        else None
    )
    cache_nodes = []
    for index, (name, value) in enumerate(placeholders):
        node = graph.placeholder(f"cache_{index}")
        node.meta["val"] = value
        cache_nodes.append((node, name))
    graph.output(input_ids)
    candidate = GraphModule(nn.Module(), graph)
    if duplicate is not None:
        duplicate.name = duplicate_non_placeholder
    for node, name in cache_nodes:
        node.name = name
    return candidate


def test_build_decode_metadata_uses_first_placeholder_by_graph_order() -> None:
    candidate = _candidate(
        (
            ("past_0_key", torch.ones(1)),
            ("past_0_key", torch.ones(1, device="meta")),
            ("past_0_value", torch.ones(1)),
        ),
        duplicate_non_placeholder="past_0_key",
    )

    result = build_decode_metadata(candidate, _metadata(1))

    cache_devices = result.properties["cache_devices"]
    assert cache_devices["past.0.key"] == "cpu"
    assert tuple(cache_devices) == ("past.0.key", "past.0.value")


def test_build_decode_metadata_rejects_missing_cache_placeholder() -> None:
    candidate = _candidate((("past_0_value", torch.ones(1)),))

    with pytest.raises(
        GraphStateError,
        match=r"^Decode cache device is unavailable for layer 0 key$",
    ):
        build_decode_metadata(candidate, _metadata(1))


def test_build_decode_metadata_rejects_non_tensor_cache_metadata() -> None:
    candidate = _candidate(
        (
            ("past_0_key", object()),
            ("past_0_value", torch.ones(1)),
        )
    )

    with pytest.raises(
        GraphStateError,
        match=r"^Decode cache device is unavailable for layer 0 key$",
    ):
        build_decode_metadata(candidate, _metadata(1))


def test_build_decode_metadata_preserves_cache_error_order() -> None:
    candidate = _candidate(
        (
            ("past_1_key", object()),
            ("past_0_key", torch.ones(1)),
            ("past_0_value", object()),
            ("past_1_value", torch.ones(1)),
        )
    )

    with pytest.raises(
        GraphStateError,
        match=r"^Decode cache device is unavailable for layer 0 value$",
    ):
        build_decode_metadata(candidate, _metadata(2))


@pytest.mark.parametrize(
    ("tensor", "device"),
    (
        (torch.ones(1), "cpu"),
        (torch.ones(1, device="meta"), "meta"),
    ),
)
def test_build_decode_metadata_preserves_cache_device(
    tensor: Tensor,
    device: str,
) -> None:
    candidate = _candidate(
        (
            ("past_0_key", tensor),
            ("past_0_value", tensor),
        )
    )

    result = build_decode_metadata(candidate, _metadata(1))

    assert result.properties["cache_devices"] == {
        "past.0.key": device,
        "past.0.value": device,
    }
