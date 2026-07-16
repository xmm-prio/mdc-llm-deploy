"""ONNX symbolic ABI contract tests."""

from __future__ import annotations

import io
from typing import Any

import onnx
import torch

from mdc_llm_deploy.attention_layout import (
    ATTENTION_INPUT_COUNT,
    RELEASE_ATTENTION_ATTRIBUTES,
    AttentionInput,
)
from mdc_llm_deploy.mdc_ops.symbolics import (
    _attention_symbolic,
    register_onnx_symbolics,
)


class _RecordingGraph:
    def __init__(self) -> None:
        self.name = ""
        self.inputs: tuple[Any, ...] = ()
        self.attributes: dict[str, Any] = {}

    def op(self, name: str, *inputs: Any, **attributes: Any) -> Any:
        if name == "prim::Constant":
            return _OptionalPlaceholder()
        self.name = name
        self.inputs = inputs
        self.attributes = attributes
        return ("output", "lse")


class _OptionalPlaceholder:
    def setType(self, value: Any) -> None:  # noqa: N802
        self.value = value


def test_attention_symbolic_emits_complete_release_abi() -> None:
    graph = _RecordingGraph()
    values = {
        name: object()
        for name in (
            "query",
            "key",
            "value",
            "mask",
            "key_scale",
            "key_offset",
            "value_scale",
            "value_offset",
            "query_scale",
            "score_scale",
        )
    }

    outputs = _attention_symbolic(
        graph,
        values["query"],
        values["key"],
        values["value"],
        values["mask"],
        0.125,
        4,
        2,
        values["key_scale"],
        values["key_offset"],
        values["value_scale"],
        values["value_offset"],
        values["query_scale"],
        values["score_scale"],
        False,
    )

    assert outputs == ("output", "lse")
    assert graph.name == "FusedInferAttentionScore"
    assert len(graph.inputs) == ATTENTION_INPUT_COUNT
    expected_slots = {
        AttentionInput.QUERY: values["query"],
        AttentionInput.KEY: values["key"],
        AttentionInput.VALUE: values["value"],
        AttentionInput.ATTEN_MASK: values["mask"],
        AttentionInput.QUANT_SCALE1: values["score_scale"],
        AttentionInput.KEY_ANTIQUANT_SCALE: values["key_scale"],
        AttentionInput.KEY_ANTIQUANT_OFFSET: values["key_offset"],
        AttentionInput.VALUE_ANTIQUANT_SCALE: values["value_scale"],
        AttentionInput.VALUE_ANTIQUANT_OFFSET: values["value_offset"],
        AttentionInput.DEQUANT_SCALE_QUERY: values["query_scale"],
    }
    for index, value in enumerate(graph.inputs):
        expected = expected_slots.get(index)
        if expected is None:
            assert isinstance(value, _OptionalPlaceholder)
        else:
            assert value is expected
    assert graph.attributes["num_heads_i"] == 4
    assert graph.attributes["num_key_value_heads_i"] == 2
    assert graph.attributes["scale_f"] == 0.125
    assert graph.attributes["outputs"] == 2
    for name, expected in RELEASE_ATTENTION_ATTRIBUTES.items():
        suffix = "s" if isinstance(expected, str) else "i"
        assert graph.attributes[f"{name}_{suffix}"] == expected


class _AttentionModule(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.mdc_llm_deploy.fused_infer_attention_score.default(
            query,
            key,
            value,
            mask,
            0.125,
            4,
            2,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
        )


def test_attention_symbolic_preserves_optional_slots_in_protobuf() -> None:
    register_onnx_symbolics()
    buffer = io.BytesIO()
    torch.onnx.export(
        _AttentionModule().eval(),
        (
            torch.randn(1, 4, 2, 8),
            torch.randn(1, 2, 3, 8),
            torch.randn(1, 2, 3, 8),
            torch.zeros(1, 1, 2, 3, dtype=torch.bool),
        ),
        buffer,
        opset_version=18,
        dynamo=False,
        input_names=["query", "key", "value", "mask"],
        output_names=["attention", "softmax_lse"],
    )

    model = onnx.load_model_from_string(buffer.getvalue())
    node = next(
        item
        for item in model.graph.node
        if item.op_type == "FusedInferAttentionScore"
    )
    populated = {
        AttentionInput.QUERY,
        AttentionInput.KEY,
        AttentionInput.VALUE,
        AttentionInput.ATTEN_MASK,
    }
    assert node.domain == ""
    assert len(node.input) == ATTENTION_INPUT_COUNT
    assert all(
        bool(value) if index in populated else value == ""
        for index, value in enumerate(node.input)
    )

    reparsed = onnx.load_model_from_string(model.SerializeToString())
    reparsed_node = next(
        item
        for item in reparsed.graph.node
        if item.op_type == "FusedInferAttentionScore"
    )
    assert list(reparsed_node.input) == list(node.input)
