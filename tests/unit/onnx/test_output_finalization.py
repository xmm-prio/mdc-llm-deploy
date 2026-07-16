from __future__ import annotations

import onnx
import pytest
from onnx import TensorProto, helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.onnx.transform.output import (
    retain_logits_output,
)


def _model(*output_names: str) -> onnx.ModelProto:
    outputs = [
        helper.make_tensor_value_info(
            name,
            TensorProto.FLOAT,
            (1, 1, 4),
        )
        for name in output_names
    ]
    graph = helper.make_graph([], "outputs", [], outputs)
    return helper.make_model(graph)


def test_output_finalization_retains_only_logits() -> None:
    model = _model("logits", "present.0.key", "present.0.value")

    retain_logits_output(model)

    assert [output.name for output in model.graph.output] == ["logits"]


def test_output_finalization_requires_one_logits_output() -> None:
    model = _model("present.0.key")

    with pytest.raises(
        OnnxExportError,
        match="found 0 logits outputs",
    ):
        retain_logits_output(model)
