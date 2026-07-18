from __future__ import annotations

import onnx
import pytest
from onnx import TensorProto, helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.graph.metadata import (
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    TensorAbi,
)
from mdc_llm_deploy.onnx.transform.output import finalize_artifact_outputs


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


def _metadata(*, save_kv_cache: bool) -> GraphMetadata:
    shape = (1, 2, 4, 8)
    return GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="dense",
        input_abi=(TensorAbi("input_ids", "int64", (1, 4)),),
        output_abi=(
            TensorAbi("logits", "float32", (1, 1, 4)),
            TensorAbi("present.0.key", "float32", shape),
            TensorAbi("present.0.value", "float32", shape),
        ),
        boundaries=(
            FusionBoundary("attention", "model.layers.0.self_attn"),
        ),
        sequence_length=4,
        properties={"save_kv_cache": save_kv_cache},
    )


@pytest.mark.parametrize(
    ("save_kv_cache", "expected"),
    [
        (True, ["logits", "present.0.key", "present.0.value"]),
        (False, ["logits"]),
    ],
)
def test_output_finalization_applies_artifact_policy(
    save_kv_cache: bool,
    expected: list[str],
) -> None:
    model = _model("present.0.value", "logits", "present.0.key")

    finalize_artifact_outputs(
        model,
        _metadata(save_kv_cache=save_kv_cache),
    )

    assert [output.name for output in model.graph.output] == expected


def test_output_finalization_requires_every_public_output() -> None:
    model = _model("present.0.key")

    with pytest.raises(
        OnnxExportError,
        match="requires one source",
    ):
        finalize_artifact_outputs(model, _metadata(save_kv_cache=True))
