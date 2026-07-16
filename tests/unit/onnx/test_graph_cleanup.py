from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.onnx.transform.cleanup import (
    remove_dynamic_value_info,
    remove_redundant_identities,
    topologically_sort,
)


def _model(
    nodes: list[onnx.NodeProto],
    *,
    outputs: tuple[str, ...] = ("output",),
) -> onnx.ModelProto:
    zero = numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), name="zero")
    graph = helper.make_graph(
        nodes,
        "cleanup",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1,))],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, (1,))
            for name in outputs
        ],
        initializer=[zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_topologically_sort_restores_dependency_order() -> None:
    model = _model(
        [
            helper.make_node("Relu", ["hidden"], ["output"], name="second"),
            helper.make_node("Add", ["input", "zero"], ["hidden"], name="first"),
        ]
    )

    topologically_sort(model)

    assert [node.name for node in model.graph.node] == ["first", "second"]


def test_topologically_sort_rejects_missing_producer() -> None:
    model = _model(
        [helper.make_node("Relu", ["missing"], ["output"], name="blocked")]
    )

    with pytest.raises(
        OnnxExportError,
        match="cannot be topologically sorted",
    ):
        topologically_sort(model)


def test_remove_dynamic_value_info_retains_only_static_shapes() -> None:
    model = _model(
        [helper.make_node("Identity", ["input"], ["output"])],
    )
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info("static", TensorProto.FLOAT, (1, 2)),
            helper.make_tensor_value_info(
                "dynamic",
                TensorProto.FLOAT,
                ("sequence", 2),
            ),
        ]
    )

    remove_dynamic_value_info(model)

    assert [item.name for item in model.graph.value_info] == ["static"]


def test_remove_redundant_identities_preserves_fan_out_source() -> None:
    model = _model(
        [
            helper.make_node("Add", ["input", "zero"], ["hidden"], name="source"),
            helper.make_node("Identity", ["hidden"], ["output"], name="identity"),
            helper.make_node("Neg", ["hidden"], ["side"], name="side"),
        ],
        outputs=("output", "side"),
    )

    remove_redundant_identities(model)

    assert [node.name for node in model.graph.node] == [
        "source",
        "identity",
        "side",
    ]
