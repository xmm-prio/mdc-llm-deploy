from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn
from torch.fx import Graph, GraphModule

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.graph import (
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    TensorAbi,
)
from mdc_llm_deploy.onnx_export.standard_export import export_standard_onnx
from mdc_llm_deploy.onnx_protocol import MDC_ONNX_OPSET


def _metadata() -> GraphMetadata:
    return GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="dense",
        input_abi=(TensorAbi("x", "float32", (1, 2)),),
        output_abi=(TensorAbi("output", "float32", (1, 2)),),
        sequence_length=2,
    )


def _linear_graph() -> GraphModule:
    root = nn.Module()
    root.register_parameter("first_weight", nn.Parameter(torch.eye(2)))
    root.register_parameter("second_weight", nn.Parameter(torch.eye(2)))
    graph = Graph()
    value = graph.placeholder("x")
    first_weight = graph.get_attr("first_weight")
    value = graph.call_function(
        torch.ops.aten.linear.default,
        (value, first_weight, None),
    )
    second_weight = graph.get_attr("second_weight")
    value = graph.call_function(
        torch.ops.aten.linear.default,
        (value, second_weight, None),
    )
    graph.output(value)
    return GraphModule(root, graph)


def _rms_graph() -> GraphModule:
    root = nn.Module()
    root.add_module("norm", nn.Module())
    root.norm.register_parameter(  # type: ignore[attr-defined]
        "weight",
        nn.Parameter(torch.ones(2)),
    )
    graph = Graph()
    value = graph.placeholder("x")
    weight = graph.get_attr("norm.weight")
    value = graph.call_function(torch.ops.aten.mul.Tensor, (value, weight))
    graph.output(value)
    return GraphModule(root, graph)


def _standard_model() -> onnx.ModelProto:
    first = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="onnx_first")
    second = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="onnx_second")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "onnx_first"], ["hidden"]),
            helper.make_node("Identity", ["onnx_first"], ["weight_copy"]),
            helper.make_node("MatMul", ["hidden", "onnx_second"], ["output"]),
        ],
        "standard",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 2))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 2))],
        initializer=[first, second],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _rms_standard_model() -> onnx.ModelProto:
    weight = numpy_helper.from_array(
        np.ones(2, dtype=np.float32),
        name="onnx_norm",
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["onnx_norm"],
                ["graph.norm.weight"],
            ),
            helper.make_node(
                "Mul",
                ["x", "graph.norm.weight"],
                ["output"],
            ),
        ],
        "rms",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 2))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 2))],
        initializer=[weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_standard_export_wraps_external_failure_and_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = RuntimeError("boom")

    def fail_export(*args: Any, **kwargs: Any) -> None:
        raise original

    monkeypatch.setattr(torch.onnx, "export", fail_export)

    with pytest.raises(
        OnnxExportError,
        match="Standard ONNX validation failed: boom",
    ) as captured:
        export_standard_onnx(_linear_graph(), _metadata(), tmp_path)

    assert captured.value.__cause__ is original
    assert list(tmp_path.iterdir()) == []


def test_standard_export_restores_initializer_fqns_and_all_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_options: dict[str, Any] = {}

    def save_standard(
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: Path,
        **kwargs: Any,
    ) -> None:
        del model, args
        export_options.update(kwargs)
        onnx.save_model(_standard_model(), output)

    monkeypatch.setattr(torch.onnx, "export", save_standard)

    model = export_standard_onnx(_linear_graph(), _metadata(), tmp_path)

    assert [item.name for item in model.graph.initializer] == [
        "graph.first_weight",
        "graph.second_weight",
    ]
    assert [list(node.input) for node in model.graph.node] == [
        ["x", "graph.first_weight"],
        ["graph.first_weight"],
        ["hidden", "graph.second_weight"],
    ]
    assert export_options["opset_version"] == MDC_ONNX_OPSET
    assert export_options["do_constant_folding"] is True
    assert (
        export_options["training"]
        is torch.onnx.TrainingMode.PRESERVE
    )
    assert export_options["dynamo"] is False
    assert list(tmp_path.iterdir()) == []


def test_standard_export_folds_rms_norm_weight_into_initializer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def save_standard(
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: Path,
        **kwargs: Any,
    ) -> None:
        del model, args, kwargs
        onnx.save_model(_rms_standard_model(), output)

    monkeypatch.setattr(torch.onnx, "export", save_standard)
    value = _metadata()
    value = GraphMetadata(
        schema_version=value.schema_version,
        stage=value.stage,
        model_kind=value.model_kind,
        input_abi=value.input_abi,
        output_abi=value.output_abi,
        boundaries=(
            FusionBoundary("rms_norm", "norm", ("mul",)),
        ),
        sequence_length=value.sequence_length,
    )

    model = export_standard_onnx(_rms_graph(), value, tmp_path)

    assert [item.name for item in model.graph.initializer] == [
        "graph.norm.weight"
    ]
    assert all(node.op_type != "Identity" for node in model.graph.node)
    assert next(node for node in model.graph.node if node.op_type == "Mul").input[1] == (
        "graph.norm.weight"
    )
    assert list(tmp_path.iterdir()) == []


def test_standard_export_materializes_external_data_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def save_external_standard(
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: Path,
        **kwargs: Any,
    ) -> None:
        del model, args, kwargs
        onnx.save_model(
            _standard_model(),
            output,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.data",
            size_threshold=0,
        )

    monkeypatch.setattr(torch.onnx, "export", save_external_standard)

    model = export_standard_onnx(_linear_graph(), _metadata(), tmp_path)

    assert all(
        item.data_location == TensorProto.DEFAULT
        and bool(item.raw_data)
        and not item.external_data
        for item in model.graph.initializer
    )
    assert list(tmp_path.iterdir()) == []
