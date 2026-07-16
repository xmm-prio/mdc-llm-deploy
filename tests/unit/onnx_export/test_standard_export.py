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
from mdc_llm_deploy.onnx_export.standard_export import (
    _example_arguments,
    export_standard_onnx,
)
from mdc_llm_deploy.onnx_protocol import MDC_ONNX_OPSET


def _metadata(*, properties: dict[str, Any] | None = None) -> GraphMetadata:
    return GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="dense",
        input_abi=(TensorAbi("x", "float32", (1, 2)),),
        output_abi=(TensorAbi("output", "float32", (1, 2)),),
        sequence_length=2,
        properties=(
            {"input_devices": {"x": "cpu"}}
            if properties is None
            else properties
        ),
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


def _unfolded_standard_model() -> onnx.ModelProto:
    first = numpy_helper.from_array(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        name="raw_first",
    )
    second = numpy_helper.from_array(
        np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        name="raw_second",
    )
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["raw_first"], ["first_weight"]),
            helper.make_node("MatMul", ["x", "first_weight"], ["hidden"]),
            helper.make_node("Transpose", ["raw_second"], ["second_weight"]),
            helper.make_node("MatMul", ["hidden", "second_weight"], ["output"]),
        ],
        "unfolded",
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


def test_example_arguments_follow_per_input_device_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[torch.device] = []
    original_zeros = torch.zeros

    def record_zeros(
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        captured.append(device)
        return original_zeros(shape, dtype=dtype)

    monkeypatch.setattr(torch, "zeros", record_zeros)
    value = GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="dense",
        input_abi=(
            TensorAbi("tokens", "int64", (1, 2)),
            TensorAbi("mask", "bool", (1, 2)),
        ),
        output_abi=(TensorAbi("output", "float32", (1, 2)),),
        properties={
            "input_devices": {
                "tokens": "cuda:1",
                "mask": "cpu",
            }
        },
    )

    arguments = _example_arguments(value)

    assert captured == [torch.device("cuda:1"), torch.device("cpu")]
    assert [argument.dtype for argument in arguments] == [torch.int64, torch.bool]


@pytest.mark.parametrize(
    ("properties", "message"),
    [
        ({}, "contract is missing"),
        ({"input_devices": {"other": "cpu"}}, "missing=\\['x'\\]"),
        ({"input_devices": {"x": 0}}, "for 'x' must be a string"),
        ({"input_devices": {"x": "not-a-device"}}, "for 'x' is invalid"),
    ],
)
def test_example_arguments_reject_invalid_input_device_contract(
    properties: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(OnnxExportError, match=message):
        _example_arguments(_metadata(properties=properties))


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
        del args
        export_options["model_training"] = model.training
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
    assert export_options["export_params"] is True
    assert export_options["model_training"] is False
    assert export_options["do_constant_folding"] is False
    assert export_options["training"] is torch.onnx.TrainingMode.PRESERVE
    assert export_options["dynamo"] is False
    assert list(tmp_path.iterdir()) == []


def test_standard_export_folds_linear_transposes_without_jit_constant_folding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def save_unfolded(
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: Path,
        **kwargs: Any,
    ) -> None:
        del model, args, kwargs
        onnx.save_model(_unfolded_standard_model(), output)

    monkeypatch.setattr(torch.onnx, "export", save_unfolded)

    model = export_standard_onnx(_linear_graph(), _metadata(), tmp_path)

    initializers = {item.name: item for item in model.graph.initializer}
    assert set(initializers) == {
        "graph.first_weight",
        "graph.second_weight",
    }
    np.testing.assert_array_equal(
        numpy_helper.to_array(initializers["graph.first_weight"]),
        np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
    )
    assert all(node.op_type != "Transpose" for node in model.graph.node)
    assert [node.input[1] for node in model.graph.node if node.op_type == "MatMul"] == [
        "graph.first_weight",
        "graph.second_weight",
    ]
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
        properties=value.properties,
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
