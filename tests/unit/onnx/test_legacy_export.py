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
from mdc_llm_deploy.graph.lifecycle import GraphMetadata, GraphStage, TensorAbi
from mdc_llm_deploy.onnx.export.legacy import (
    _example_arguments,
    _PositionalGraph,
    export_legacy_onnx,
)
from mdc_llm_deploy.operators.contracts.onnx import MDC_ONNX_OPSET


def _metadata(
    *,
    stage: GraphStage = GraphStage.FLOAT_PREFILL,
    dtype: str = "float32",
    properties: dict[str, Any] | None = None,
) -> GraphMetadata:
    return GraphMetadata(
        schema_version=1,
        stage=stage,
        model_kind="dense",
        input_abi=(TensorAbi("x", dtype, (1, 2)),),
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
    root.register_parameter(
        "first_weight",
        nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 5.0]])),
    )
    root.register_parameter(
        "second_weight",
        nn.Parameter(torch.tensor([[7.0, 11.0], [13.0, 17.0]])),
    )
    graph = Graph()
    value = graph.placeholder("x")
    first = graph.get_attr("first_weight")
    value = graph.call_function(torch.ops.aten.linear.default, (value, first, None))
    second = graph.get_attr("second_weight")
    value = graph.call_function(torch.ops.aten.linear.default, (value, second, None))
    graph.output(value)
    return GraphModule(root, graph)


def _model() -> onnx.ModelProto:
    weight = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["output"])],
        "legacy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 2))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 2))],
        initializer=[weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_legacy_export_preserves_exporter_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def save_model(
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: Path,
        **kwargs: Any,
    ) -> None:
        captured.update(kwargs)
        captured["model_training"] = model.training
        captured["arguments"] = args
        onnx.save_model(_model(), output)

    monkeypatch.setattr(torch.onnx, "export", save_model)
    export_legacy_onnx(_linear_graph(), _metadata(), tmp_path)

    assert captured["export_params"] is True
    assert captured["opset_version"] == MDC_ONNX_OPSET
    assert captured["do_constant_folding"] is False
    assert captured["input_names"] == ["x"]
    assert captured["output_names"] == ["output"]
    assert captured["model_training"] is False
    assert captured["training"] is torch.onnx.TrainingMode.PRESERVE
    assert captured["dynamo"] is False
    assert list(tmp_path.iterdir()) == []


def test_positional_graph_preserves_prefill_and_decode_call_conventions() -> None:
    class Calls(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received: object = None

        def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
            self.received = kwargs if kwargs else args
            return next(iter(kwargs.values())) if kwargs else args[0]

    value = torch.ones(1)
    prefill = Calls()
    _PositionalGraph(prefill, ("x",), use_kwargs=True)(value)  # type: ignore[arg-type]
    assert prefill.received == {"x": value}
    decode = Calls()
    _PositionalGraph(decode, ("x",), use_kwargs=False)(value)  # type: ignore[arg-type]
    assert decode.received == (value,)


def test_example_arguments_preserves_domain_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(OnnxExportError, match="Unsupported input dtype: complex64"):
        _example_arguments(_metadata(dtype="complex64"))

    original = RuntimeError("allocation failed")

    def fail_zeros(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise original

    monkeypatch.setattr(torch, "zeros", fail_zeros)
    with pytest.raises(
        OnnxExportError,
        match="Cannot create ONNX input 'x' on cpu: allocation failed",
    ) as captured:
        _example_arguments(_metadata())
    assert captured.value.__cause__ is original


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


def test_legacy_export_materializes_external_data_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def save_external(
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: Path,
        **kwargs: Any,
    ) -> None:
        del model, args, kwargs
        onnx.save_model(
            _model(),
            output,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.data",
            size_threshold=0,
        )

    monkeypatch.setattr(torch.onnx, "export", save_external)
    model = export_legacy_onnx(_linear_graph(), _metadata(), tmp_path)

    assert all(
        item.data_location == TensorProto.DEFAULT
        and bool(item.raw_data)
        and not item.external_data
        for item in model.graph.initializer
    )
    assert list(tmp_path.iterdir()) == []


def test_legacy_export_runs_real_cpu_two_linear_graph(tmp_path: Path) -> None:
    model = export_legacy_onnx(_linear_graph(), _metadata(), tmp_path)
    onnx.checker.check_model(model)
    assert sum(node.op_type in {"Gemm", "MatMul"} for node in model.graph.node) == 2
    assert list(tmp_path.iterdir()) == []
