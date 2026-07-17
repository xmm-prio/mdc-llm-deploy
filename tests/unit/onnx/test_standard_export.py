from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from torch import nn
from torch.fx import Graph, GraphModule

import mdc_llm_deploy.onnx.export.standard as standard_module
from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.graph.lifecycle import GraphMetadata, GraphStage, TensorAbi
from mdc_llm_deploy.onnx.export.standard import build_standard_onnx


def _metadata() -> GraphMetadata:
    return GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="dense",
        input_abi=(TensorAbi("x", "float32", (1, 2)),),
        output_abi=(TensorAbi("output", "float32", (1, 2)),),
        sequence_length=2,
        properties={"input_devices": {"x": "cpu"}},
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


def _standard_model() -> onnx.ModelProto:
    first = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="onnx_first")
    second = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="onnx_second")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "onnx_first"], ["hidden"]),
            helper.make_node("MatMul", ["hidden", "onnx_second"], ["output"]),
        ],
        "standard",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 2))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 2))],
        initializer=[first, second],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_standard_facade_calls_legacy_then_normalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    raw = _standard_model()

    def export(
        graph: GraphModule,
        metadata: GraphMetadata,
        directory: Path,
    ) -> onnx.ModelProto:
        del graph, metadata
        assert directory == tmp_path
        calls.append("legacy")
        return raw

    def normalize(
        model: onnx.ModelProto,
        graph: GraphModule,
        metadata: GraphMetadata,
    ) -> onnx.ModelProto:
        del graph, metadata
        assert model is raw
        calls.append("normalization")
        return model

    monkeypatch.setattr(standard_module, "export_legacy_onnx", export)
    monkeypatch.setattr(standard_module, "normalize_standard_onnx", normalize)

    assert build_standard_onnx(_linear_graph(), _metadata(), tmp_path) is raw
    assert calls == ["legacy", "normalization"]


def test_standard_facade_real_cpu_graph_preserves_numeric_contract(
    tmp_path: Path,
) -> None:
    graph = _linear_graph()
    value = torch.tensor([[2.0, -3.0]], dtype=torch.float32)

    model = build_standard_onnx(graph, _metadata(), tmp_path)
    actual = ReferenceEvaluator(model).run(None, {"x": value.numpy()})[0]

    np.testing.assert_allclose(actual, graph(x=value).detach().numpy())
    assert [item.name for item in model.graph.initializer] == [
        "graph.first_weight",
        "graph.second_weight",
    ]
    assert all(node.op_type != "Transpose" for node in model.graph.node)


@pytest.mark.parametrize("owner", ["export_legacy_onnx", "normalize_standard_onnx"])
def test_standard_facade_preserves_domain_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    owner: str,
) -> None:
    original = OnnxExportError("domain failure")

    def fail(*args: Any, **kwargs: Any) -> onnx.ModelProto:
        raise original

    raw = _standard_model()
    monkeypatch.setattr(
        standard_module,
        "export_legacy_onnx",
        lambda *args, **kwargs: raw,
    )
    monkeypatch.setattr(standard_module, owner, fail)
    with pytest.raises(OnnxExportError) as captured:
        build_standard_onnx(_linear_graph(), _metadata(), tmp_path)
    assert captured.value is original


@pytest.mark.parametrize("owner", ["export_legacy_onnx", "normalize_standard_onnx"])
def test_standard_facade_wraps_unknown_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    owner: str,
) -> None:
    original = RuntimeError("boom")

    def fail(*args: Any, **kwargs: Any) -> onnx.ModelProto:
        raise original

    raw = _standard_model()
    monkeypatch.setattr(
        standard_module,
        "export_legacy_onnx",
        lambda *args, **kwargs: raw,
    )
    monkeypatch.setattr(
        standard_module,
        "normalize_standard_onnx",
        lambda *args, **kwargs: raw,
    )
    monkeypatch.setattr(standard_module, owner, fail)
    with pytest.raises(
        OnnxExportError,
        match="Standard ONNX validation failed: boom",
    ) as captured:
        build_standard_onnx(_linear_graph(), _metadata(), tmp_path)
    assert captured.value.__cause__ is original
