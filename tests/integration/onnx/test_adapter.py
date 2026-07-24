from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.onnx import AdapterConfig, OnnxAdapter
from mdc_llm_deploy.onnx.pipeline import adapter as adapter_module


def _identity_model(*, opset: int = 21) -> onnx.ModelProto:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="identity")],
        "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _constant_expression_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["left", "right"], ["constant"]),
            helper.make_node("Mul", ["x", "constant"], ["y"]),
        ],
        "constant_expression",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        initializer=[
            numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), "left"),
            numpy_helper.from_array(np.array([3.0, 4.0], dtype=np.float32), "right"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def _qdq_model() -> onnx.ModelProto:
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "a_scale", "a_zp"], ["a_q"]),
        helper.make_node("DequantizeLinear", ["a_q", "a_scale", "a_zp"], ["a_dq"]),
        helper.make_node(
            "QuantizeLinear",
            ["weight", "w_scale", "w_zp"],
            ["w_q"],
            axis=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["w_q", "w_scale", "w_zp"],
            ["w_dq"],
            axis=1,
        ),
        helper.make_node("MatMul", ["a_dq", "w_dq"], ["y"], name="linear"),
    ]
    graph = helper.make_graph(
        nodes,
        "adapter",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 2, 4])],
        initializer=[
            numpy_helper.from_array(np.array(0.25, dtype=np.float16), "a_scale"),
            numpy_helper.from_array(np.array(0, dtype=np.int8), "a_zp"),
            numpy_helper.from_array(np.ones((3, 4), dtype=np.float16), "weight"),
            numpy_helper.from_array(np.full((4,), 0.5, dtype=np.float16), "w_scale"),
            numpy_helper.from_array(np.zeros((4,), dtype=np.int8), "w_zp"),
        ],
        value_info=[
            helper.make_tensor_value_info("a_q", TensorProto.INT8, [1, 2, 3]),
            helper.make_tensor_value_info("a_dq", TensorProto.FLOAT16, [1, 2, 3]),
            helper.make_tensor_value_info("w_q", TensorProto.INT8, [3, 4]),
            helper.make_tensor_value_info("w_dq", TensorProto.FLOAT16, [3, 4]),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def test_adapter_runs_complete_atomic_pipeline() -> None:
    model = _qdq_model()

    returned = OnnxAdapter(AdapterConfig())(model)

    assert returned is model
    assert model.opset_import[0].version == 18
    assert [node.op_type for node in model.graph.node] == [
        "NPUAscendQuantV2",
        "MatMul",
        "AscendDequant",
    ]
    onnx.checker.check_model(model)


def test_adapter_folds_constant_subgraphs() -> None:
    model = _constant_expression_model()

    returned = OnnxAdapter(AdapterConfig(show_progress=False))(model)

    assert returned is model
    assert [node.op_type for node in model.graph.node] == ["Mul"]
    initializers = {
        tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer
    }
    assert set(initializers) == {"constant"}
    np.testing.assert_array_equal(initializers["constant"], [4.0, 6.0])
    onnx.checker.check_model(model)


def test_adapter_can_disable_constant_folding() -> None:
    model = _constant_expression_model()

    OnnxAdapter(AdapterConfig(fold_constants=False, show_progress=False))(model)

    assert [node.op_type for node in model.graph.node] == ["Add", "Mul"]
    assert {tensor.name for tensor in model.graph.initializer} == {"left", "right"}


def test_adapter_progress_and_stage_logs_can_be_controlled(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with caplog.at_level("DEBUG", logger="mdc_llm_deploy.onnx.pipeline.adapter"):
        OnnxAdapter(AdapterConfig(show_progress=True))(_identity_model())

    captured = capsys.readouterr()
    assert "Processing ONNX pipeline" in captured.out + captured.err
    assert "ONNX adapter completed" in caplog.text
    assert "ONNX final validation completed" in caplog.text
    assert "source_opset=21 fusion_passes=3 show_progress=True" in caplog.text
    assert "target_opset=18" in caplog.text
    assert "fold_constants=True" in caplog.text
    assert "fuse_rms_norm=True" in caplog.text

    OnnxAdapter(AdapterConfig(show_progress=False))(_identity_model())
    captured = capsys.readouterr()
    assert "Processing ONNX pipeline" not in captured.out + captured.err


def test_adapter_stage_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    model = _identity_model()

    def stage(name: str) -> Callable[..., object]:
        def record(_model: onnx.ModelProto, **_kwargs: object) -> object:
            calls.append(name)
            return object()

        return record

    monkeypatch.setattr(adapter_module, "lower_qdq_core", stage("lower"))
    monkeypatch.setattr(adapter_module, "_register_required_schemas", stage("register"))
    monkeypatch.setattr(adapter_module, "fold_constants_core", stage("constant_folding"))
    monkeypatch.setattr(
        adapter_module,
        "lower_opset_compatibility_core",
        stage("compatibility"),
    )
    monkeypatch.setattr(adapter_module, "downgrade_opset_core", stage("downgrade"))
    monkeypatch.setattr(adapter_module, "normalize_graph_core", stage("normalization"))
    monkeypatch.setattr(adapter_module, "run_fusion_passes", stage("fusion"))
    monkeypatch.setattr(adapter_module, "_validate_final_graph", stage("checker"))

    OnnxAdapter(AdapterConfig(show_progress=False))(model)

    assert calls == [
        "lower",
        "register",
        "constant_folding",
        "compatibility",
        "downgrade",
        "normalization",
        "fusion",
        "register",
        "checker",
    ]


def test_adapter_skips_disabled_constant_folding_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def record(_model: onnx.ModelProto) -> None:
        calls.append("constant_folding")

    monkeypatch.setattr(adapter_module, "fold_constants_core", record)

    OnnxAdapter(AdapterConfig(fold_constants=False, show_progress=False))(_identity_model())

    assert calls == []


@pytest.mark.parametrize(
    "failing_stage",
    [
        "lower",
        "register",
        "constant_folding",
        "compatibility",
        "downgrade",
        "normalization",
        "fusion",
        "checker",
    ],
)
def test_adapter_failure_rolls_back_original_model(
    monkeypatch: pytest.MonkeyPatch,
    failing_stage: str,
) -> None:
    model = _identity_model()
    original = model.SerializeToString()

    def fail(working: onnx.ModelProto, **_kwargs: object) -> None:
        working.doc_string = "mutated working clone"
        raise RuntimeError(f"{failing_stage} failed")

    target = {
        "lower": "lower_qdq_core",
        "register": "_register_required_schemas",
        "constant_folding": "fold_constants_core",
        "compatibility": "lower_opset_compatibility_core",
        "downgrade": "downgrade_opset_core",
        "normalization": "normalize_graph_core",
        "fusion": "run_fusion_passes",
        "checker": "_validate_final_graph",
    }[failing_stage]
    monkeypatch.setattr(adapter_module, target, fail)

    with pytest.raises(RuntimeError, match=f"{failing_stage} failed"):
        OnnxAdapter(AdapterConfig(show_progress=False))(model)

    assert model.SerializeToString() == original


def test_adapter_rejects_non_model() -> None:
    with pytest.raises(TypeError, match=r"onnx\.ModelProto"):
        OnnxAdapter(AdapterConfig())(object())  # type: ignore[arg-type]


def test_package_import_has_no_schema_registration_side_effect() -> None:
    code = """
import onnx
from mdc_llm_deploy.onnx.schema import ALL_SCHEMA_NAMES
import mdc_llm_deploy.onnx
for name in ALL_SCHEMA_NAMES:
    try:
        onnx.defs.get_schema(name, 18, "")
    except onnx.defs.SchemaError:
        continue
    raise AssertionError(f"schema registered during import: {name}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_adapter_registers_only_custom_schemas_present_in_graph() -> None:
    code = """
import onnx
from onnx import TensorProto, helper
from mdc_llm_deploy.onnx import AdapterConfig, OnnxAdapter
from mdc_llm_deploy.onnx.schema import ALL_SCHEMA_NAMES, RMS_NORM_OP

graph = helper.make_graph(
    [helper.make_node(RMS_NORM_OP, ["x", "gamma"], ["y", "rstd"], epsilon=1e-6)],
    "custom",
    [
        helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8]),
        helper.make_tensor_value_info("gamma", TensorProto.FLOAT16, [8]),
    ],
    [
        helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 8]),
        helper.make_tensor_value_info("rstd", TensorProto.FLOAT, [1]),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
config = AdapterConfig(
    fuse_rms_norm=False,
    fuse_apply_rotary_pos_emb=False,
    fuse_fused_infer_attention_score=False,
)
assert OnnxAdapter(config)(model) is model
assert onnx.defs.get_schema(RMS_NORM_OP, 18, "").since_version == 18
for name in ALL_SCHEMA_NAMES:
    if name == RMS_NORM_OP:
        continue
    try:
        onnx.defs.get_schema(name, 18, "")
    except onnx.defs.SchemaError:
        continue
    raise AssertionError(f"unneeded schema registered: {name}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
