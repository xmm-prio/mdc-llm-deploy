"""Structural tests for QuantLinear hardware accuracy cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.hardware.onnx.quant_linear_cases import (
    QUANT_LINEAR_CASES,
    QuantLinearCase,
    case_input,
    generate,
    write_input,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def generated_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate all QuantLinear cases once."""
    return generate(tmp_path_factory.mktemp("quant_linear_hardware"))


def _initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }


def _attribute(node: onnx.NodeProto, name: str) -> int:
    return next(
        int(helper.get_attribute_value(attribute))
        for attribute in node.attribute
        if attribute.name == name
    )


def _assert_file(root: Path, spec: dict[str, str | int]) -> Path:
    path = root / str(spec["path"])
    payload = path.read_bytes()
    assert len(payload) == spec["byte_size"]
    assert hashlib.sha256(payload).hexdigest() == spec["sha256"]
    return path


def test_manifest_is_complete_and_bundle_contains_no_inputs(generated_bundle: Path) -> None:
    manifest = json.loads((generated_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["soc_version"] == "MC62CM12AA"
    assert manifest["case_count"] == len(QUANT_LINEAR_CASES)
    assert manifest["purpose"] == "quant_linear_deployment_fidelity"
    assert manifest["golden"] == "onnxruntime_raw_qdq"
    assert {item["name"] for item in manifest["cases"]} == {
        case.name for case in QUANT_LINEAR_CASES
    }
    assert not tuple(generated_bundle.rglob("*.bin"))

    for item in manifest["cases"]:
        _assert_file(generated_bundle, item["raw_model"])
        _assert_file(generated_bundle, item["adapted_model"])
        assert item["dtype"] == "float16"
        assert item["quantization"] == {
            "weight": {
                "dtype": "int8",
                "granularity": "per_channel",
                "symmetric": True,
                "axis": 1,
            },
            "activation": {
                "dtype": "int8",
                "granularity": "per_token",
                "symmetric": False,
                "axis": -2,
            },
        }


@pytest.mark.parametrize("case", QUANT_LINEAR_CASES, ids=lambda case: case.name)
def test_raw_models_have_requested_qdq_semantics(
    generated_bundle: Path,
    case: QuantLinearCase,
) -> None:
    model = onnx.load(generated_bundle / case.name / "raw.onnx")
    onnx.checker.check_model(model, full_check=True)
    initializers = _initializers(model)
    nodes = {node.name: node for node in model.graph.node}

    activation_q = nodes["activation_q"]
    activation_dq = nodes["activation_dq"]
    assert _attribute(activation_q, "axis") == -2
    assert _attribute(activation_dq, "axis") == -2
    assert activation_q.input[1:] == activation_dq.input[1:]
    activation_scale = initializers[activation_q.input[1]]
    activation_zero_point = initializers[activation_q.input[2]]
    assert activation_scale.dtype == np.float16
    assert activation_scale.shape == (case.tokens,)
    assert bool((activation_scale > 0).all())
    assert activation_zero_point.dtype == np.int8
    assert activation_zero_point.shape == (case.tokens,)
    assert bool((activation_zero_point != 0).all())
    if case.tokens > 1:
        assert np.unique(activation_zero_point).size > 1

    weight_q = nodes["weight_q"]
    weight_dq = nodes["weight_dq"]
    assert _attribute(weight_q, "axis") == 1
    assert _attribute(weight_dq, "axis") == 1
    assert weight_q.input[1:] == weight_dq.input[1:]
    weight_scale = initializers[weight_q.input[1]]
    weight_zero_point = initializers[weight_q.input[2]]
    assert weight_scale.dtype == np.float16
    assert weight_scale.shape == (case.out_features,)
    assert weight_zero_point.dtype == np.int8
    np.testing.assert_array_equal(
        weight_zero_point,
        np.zeros((case.out_features,), dtype=np.int8),
    )
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT16
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16


@pytest.mark.parametrize("case", QUANT_LINEAR_CASES, ids=lambda case: case.name)
def test_adapted_models_have_mdc_per_token_pipeline(
    generated_bundle: Path,
    case: QuantLinearCase,
) -> None:
    model = onnx.load(generated_bundle / case.name / "adapted.onnx")
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == [
        "NPUAscendQuantV2",
        "MatMul",
        "AscendDequant",
        "Sub",
        "Mul",
    ]
    quant, matmul, dequant, subtract, multiply = model.graph.node
    initializers = _initializers(model)

    assert _attribute(quant, "axis") == -2
    assert _attribute(quant, "dtype") == 2
    assert initializers[quant.input[1]].shape == (case.tokens,)
    assert initializers[quant.input[2]].shape == (case.tokens,)
    assert initializers[quant.input[2]].dtype == np.float16
    assert initializers[matmul.input[1]].dtype == np.int8
    assert initializers[matmul.input[1]].shape == (case.in_features, case.out_features)
    expected_correction = (
        initializers[quant.input[2]].astype(np.int64).reshape(case.tokens, 1)
        * initializers[matmul.input[1]].astype(np.int64).sum(axis=0).reshape(1, case.out_features)
    )
    dequant_scale = initializers[dequant.input[1]].astype(np.uint32).view(np.float32)
    expected_correction = (expected_correction * dequant_scale.reshape(1, -1)).astype(np.float16)
    expected_correction = expected_correction.reshape(1, case.tokens, case.out_features)
    np.testing.assert_array_equal(initializers[subtract.input[1]], expected_correction)
    assert dequant.input[0] == matmul.output[0]
    assert subtract.input[0] == dequant.output[0]
    assert _attribute(dequant, "dtype") == 1
    assert initializers[dequant.input[1]].shape == (case.out_features,)
    assert initializers[multiply.input[1]].shape == (1, case.tokens, 1)
    assert multiply.output[0] == model.graph.output[0].name
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT16
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
    assert not any(
        node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        and node.domain in ("", "ai.onnx")
        for node in model.graph.node
    )


@pytest.mark.parametrize("case", QUANT_LINEAR_CASES, ids=lambda case: case.name)
def test_input_recipe_is_deterministic_and_written_outside_bundle(
    tmp_path: Path,
    case: QuantLinearCase,
) -> None:
    first = case_input(case)
    second = case_input(case)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float16
    assert first.shape == case.input_shape
    assert bool(np.isfinite(first).all())

    path = write_input(case.name, tmp_path / case.name / "x.bin")
    np.testing.assert_array_equal(np.fromfile(path, dtype=np.float16).reshape(case.input_shape), first)
