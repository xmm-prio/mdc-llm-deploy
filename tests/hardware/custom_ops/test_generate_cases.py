"""Tests for deterministic hardware case generation."""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest
import torch

from tests.hardware.custom_ops import (
    apply_rotary_pos_emb,
    fused_infer_attention_score,
    moe_expert,
    rms_norm,
)
from tests.hardware.custom_ops.common import CaseDefinition
from tests.hardware.custom_ops.generate_cases import generate_all

_EXPECTED_CUSTOM_ABIS = {
    "apply_rotary_pos_emb": ("ApplyRotaryPosEmb", 4),
    "rms_norm": ("NPURmsNorm", 2),
    "fused_infer_attention_score": ("FusedInferAttentionScore", 29),
    "moe_expert_int8": ("MoeExpert", 6),
}


@pytest.fixture(scope="module")
def generated_cases(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate cases once for serialized artifact assertions."""
    output_root = tmp_path_factory.mktemp("custom_op_cases")
    generate_all(output_root)
    return output_root


def test_generate_all_writes_complete_manifests(generated_cases: Path) -> None:
    assert {path.name for path in generated_cases.iterdir()} == set(
        _EXPECTED_CUSTOM_ABIS
    )
    for case_name in _EXPECTED_CUSTOM_ABIS:
        case_dir = generated_cases / case_name
        manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["name"] == case_name
        assert manifest["opset_version"] == 18
        assert manifest["models"] == {
            "golden": "golden.onnx",
            "custom": "custom.onnx",
        }
        assert manifest["inputs"]
        for metadata in manifest["inputs"].values():
            input_path = case_dir / metadata["path"]
            assert input_path.stat().st_size == metadata["byte_size"]
            assert metadata["byte_size"] > 0


def test_generated_onnx_models_have_expected_abis(generated_cases: Path) -> None:
    for case_name, (custom_op_type, input_slot_count) in _EXPECTED_CUSTOM_ABIS.items():
        case_dir = generated_cases / case_name
        golden = onnx.load(case_dir / "golden.onnx")
        custom = onnx.load(case_dir / "custom.onnx")
        onnx.checker.check_model(golden)
        assert [(opset.domain, opset.version) for opset in golden.opset_import] == [("", 18)]
        assert [(opset.domain, opset.version) for opset in custom.opset_import] == [("", 18)]
        nodes = [node for node in custom.graph.node if node.op_type == custom_op_type]
        assert len(nodes) == 1
        assert nodes[0].domain == ""
        assert len(nodes[0].input) == input_slot_count


def test_moe_expert_case_matches_real_mdc_types_shapes_and_empty_offset(
    generated_cases: Path,
) -> None:
    case_dir = generated_cases / "moe_expert_int8"
    custom = onnx.load(case_dir / "custom.onnx")
    golden = onnx.load(case_dir / "golden.onnx")
    node = next(node for node in custom.graph.node if node.op_type == "MoeExpert")
    inputs = {
        value.name: (
            value.type.tensor_type.elem_type,
            tuple(dimension.dim_value for dimension in value.type.tensor_type.shape.dim),
        )
        for value in custom.graph.input
    }

    assert inputs == {
        "x": (onnx.TensorProto.INT8, (1, 256)),
        "topk_ids": (onnx.TensorProto.INT16, (1, 2)),
        "topk_weight": (onnx.TensorProto.FLOAT16, (1, 2)),
        "expert_weights": (onnx.TensorProto.INT8, (3072, 256)),
        "quant_scales": (onnx.TensorProto.FLOAT, (17,)),
    }
    assert list(node.input) == [
        "x",
        "topk_ids",
        "topk_weight",
        "expert_weights",
        "quant_scales",
        "",
    ]
    assert custom.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert golden.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16


def test_attention_hardware_case_uses_fp16_qkv_with_head_dim_128(
    generated_cases: Path,
) -> None:
    custom = onnx.load(
        generated_cases / "fused_infer_attention_score" / "custom.onnx"
    )
    input_types = {
        value.name: value.type.tensor_type.elem_type for value in custom.graph.input
    }
    input_shapes = {
        value.name: tuple(
            dimension.dim_value for dimension in value.type.tensor_type.shape.dim
        )
        for value in custom.graph.input
    }

    assert input_types["query"] == onnx.TensorProto.FLOAT16
    assert input_types["key"] == onnx.TensorProto.FLOAT16
    assert input_types["value"] == onnx.TensorProto.FLOAT16
    assert input_shapes["query"] == (2, 4, 3, 128)
    assert input_shapes["key"] == (2, 2, 5, 128)
    assert input_shapes["value"] == (2, 2, 5, 128)


def test_input_generation_is_deterministic(tmp_path: Path) -> None:
    first = apply_rotary_pos_emb.generate(tmp_path / "first")
    second = apply_rotary_pos_emb.generate(tmp_path / "second")
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest == second_manifest
    for input_name in first_manifest["inputs"]:
        assert (first / f"{input_name}.bin").read_bytes() == (
            second / f"{input_name}.bin"
        ).read_bytes()


@pytest.mark.parametrize(
    "definition",
    [
        apply_rotary_pos_emb.case_definition(),
        rms_norm.case_definition(),
        fused_infer_attention_score.case_definition(),
        moe_expert.case_definition(),
    ],
    ids=list(_EXPECTED_CUSTOM_ABIS),
)
def test_golden_and_custom_models_match(definition: CaseDefinition) -> None:
    inputs = tuple(definition.inputs.values())
    expected = definition.golden_model(*inputs)
    actual = definition.custom_model(*inputs)
    expected_outputs = expected if isinstance(expected, tuple) else (expected,)
    actual_outputs = actual if isinstance(actual, tuple) else (actual,)
    assert len(expected_outputs) == len(actual_outputs)
    for expected_output, actual_output in zip(
        expected_outputs, actual_outputs, strict=True
    ):
        torch.testing.assert_close(actual_output, expected_output)
