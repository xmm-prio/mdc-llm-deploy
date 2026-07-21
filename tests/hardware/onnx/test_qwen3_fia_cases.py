"""Structural tests for Qwen3 FIA hardware bundle generation."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import onnx
import pytest

from mdc_llm_deploy.onnx.schemas import FUSED_INFER_ATTENTION_SCORE_OP
from tests.hardware.onnx.qwen3_fia_cases import HARDWARE_CASES, generate

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_MEBIBYTE = 1024 * 1024


@pytest.fixture(scope="module")
def generated_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the full matrix once for serialized artifact assertions."""
    return generate(tmp_path_factory.mktemp("qwen3_hardware"))


def test_hardware_matrix_covers_all_required_cases() -> None:
    actual = {
        (
            case.family.value,
            case.attention_backend.value,
            case.stage,
            str(case.dtype).removeprefix("torch."),
        )
        for case in HARDWARE_CASES
    }
    expected = set(
        product(
            ("qwen3-4b", "qwen3-30b-a3b"),
            ("eager", "sdpa"),
            ("prefill", "decode"),
            ("float16", "bfloat16"),
        )
    )

    assert actual == expected
    assert len(HARDWARE_CASES) == 16


def test_generated_manifest_is_compact_and_complete(generated_bundle: Path) -> None:
    manifest = json.loads(
        (generated_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    case_names = {case["name"] for case in manifest["cases"]}
    model_paths = {case["model"] for case in manifest["cases"]}

    assert manifest["case_count"] == 16
    assert len(case_names) == 16
    assert manifest["unique_model_count"] == len(model_paths)
    assert manifest["unique_model_count"] <= manifest["case_count"]
    assert not tuple(generated_bundle.rglob("*.bin"))
    assert sum(
        (generated_bundle / model_path).stat().st_size for model_path in model_paths
    ) < 8 * _MEBIBYTE

    for case in manifest["cases"]:
        model_path = generated_bundle / case["model"]
        assert model_path.is_file()
        assert model_path.stat().st_size == case["model_byte_size"]
        assert len(case["model_sha256"]) == 64
        assert case["inputs"]
        assert all(
            set(input_spec) == {"name", "dtype", "shape", "recipe"}
            for input_spec in case["inputs"]
        )


def test_generated_models_match_fia_abi_and_static_inputs(
    generated_bundle: Path,
) -> None:
    manifest = json.loads(
        (generated_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    loaded: dict[str, onnx.ModelProto] = {}

    for case in manifest["cases"]:
        model = loaded.setdefault(
            case["model"],
            onnx.load(generated_bundle / case["model"]),
        )
        onnx.checker.check_model(model, full_check=True)
        fia_nodes = [
            node
            for node in model.graph.node
            if node.op_type == FUSED_INFER_ATTENTION_SCORE_OP
        ]
        assert len(fia_nodes) == 1
        assert len(fia_nodes[0].input) == 31
        assert len(fia_nodes[0].output) == 2
        assert all(
            dimension.dim_value > 0 and not dimension.dim_param
            for value in model.graph.input
            for dimension in value.type.tensor_type.shape.dim
        )

        input_names = {input_spec["name"] for input_spec in case["inputs"]}
        assert input_names == {value.name for value in model.graph.input}
        if case["stage"] == "decode":
            assert sum("past_key_values.layers.0" in name for name in input_names) == 2
        else:
            assert all("past_key_values.layers.0" not in name for name in input_names)
