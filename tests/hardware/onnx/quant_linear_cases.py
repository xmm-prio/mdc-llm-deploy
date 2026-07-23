"""Generate deterministic QuantLinear hardware accuracy cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.onnx import process_onnx

InputRecipe = Literal["normal", "biased", "outlier"]


@dataclass(frozen=True, slots=True)
class QuantLinearCase:
    """Describe one static QuantLinear deployment accuracy case."""

    name: str
    batch: int
    tokens: int
    in_features: int
    out_features: int
    input_recipe: InputRecipe
    seed: int

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Return static activation shape."""
        return (self.batch, self.tokens, self.in_features)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Return static output shape."""
        return (self.batch, self.tokens, self.out_features)


QUANT_LINEAR_CASES = (
    QuantLinearCase("decode_normal", 1, 1, 32, 64, "normal", 11),
    QuantLinearCase("prefill_biased", 1, 16, 32, 64, "biased", 23),
    QuantLinearCase("prefill_odd_outlier", 2, 17, 64, 96, "outlier", 37),
)


def case_input(case: QuantLinearCase) -> np.ndarray:
    """Build deterministic FP16 input for one case."""
    generator = np.random.default_rng(case.seed)
    values = generator.normal(0.0, 0.65, size=case.input_shape).astype(np.float32)
    if case.input_recipe == "normal":
        values += np.float32(0.35)
    elif case.input_recipe == "biased":
        token_bias = np.linspace(-0.4, 1.4, case.tokens, dtype=np.float32)
        values += token_bias.reshape(1, case.tokens, 1)
    else:
        token_bias = np.linspace(-0.8, 0.9, case.tokens, dtype=np.float32)
        values += token_bias.reshape(1, case.tokens, 1)
        for batch in range(case.batch):
            for token in range(case.tokens):
                feature = (batch * case.tokens + token) % case.in_features
                sign = np.float32(-1.0 if token % 3 == 0 else 1.0)
                values[batch, token, feature] += sign * np.float32(5.0 + token / 8.0)
    return values.astype(np.float16)


def _activation_qparams(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reduced = values.astype(np.float32)
    minimum = np.min(reduced, axis=(0, 2))
    maximum = np.max(reduced, axis=(0, 2))
    scale = np.maximum((maximum - minimum) / np.float32(255.0), np.float32(1e-5))
    zero_point = np.clip(
        np.rint(np.float32(-128.0) - minimum / scale),
        -128,
        127,
    ).astype(np.int8)
    if np.any(zero_point == 0):
        raise ValueError("activation zero points must all be non-zero")
    if zero_point.size > 1 and np.unique(zero_point).size == 1:
        raise ValueError("activation zero points must vary across tokens")
    return scale.astype(np.float16), zero_point


def _weight(case: QuantLinearCase) -> np.ndarray:
    generator = np.random.default_rng(case.seed + 10_000)
    return generator.normal(
        0.0,
        0.25,
        size=(case.in_features, case.out_features),
    ).astype(np.float16)


def _raw_model(case: QuantLinearCase) -> onnx.ModelProto:
    values = case_input(case)
    activation_scale, activation_zero_point = _activation_qparams(values)
    weight = _weight(case)
    weight_scale = np.maximum(
        np.max(np.abs(weight.astype(np.float32)), axis=0) / np.float32(127.0),
        np.float32(1e-5),
    ).astype(np.float16)
    graph = helper.make_graph(
        [
            helper.make_node(
                "QuantizeLinear",
                ["x", "activation_scale", "activation_zero_point"],
                ["activation_q"],
                name="activation_q",
                axis=-2,
            ),
            helper.make_node(
                "DequantizeLinear",
                ["activation_q", "activation_scale", "activation_zero_point"],
                ["activation_dq"],
                name="activation_dq",
                axis=-2,
            ),
            helper.make_node(
                "QuantizeLinear",
                ["weight", "weight_scale", "weight_zero_point"],
                ["weight_q"],
                name="weight_q",
                axis=1,
            ),
            helper.make_node(
                "DequantizeLinear",
                ["weight_q", "weight_scale", "weight_zero_point"],
                ["weight_dq"],
                name="weight_dq",
                axis=1,
            ),
            helper.make_node(
                "MatMul",
                ["activation_dq", "weight_dq"],
                ["output"],
                name="quant_linear",
            ),
        ],
        f"mdc_quant_linear_{case.name}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, list(case.input_shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, list(case.output_shape))],
        initializer=[
            numpy_helper.from_array(activation_scale, "activation_scale"),
            numpy_helper.from_array(activation_zero_point, "activation_zero_point"),
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(weight_scale, "weight_scale"),
            numpy_helper.from_array(
                np.zeros((case.out_features,), dtype=np.int8),
                "weight_zero_point",
            ),
        ],
        value_info=[
            helper.make_tensor_value_info(
                "activation_q",
                TensorProto.INT8,
                list(case.input_shape),
            ),
            helper.make_tensor_value_info(
                "activation_dq",
                TensorProto.FLOAT16,
                list(case.input_shape),
            ),
            helper.make_tensor_value_info(
                "weight_q",
                TensorProto.INT8,
                [case.in_features, case.out_features],
            ),
            helper.make_tensor_value_info(
                "weight_dq",
                TensorProto.FLOAT16,
                [case.in_features, case.out_features],
            ),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def _file_spec(path: Path, root: Path) -> dict[str, str | int]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _case_manifest(case: QuantLinearCase, root: Path, case_dir: Path) -> dict[str, object]:
    raw_path = case_dir / "raw.onnx"
    adapted_path = case_dir / "adapted.onnx"
    return {
        "name": case.name,
        "dtype": "float16",
        "input_shape": list(case.input_shape),
        "output_shape": list(case.output_shape),
        "input": {
            "name": "x",
            "dtype": "float16",
            "shape": list(case.input_shape),
            "recipe": case.input_recipe,
            "seed": case.seed,
            "byte_size": int(np.prod(case.input_shape)) * np.dtype(np.float16).itemsize,
        },
        "quantization": {
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
        },
        "raw_model": _file_spec(raw_path, root),
        "adapted_model": _file_spec(adapted_path, root),
    }


def generate(output_root: Path) -> Path:
    """Generate raw/adapted models and deterministic inputs atomically."""
    output_root = output_root.resolve()
    bundle = output_root / "quant_linear"
    temporary = output_root / ".quant_linear.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        cases: list[dict[str, object]] = []
        for case in QUANT_LINEAR_CASES:
            case_dir = temporary / case.name
            case_dir.mkdir()
            raw = _raw_model(case)
            onnx.checker.check_model(raw, full_check=True)
            onnx.save(raw, case_dir / "raw.onnx")

            adapted = onnx.ModelProto()
            adapted.CopyFrom(raw)
            process_onnx(adapted)
            onnx.checker.check_model(adapted)
            onnx.save(adapted, case_dir / "adapted.onnx")
            cases.append(_case_manifest(case, temporary, case_dir))

        manifest = {
            "schema_version": 1,
            "soc_version": "MC62CM12AA",
            "case_count": len(cases),
            "purpose": "quant_linear_deployment_fidelity",
            "golden": "onnxruntime_raw_qdq",
            "cases": cases,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        bundle.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(bundle, ignore_errors=True)
        temporary.replace(bundle)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return bundle


def write_input(case_name: str, output: Path) -> Path:
    """Write one case input binary outside the generated model bundle."""
    case = next((item for item in QUANT_LINEAR_CASES if item.name == case_name), None)
    if case is None:
        raise ValueError(f"unknown QuantLinear case: {case_name}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    case_input(case).tofile(output)
    return output


def main() -> None:
    """Write a deterministic input binary for B-side validation."""
    parser = argparse.ArgumentParser(description="生成 QuantLinear B 端精度验证输入。")
    parser.add_argument("--case", required=True, choices=[case.name for case in QUANT_LINEAR_CASES])
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    write_input(arguments.case, arguments.output)


if __name__ == "__main__":
    main()


__all__ = ["QUANT_LINEAR_CASES", "QuantLinearCase", "case_input", "generate", "write_input"]
