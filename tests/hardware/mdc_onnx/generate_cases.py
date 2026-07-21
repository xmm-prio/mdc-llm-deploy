"""Generate deterministic MC62 QDQ lowering ATC cases."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.mdc_onnx import process_onnx

_K = 32
_M = 16
_N = 64


def _qdq_model(activation_zero_point: int) -> onnx.ModelProto:
    generator = np.random.default_rng(0)
    weight = generator.normal(0.0, 0.25, size=(_K, _N)).astype(np.float16)
    weight_scale = np.maximum(
        np.max(np.abs(weight.astype(np.float32)), axis=0) / 127.0,
        np.float32(1e-5),
    ).astype(np.float16)
    graph = helper.make_graph(
        [
            helper.make_node(
                "QuantizeLinear",
                ["x", "activation_scale", "activation_zero_point"],
                ["activation_q"],
                name="activation_q",
            ),
            helper.make_node(
                "DequantizeLinear",
                ["activation_q", "activation_scale", "activation_zero_point"],
                ["activation_dq"],
                name="activation_dq",
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
                name="linear",
            ),
        ],
        "mdc_qdq_linear",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, _M, _K])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, _M, _N])],
        initializer=[
            numpy_helper.from_array(np.array(0.03125, dtype=np.float16), "activation_scale"),
            numpy_helper.from_array(
                np.array(activation_zero_point, dtype=np.int8),
                "activation_zero_point",
            ),
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(weight_scale, "weight_scale"),
            numpy_helper.from_array(np.zeros((_N,), dtype=np.int8), "weight_zero_point"),
        ],
        value_info=[
            helper.make_tensor_value_info("activation_q", TensorProto.INT8, [1, _M, _K]),
            helper.make_tensor_value_info("activation_dq", TensorProto.FLOAT16, [1, _M, _K]),
            helper.make_tensor_value_info("weight_q", TensorProto.INT8, [_K, _N]),
            helper.make_tensor_value_info("weight_dq", TensorProto.FLOAT16, [_K, _N]),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def generate_all(output_root: Path) -> tuple[Path, ...]:
    """Generate symmetric and asymmetric activation cases atomically."""
    output_root = output_root.resolve()
    cases = (("symmetric", 0), ("asymmetric", 7))
    generated: list[Path] = []
    for case_name, zero_point in cases:
        case_dir = output_root / case_name
        temporary = output_root / f".{case_name}.tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        try:
            model = _qdq_model(zero_point)
            process_onnx(model)
            onnx.checker.check_model(model)
            model_path = temporary / "adapted.onnx"
            onnx.save(model, model_path)
            manifest = {
                "name": case_name,
                "activation_zero_point": zero_point,
                "soc_version": "MC62CM12AA",
                "model": model_path.name,
                "input_shape": [1, _M, _K],
                "output_shape": [1, _M, _N],
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            case_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(case_dir, ignore_errors=True)
            temporary.replace(case_dir)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        generated.append(case_dir)
    return tuple(generated)


def main() -> None:
    """Run command-line ATC case generation."""
    parser = argparse.ArgumentParser(description="生成 MDC ONNX QDQ lowering 的 ATC 验证模型。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mdc_onnx"),
        help="输出目录, 默认 artifacts/mdc_onnx。",
    )
    arguments = parser.parse_args()
    generate_all(arguments.output)


if __name__ == "__main__":
    main()
