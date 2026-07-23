"""Generate deterministic MC62 ONNX transformation ATC cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import quant_linear_cases, qwen3_fia_cases


def generate_all(output_root: Path) -> tuple[Path, ...]:
    """Generate QuantLinear accuracy and Qwen3 FIA ATC validation cases."""
    quant_linear_bundle = quant_linear_cases.generate(output_root)
    qwen3_bundle = qwen3_fia_cases.generate(output_root)
    return (quant_linear_bundle, qwen3_bundle)


def main() -> None:
    """Run command-line ATC case generation."""
    parser = argparse.ArgumentParser(description="生成 MDC ONNX 的确定性 ATC 验证模型。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/onnx"),
        help="输出目录, 默认 artifacts/onnx。",
    )
    arguments = parser.parse_args()
    generate_all(arguments.output)


if __name__ == "__main__":
    main()
