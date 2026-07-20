"""Generate all deterministic custom-operator hardware cases."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from . import (
    apply_rotary_pos_emb,
    fused_infer_attention_score,
    moe_expert,
    rms_norm,
)

Generator = Callable[[Path], object]
GENERATORS: tuple[Generator, ...] = (
    apply_rotary_pos_emb.generate,
    rms_norm.generate,
    fused_infer_attention_score.generate,
    moe_expert.generate,
)


def generate_all(output_root: Path) -> None:
    """Generate every hardware case under one output root."""
    for generator in GENERATORS:
        generator(output_root)


def main() -> None:
    """Run command-line case generation."""
    parser = argparse.ArgumentParser(
        description="生成四个 custom op 的确定性 B 端验证用例。"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/custom_ops"),
        help="输出目录, 默认 artifacts/custom_ops。",
    )
    arguments = parser.parse_args()
    generate_all(arguments.output.resolve())


if __name__ == "__main__":
    main()
