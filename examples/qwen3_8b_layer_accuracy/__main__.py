"""Generate Qwen3 layer artifacts or compare MDC output with Torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifacts import GenerationConfig, generate_artifacts
from .metrics import compare_arrays, load_array
from .modeling import MODEL_ID, SEQUENCE_LENGTH


def _shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must contain comma-separated integers") from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise argparse.ArgumentTypeError("CUDA was requested but is not available")
    return device


def _generate(args: argparse.Namespace) -> int:
    config = GenerationConfig(
        model_id=args.model,
        sequence_length=args.sequence_length,
        cosine_threshold=args.cosine_threshold,
        process_graph=not args.skip_process,
    )
    generate_artifacts(args.output_dir, config, device=args.device)
    return 0


def _compare(args: argparse.Namespace) -> int:
    reference = load_array(args.reference)
    actual = load_array(args.actual, dtype=args.actual_dtype, shape=args.actual_shape)
    metrics = compare_arrays(reference, actual)
    result = {
        "reference": str(args.reference),
        "actual": str(args.actual),
        "threshold": args.cosine_threshold,
        "passed": metrics.finite and metrics.cosine >= args.cosine_threshold,
        "metrics": metrics.to_dict(),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate Torch and ONNX artifacts")
    generate.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or local path")
    generate.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    generate.add_argument("--cosine-threshold", type=float, default=0.999)
    generate.add_argument("--device", type=_device, default=_device("cuda" if torch.cuda.is_available() else "cpu"))
    generate.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/qwen3_8b_layer_accuracy"),
    )
    generate.add_argument(
        "--skip-process",
        action="store_true",
        help="Export raw ONNX only without the MDC graph pipeline",
    )
    generate.set_defaults(handler=_generate)

    compare = subparsers.add_parser("compare", help="Compare one MDC output with Torch")
    compare.add_argument("--reference", type=Path, required=True, help="Torch NPY reference")
    compare.add_argument("--actual", type=Path, required=True, help="MDC NPY or raw binary output")
    compare.add_argument("--actual-dtype", default="float16")
    compare.add_argument("--actual-shape", type=_shape)
    compare.add_argument("--cosine-threshold", type=float, default=0.999)
    compare.add_argument("--output", type=Path)
    compare.set_defaults(handler=_compare)
    return parser


def main() -> int:
    """Run the selected validation command."""
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
