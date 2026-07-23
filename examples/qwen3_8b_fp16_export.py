"""Export and validate one-layer Qwen3-8B FP16 chunked-attention ONNX graphs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnxscript import optimizer
from torch import nn
from torch.onnx import ONNXProgram
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, PreTrainedModel
from transformers.exporters import OnnxConfig, OnnxExporter

from mdc_llm_deploy.onnx import (
    downgrade_opset,
    lower_opset_compatibility,
    normalize_graph,
)
from mdc_llm_deploy.onnx.fusion_pass import (
    fuse_apply_rotary_pos_emb,
    fuse_rms_norm,
)
from mdc_llm_deploy.onnx.schemas import (
    FUSED_INFER_ATTENTION_SCORE_OP,
    RMS_NORM_OP,
    ROTARY_POSITION_EMBEDDING_OP,
    register_schemas,
)

MODEL_ID = "Qwen/Qwen3-8B"
PREFILL_LENGTH = 2048
KV_CAPACITY = 32000
VOCAB_SIZE = 1024
_STAGES = ("prefill", "decode")
_OUTPUT_NAMES = ("logits", "present_key", "present_value")
_EXPECTED_RMS_NORM_COUNT = 5
_EXPECTED_ROPE_COUNT = 1
_ATC_FUSION_SWITCH = "atc_fusion_switch.json"
_DISABLED_GRAPH_FUSIONS = (
    "VenBatchMatMulActEltwiseFusionPassManager",
    "VenBatchMatMulEltwiseFusionPassManager",
)


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one static chunked-attention graph."""

    name: str
    query_length: int
    valid_kv_length: int
    kv_capacity: int = KV_CAPACITY

    @property
    def attention_length(self) -> int:
        """Return physical cache plus current query length."""
        return self.kv_capacity + self.query_length


PREFILL_SPEC = StageSpec("prefill", PREFILL_LENGTH, 0)
DECODE_SPEC = StageSpec("decode", 1, PREFILL_LENGTH)


def position_ids_from_mask(
    attention_mask: torch.Tensor,
    query_length: int,
) -> torch.Tensor:
    """Derive positions for the current chunk from the valid-token mask."""
    return (
        attention_mask.to(dtype=torch.long).cumsum(dim=-1)[:, -query_length:] - 1
    ).clamp_min(0)


class ChunkedQwen3(nn.Module):
    """Expose fixed KV buffers and return only KV produced by the current chunk."""

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run one chunk without writing the returned KV into the input buffers."""
        query_length = input_ids.shape[1]
        position_ids = position_ids_from_mask(attention_mask, query_length)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=DynamicCache([(past_key, past_value)]),
            use_cache=True,
        )
        cache = outputs.past_key_values
        if cache is None:
            raise RuntimeError("Qwen3 did not return a KV cache")
        layer = cache.layers[0]
        return {
            "logits": outputs.logits,
            "present_key": layer.keys[:, :, -query_length:, :],
            "present_value": layer.values[:, :, -query_length:, :],
        }


def select_device() -> torch.device:
    """Prefer CUDA and fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_one_layer(model_id: str, vocab_size: int) -> PreTrainedModel:
    """Load pretrained Qwen3-8B weights into a one-layer, small-vocabulary model."""
    config = AutoConfig.from_pretrained(model_id)
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if vocab_size > config.vocab_size:
        raise ValueError(f"vocab_size cannot exceed source vocabulary size {config.vocab_size}")
    config.num_hidden_layers = 1
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.set_attn_implementation("eager")
    model.resize_token_embeddings(vocab_size, mean_resizing=False)
    return model.eval()


def make_stage_inputs(
    model: PreTrainedModel,
    spec: StageSpec,
    device: torch.device,
    *,
    seed: int,
    initial_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Create deterministic tensors matching one graph ABI."""
    config = model.config
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        config.vocab_size,
        (1, spec.query_length),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    cache_shape = (
        1,
        config.num_key_value_heads,
        spec.kv_capacity,
        config.head_dim,
    )
    if not 0 <= spec.valid_kv_length <= spec.kv_capacity:
        raise ValueError("valid_kv_length must be within the KV buffer")
    past_key = torch.zeros(cache_shape, dtype=model.dtype, device=device)
    past_value = torch.zeros_like(past_key)
    if initial_cache is not None:
        initial_key, initial_value = initial_cache
        initial_length = initial_key.shape[2]
        if initial_key.shape != initial_value.shape:
            raise ValueError("initial key and value shapes must match")
        if initial_length != spec.valid_kv_length:
            raise ValueError(
                f"initial cache length {initial_length} does not match "
                f"valid_kv_length {spec.valid_kv_length}"
            )
        past_key[:, :, :initial_length, :].copy_(initial_key)
        past_value[:, :, :initial_length, :].copy_(initial_value)
    elif spec.valid_kv_length:
        raise ValueError("a non-empty valid cache requires initial_cache")

    attention_mask = torch.zeros(
        (1, spec.attention_length),
        dtype=torch.long,
        device=device,
    )
    attention_mask[:, : spec.valid_kv_length] = 1
    attention_mask[:, spec.kv_capacity :] = 1
    return {
        "input_ids": input_ids,
        "past_key": past_key,
        "past_value": past_value,
        "attention_mask": attention_mask,
    }


def export_config() -> OnnxConfig:
    """Return the static ONNX export configuration."""
    return OnnxConfig(
        opset_version=21,
        optimize=False,
        dynamic=False,
        external_data=False,
    )


def adapt_without_fia(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply MDC compatibility transforms while keeping Attention unfused."""
    lower_opset_compatibility(model)
    downgrade_opset(model)
    normalize_graph(model)
    optimized = optimizer.optimize(model)
    if optimized is not model:
        model.CopyFrom(optimized)
    rms_norm_result = fuse_rms_norm(model)
    rope_result = fuse_apply_rotary_pos_emb(model)
    if rms_norm_result.fused_count != _EXPECTED_RMS_NORM_COUNT:
        raise ValueError(
            f"Expected {_EXPECTED_RMS_NORM_COUNT} RMSNorm fusions, "
            f"got {rms_norm_result.fused_count}"
        )
    if rope_result.fused_count != _EXPECTED_ROPE_COUNT:
        raise ValueError(
            f"Expected {_EXPECTED_ROPE_COUNT} RoPE fusion, got {rope_result.fused_count}"
        )
    register_schemas(RMS_NORM_OP, ROTARY_POSITION_EMBEDDING_OP)
    onnx.checker.check_model(model, full_check=True)
    operators = {node.op_type for node in model.graph.node}
    if FUSED_INFER_ATTENTION_SCORE_OP in operators:
        raise ValueError("Attention must remain unfused")
    required = {"MatMul", "Softmax"}
    if missing := required.difference(operators):
        raise ValueError(f"Attention small-operator graph is missing {sorted(missing)}")
    return model


def export_stage(
    module: ChunkedQwen3,
    inputs: dict[str, torch.Tensor],
) -> onnx.ModelProto:
    """Export and adapt one static graph."""
    program = OnnxExporter().export(module, inputs, export_config())
    if not isinstance(program, ONNXProgram):
        raise TypeError(f"Expected ONNXProgram, got {type(program).__name__}")
    return adapt_without_fia(program.model_proto)


def save_inline(model: onnx.ModelProto, path: Path) -> None:
    """Atomically save an ONNX graph with embedded weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        onnx.save_model(model, temporary_path, save_as_external_data=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_atc_fusion_switch(output_dir: Path) -> Path:
    """Disable CANN graph fusions that fail on the MC62 toolchain."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _ATC_FUSION_SWITCH
    payload = {
        "Switch": {
            "GraphFusion": dict.fromkeys(_DISABLED_GRAPH_FUSIONS, "off"),
            "UBFusion": {},
        }
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _tensor_spec(name: str, tensor: torch.Tensor, path: Path) -> dict[str, object]:
    return {
        "name": name,
        "file": path.as_posix(),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
    }


def _write_tensor(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = tensor.detach().cpu().contiguous()
    contiguous.numpy().tofile(path)


def write_validation_stage(
    root: Path,
    spec: StageSpec,
    inputs: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    onnx_path: Path,
) -> dict[str, object]:
    """Write ordered inputs and PyTorch references for one stage."""
    input_specs: list[dict[str, object]] = []
    output_specs: list[dict[str, object]] = []
    for index, (name, tensor) in enumerate(inputs.items()):
        relative = Path(spec.name) / "input" / f"{index}_{name}.bin"
        _write_tensor(tensor, root / relative)
        input_specs.append(_tensor_spec(name, tensor, relative))
    for index, name in enumerate(_OUTPUT_NAMES):
        tensor = outputs[name]
        relative = Path(spec.name) / "torch" / f"{index}_{name}.bin"
        _write_tensor(tensor, root / relative)
        output_specs.append(_tensor_spec(name, tensor, relative))
    return {
        "name": spec.name,
        "onnx": onnx_path.resolve().as_posix(),
        "inputs": input_specs,
        "outputs": output_specs,
    }


def write_manifest(root: Path, stages: Sequence[dict[str, object]]) -> Path:
    """Write the validation bundle manifest."""
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    payload = {
        "schema_version": 1,
        "kv_capacity": KV_CAPACITY,
        "prefill_length": PREFILL_LENGTH,
        "stages": list(stages),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _ordered_board_files(board_dir: Path, count: int) -> list[Path]:
    files = list(board_dir.glob("*.bin"))
    if len(files) != count:
        raise ValueError(f"Expected {count} board outputs in {board_dir}, got {len(files)}")

    def output_index(path: Path) -> tuple[int, str]:
        matches = re.findall(r"(?:output[_-]?|^)(\d+)", path.stem, flags=re.IGNORECASE)
        return (int(matches[-1]) if matches else sys.maxsize, path.name)

    return sorted(files, key=output_index)


def _numpy_dtype(name: str) -> np.dtype[Any]:
    try:
        return np.dtype(name)
    except TypeError as error:
        raise ValueError(f"Unsupported output dtype {name!r}") from error


def compare_stage(
    manifest_path: Path,
    stage_name: str,
    board_dir: Path,
    cosine_threshold: float,
) -> bool:
    """Compare one MDC output directory with PyTorch references."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = next(
        (item for item in manifest["stages"] if item["name"] == stage_name),
        None,
    )
    if stage is None:
        raise ValueError(f"Unknown stage {stage_name!r}")
    output_specs = stage["outputs"]
    board_files = _ordered_board_files(board_dir, len(output_specs))
    passed = True
    for output_spec, board_path in zip(output_specs, board_files, strict=True):
        dtype = _numpy_dtype(output_spec["dtype"])
        expected = np.fromfile(manifest_path.parent / output_spec["file"], dtype=dtype)
        actual = np.fromfile(board_path, dtype=dtype)
        shape = tuple(output_spec["shape"])
        expected_size = int(np.prod(shape))
        if expected.size != expected_size or actual.size != expected_size:
            raise ValueError(
                f"{output_spec['name']} size mismatch: expected {expected_size}, "
                f"torch={expected.size}, board={actual.size}"
            )
        expected_f32 = expected.astype(np.float32)
        actual_f32 = actual.astype(np.float32)
        denominator = np.linalg.norm(expected_f32) * np.linalg.norm(actual_f32)
        cosine = float(np.dot(expected_f32, actual_f32) / denominator) if denominator else 0.0
        difference = np.abs(expected_f32 - actual_f32)
        finite = bool(np.isfinite(actual_f32).all())
        nonzero = bool(np.any(actual_f32 != 0))
        output_passed = finite and nonzero and cosine >= cosine_threshold
        passed &= output_passed
        print(
            f"{output_spec['name']}: cosine={cosine:.6f}, "
            f"max_abs={difference.max():.6g}, mean_abs={difference.mean():.6g}, "
            f"{'PASS' if output_passed else 'FAIL'}"
        )
    return passed


def run_export(
    model_id: str,
    output_dir: Path,
    vocab_size: int,
    validation_dir: Path | None,
) -> None:
    """Load, export, and optionally generate PyTorch validation references."""
    device = select_device()
    print(f"Loading one layer from {model_id} on {device}")
    model = load_one_layer(model_id, vocab_size).to(device)
    module = ChunkedQwen3(model).eval()

    print("Preparing deterministic prefill inputs")
    prefill_inputs = make_stage_inputs(model, PREFILL_SPEC, device, seed=0)
    with torch.inference_mode():
        prefill_outputs = module(**prefill_inputs)

    print("Preparing decode inputs from prefill KV")
    decode_inputs = make_stage_inputs(
        model,
        DECODE_SPEC,
        device,
        seed=1,
        initial_cache=(
            prefill_outputs["present_key"],
            prefill_outputs["present_value"],
        ),
    )
    with torch.inference_mode():
        decode_outputs = module(**decode_inputs)

    stages: list[dict[str, object]] = []
    for spec, inputs, outputs in (
        (PREFILL_SPEC, prefill_inputs, prefill_outputs),
        (DECODE_SPEC, decode_inputs, decode_outputs),
    ):
        print(f"Exporting {spec.name}")
        graph = export_stage(module, inputs)
        model_path = output_dir / f"{spec.name}.onnx"
        save_inline(graph, model_path)
        if validation_dir is not None:
            stages.append(
                write_validation_stage(
                    validation_dir,
                    spec,
                    inputs,
                    outputs,
                    model_path,
                )
            )
        del graph

    if validation_dir is not None:
        manifest_path = write_manifest(validation_dir, stages)
        print(f"Validation manifest: {manifest_path}")
    switch_path = write_atc_fusion_switch(output_dir)
    print(f"ATC fusion switch: {switch_path}")
    del module, model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export ONNX and validation data")
    export_parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or path")
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/qwen3_8b_fp16_chunked"),
    )
    export_parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    export_parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("output/qwen3_8b_fp16_chunked/validation"),
    )

    compare_parser = subparsers.add_parser("compare", help="Compare MDC and PyTorch outputs")
    compare_parser.add_argument("--manifest", type=Path, required=True)
    compare_parser.add_argument("--stage", choices=_STAGES, required=True)
    compare_parser.add_argument("--board-output", type=Path, required=True)
    compare_parser.add_argument("--cosine-threshold", type=float, default=0.999)
    return parser.parse_args()


def main() -> int:
    """Run the selected experiment command."""
    args = parse_args()
    if args.command == "export":
        run_export(args.model, args.output_dir, args.vocab_size, args.validation_dir)
        return 0
    return 0 if compare_stage(
        args.manifest,
        args.stage,
        args.board_output,
        args.cosine_threshold,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
