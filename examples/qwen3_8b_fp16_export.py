"""Export full Qwen3-8B FP16 chunked-attention ONNX graphs."""

from __future__ import annotations

import argparse
import gc
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from onnx import TensorProto
from onnx.external_data_helper import (
    ExternalDataInfo,
    load_external_data_for_tensor,
    uses_external_data,
)
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
INLINE_CONSTANT_LIMIT = 1024


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
    return (attention_mask.to(dtype=torch.long).cumsum(dim=-1)[:, -query_length:] - 1).clamp_min(0)


class ChunkedQwen3(nn.Module):
    """Expose fixed KV buffers and return only KV produced by the current chunk."""

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model
        self.num_hidden_layers = model.config.num_hidden_layers

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
        cache = DynamicCache(
            [
                (past_key[layer_index], past_value[layer_index])
                for layer_index in range(self.num_hidden_layers)
            ]
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        updated_cache = outputs.past_key_values
        if updated_cache is None:
            raise RuntimeError("Qwen3 did not return a KV cache")
        return {
            "logits": outputs.logits,
            "present_key": torch.stack(
                [layer.keys[:, :, -query_length:, :] for layer in updated_cache.layers]
            ),
            "present_value": torch.stack(
                [layer.values[:, :, -query_length:, :] for layer in updated_cache.layers]
            ),
        }


def select_device() -> torch.device:
    """Prefer CUDA and fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_id: str) -> PreTrainedModel:
    """Load the complete pretrained Qwen3-8B FP16 model."""
    config = AutoConfig.from_pretrained(model_id)
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.set_attn_implementation("eager")
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
        config.num_hidden_layers,
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
        initial_length = initial_key.shape[3]
        if initial_key.shape != initial_value.shape:
            raise ValueError("initial key and value shapes must match")
        if initial_length != spec.valid_kv_length:
            raise ValueError(
                f"initial cache length {initial_length} does not match "
                f"valid_kv_length {spec.valid_kv_length}"
            )
        past_key[:, :, :, :initial_length, :].copy_(initial_key)
        past_value[:, :, :, :initial_length, :].copy_(initial_value)

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
        external_data=True,
    )


def adapt_without_fia(
    model: onnx.ModelProto,
    num_hidden_layers: int,
    *,
    validate: bool = True,
) -> onnx.ModelProto:
    """Apply MDC compatibility transforms while keeping Attention unfused."""
    lower_opset_compatibility(model)
    downgrade_opset(model)
    normalize_graph(model)
    optimized = optimizer.optimize(model)
    if optimized is not model:
        model.CopyFrom(optimized)
    rms_norm_result = fuse_rms_norm(model)
    rope_result = fuse_apply_rotary_pos_emb(model)
    expected_rms_norm_count = 4 * num_hidden_layers + 1
    if rms_norm_result.fused_count != expected_rms_norm_count:
        raise ValueError(
            f"Expected {expected_rms_norm_count} RMSNorm fusions, got {rms_norm_result.fused_count}"
        )
    if rope_result.fused_count != num_hidden_layers:
        raise ValueError(
            f"Expected {num_hidden_layers} RoPE fusions, got {rope_result.fused_count}"
        )
    register_schemas(RMS_NORM_OP, ROTARY_POSITION_EMBEDDING_OP)
    if validate:
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
    *,
    output_path: Path | None = None,
) -> onnx.ModelProto:
    """Export and adapt one static graph."""
    program = OnnxExporter().export(module, inputs, export_config())
    if not isinstance(program, ONNXProgram):
        raise TypeError(f"Expected ONNXProgram, got {type(program).__name__}")
    if output_path is not None:
        return _adapt_external_program(program, output_path, module.num_hidden_layers)
    return adapt_without_fia(program.model_proto, module.num_hidden_layers)


def _adapt_external_program(
    program: ONNXProgram,
    path: Path,
    num_hidden_layers: int,
) -> onnx.ModelProto:
    """Adapt a large export without serializing its weight payload in protobuf."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data_name = f"{path.name}.data"
    data_path = path.parent / data_name
    path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)
    model: onnx.ModelProto | None = None
    try:
        program.save(path, external_data=True)
        model = onnx.load(path, load_external_data=False)
        _inline_small_constants(model, path.parent)
        with chdir(path.parent):
            adapt_without_fia(model, num_hidden_layers, validate=False)
        onnx.save_model(model, path)
        onnx.checker.check_model(path, full_check=True)
        return model
    except BaseException:
        del model
        gc.collect()
        path.unlink(missing_ok=True)
        data_path.unlink(missing_ok=True)
        raise


def _inline_small_constants(model: onnx.ModelProto, base_dir: Path) -> None:
    """Load small constants needed by graph rewrites while weights stay external."""
    for tensor in model.graph.initializer:
        if not uses_external_data(tensor):
            continue
        length = ExternalDataInfo(tensor).length
        if length is None or length > INLINE_CONSTANT_LIMIT:
            continue
        load_external_data_for_tensor(tensor, str(base_dir))
        tensor.data_location = TensorProto.DEFAULT
        del tensor.external_data[:]


def run_export(
    model_id: str,
    output_dir: Path,
) -> None:
    """Load and export the full Qwen3-8B model."""
    device = select_device()
    print(f"Loading full model from {model_id} on {device}")
    model = load_model(model_id).to(device)
    module = ChunkedQwen3(model).eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    for seed, spec in enumerate((PREFILL_SPEC, DECODE_SPEC)):
        print(f"Preparing {spec.name} inputs")
        inputs = make_stage_inputs(model, spec, device, seed=seed)
        print(f"Exporting {spec.name}")
        model_path = output_dir / f"{spec.name}.onnx"
        graph = export_stage(module, inputs, output_path=model_path)
        del graph, inputs

    del module, model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/qwen3_8b_fp16_chunked"),
        help="Directory for ONNX graphs and external weights",
    )
    return parser.parse_args()


def main() -> int:
    """Run the export experiment."""
    args = parse_args()
    run_export(args.model, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
