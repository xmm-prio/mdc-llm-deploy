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

# Keep the default as the full release model. Routine validation must use
# --num-hidden-layers 2; full-network export is too slow for smoke testing.

@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one static chunked-attention graph."""

    name: str
    query_length: int
    valid_kv_length: int
    kv_capacity: int = KV_CAPACITY

    @property
    def attention_length(self) -> int:
        """Return the fixed physical cache length."""
        return self.kv_capacity


PREFILL_SPEC = StageSpec("prefill", PREFILL_LENGTH, 0)
DECODE_SPEC = StageSpec("decode", 1, PREFILL_LENGTH)


def position_ids_from_index(
    index: torch.Tensor,
    query_length: int,
) -> torch.Tensor:
    """Derive current chunk positions from its cache start index."""
    return index.unsqueeze(1) + torch.arange(query_length, device=index.device).unsqueeze(0)


class ScatterCache(DynamicCache):
    """Write each layer's current KV tensors into fixed-capacity buffers."""

    def __init__(
        self,
        cache_data: list[tuple[torch.Tensor, torch.Tensor]],
        index: torch.Tensor,
    ) -> None:
        super().__init__(cache_data)
        self.index = index

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter current KV tensors at ``index`` and return full buffers."""
        del args, kwargs
        positions = self.index + torch.arange(
            key_states.shape[-2],
            device=key_states.device,
        )
        scatter_index = positions.view(1, 1, -1, 1).expand_as(key_states)
        layer = self.layers[layer_idx]
        layer.keys = layer.keys.scatter(-2, scatter_index, key_states)
        layer.values = layer.values.scatter(-2, scatter_index, value_states)
        return layer.keys, layer.values


class ChunkedQwen3(nn.Module):
    """Expose fixed KV buffers updated before each layer's Attention."""

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
        index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Scatter one chunk into fixed KV buffers and run Attention."""
        query_length = input_ids.shape[1]
        position_ids = position_ids_from_index(index, query_length)
        cache = ScatterCache(
            [
                (past_key[layer_index], past_value[layer_index])
                for layer_index in range(self.num_hidden_layers)
            ],
            index,
        )
        key_positions = torch.arange(
            past_key.shape[-2],
            device=input_ids.device,
        ).view(1, 1, 1, -1)
        query_positions = position_ids.view(1, 1, query_length, 1)
        visible = (key_positions <= query_positions) & attention_mask[:, None, None, :].bool()
        attention_bias = torch.where(
            visible,
            torch.zeros((), dtype=self.model.dtype, device=input_ids.device),
            torch.full(
                (),
                torch.finfo(self.model.dtype).min,
                dtype=self.model.dtype,
                device=input_ids.device,
            ),
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask={"full_attention": attention_bias},
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        updated_cache = outputs.past_key_values
        if updated_cache is None:
            raise RuntimeError("Qwen3 did not return a KV cache")
        return {
            "logits": outputs.logits,
            "present_key": torch.stack([layer.keys for layer in updated_cache.layers]),
            "present_value": torch.stack([layer.values for layer in updated_cache.layers]),
        }


def select_device() -> torch.device:
    """Prefer CUDA and fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_id: str,
    *,
    num_hidden_layers: int | None = None,
) -> PreTrainedModel:
    """Load pretrained Qwen3-8B FP16 weights with an optional layer limit."""
    config = AutoConfig.from_pretrained(model_id)
    configured_layers = int(config.num_hidden_layers)
    if num_hidden_layers is not None:
        if not 1 <= num_hidden_layers <= configured_layers:
            raise ValueError(
                f"num_hidden_layers must be within [1, {configured_layers}], "
                f"got {num_hidden_layers}"
            )
        config.num_hidden_layers = num_hidden_layers
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
        if initial_length not in (spec.valid_kv_length, spec.kv_capacity):
            raise ValueError(
                f"initial cache length {initial_length} must match valid_kv_length "
                f"{spec.valid_kv_length} or kv_capacity {spec.kv_capacity}"
            )
        past_key[:, :, :, :initial_length, :].copy_(initial_key)
        past_value[:, :, :, :initial_length, :].copy_(initial_value)

    attention_mask = torch.zeros(
        (1, spec.attention_length),
        dtype=torch.long,
        device=device,
    )
    attention_mask[:, : spec.valid_kv_length + spec.query_length] = 1
    return {
        "input_ids": input_ids,
        "past_key": past_key,
        "past_value": past_value,
        "attention_mask": attention_mask,
        "index": torch.tensor([spec.valid_kv_length], dtype=torch.long, device=device),
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
    optimize: bool = True,
    validate: bool = True,
) -> onnx.ModelProto:
    """Apply MDC compatibility transforms while keeping Attention unfused."""
    lower_opset_compatibility(model)
    downgrade_opset(model)
    normalize_graph(model)
    if optimize:
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
        optimizer.optimize(program.model)
        program.save(path, external_data=True)
        model = onnx.load(path, load_external_data=False)
        _inline_small_constants(model, path.parent)
        with chdir(path.parent):
            adapt_without_fia(
                model,
                num_hidden_layers,
                optimize=False,
                validate=False,
            )
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
    *,
    num_hidden_layers: int | None = None,
) -> None:
    """Load and export Qwen3-8B with an optional smoke-test layer limit."""
    device = select_device()
    layer_scope = "full" if num_hidden_layers is None else f"{num_hidden_layers}-layer"
    print(f"Loading {layer_scope} model from {model_id} on {device}")
    model = load_model(model_id, num_hidden_layers=num_hidden_layers).to(device)
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
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Limit loaded layers for smoke validation; use 2 instead of validating full 8B",
    )
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
    run_export(
        args.model,
        args.output_dir,
        num_hidden_layers=args.num_hidden_layers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
