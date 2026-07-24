"""Export Qwen3-4B FP16 TP=2 rank graphs with HCCL-style all-reduce nodes."""

from __future__ import annotations

import argparse
import gc
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper
from onnx.defs import OpSchema
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

from mdc_llm_deploy.onnx import AdapterConfig, OnnxAdapter
from mdc_llm_deploy.onnx.schema import register_schema_objects

MODEL_ID = "Qwen/Qwen3-4B"
TP_SIZE = 2
PREFILL_LENGTH = 2048
KV_CAPACITY = 32000
VOCAB_SIZE = 1024
INLINE_CONSTANT_LIMIT = 1024
HCOM_ALL_GATHER_OP = "HcomAllGather"
HCOM_GROUP = "hccl_sub_group"
MDC_ONNX_OPSET = 18


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one static chunked-attention graph."""

    name: str
    query_length: int
    valid_kv_length: int
    kv_capacity: int

    @property
    def attention_length(self) -> int:
        """Return physical KV-cache length."""
        return self.kv_capacity


class ScatterCache(DynamicCache):
    """Write current K/V tensors into fixed-capacity layer buffers."""

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
        """Scatter current K/V states and return full buffers."""
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
    """Expose fixed K/V buffers around a rank-local Qwen3 model."""

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
        """Run one chunk using rank-local attention heads and fixed caches."""
        query_length = input_ids.shape[1]
        position_ids = index.unsqueeze(1) + torch.arange(
            query_length,
            device=index.device,
        ).unsqueeze(0)
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


def _validate_divisible(value: int, name: str) -> None:
    if value % TP_SIZE:
        raise ValueError(f"{name}={value} must be divisible by TP size {TP_SIZE}")


def _shard_colwise(linear: nn.Linear, rank: int) -> None:
    """Keep one contiguous output-feature shard."""
    _validate_divisible(linear.out_features, "out_features")
    shard_size = linear.out_features // TP_SIZE
    start = rank * shard_size
    stop = start + shard_size
    linear.weight = nn.Parameter(
        linear.weight.detach()[start:stop].contiguous(),
        requires_grad=False,
    )
    if linear.bias is not None:
        linear.bias = nn.Parameter(
            linear.bias.detach()[start:stop].contiguous(),
            requires_grad=False,
        )
    linear.out_features = shard_size


def _shard_rowwise(linear: nn.Linear, rank: int) -> None:
    """Keep one contiguous input-feature shard."""
    _validate_divisible(linear.in_features, "in_features")
    shard_size = linear.in_features // TP_SIZE
    start = rank * shard_size
    stop = start + shard_size
    linear.weight = nn.Parameter(
        linear.weight.detach()[:, start:stop].contiguous(),
        requires_grad=False,
    )
    linear.in_features = shard_size


def apply_tp_sharding(model: PreTrainedModel, rank: int) -> None:
    """Apply Qwen3's inference TP plan in place without distributed runtime."""
    if not 0 <= rank < TP_SIZE:
        raise ValueError(f"rank must be within [0, {TP_SIZE}), got {rank}")
    config = model.config
    _validate_divisible(config.num_attention_heads, "num_attention_heads")
    _validate_divisible(config.num_key_value_heads, "num_key_value_heads")

    for layer in model.model.layers:
        attention = layer.self_attn
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
            _shard_colwise(projection, rank)
        _shard_rowwise(attention.o_proj, rank)

        mlp = layer.mlp
        _shard_colwise(mlp.gate_proj, rank)
        _shard_colwise(mlp.up_proj, rank)
        _shard_rowwise(mlp.down_proj, rank)

    config.num_attention_heads //= TP_SIZE
    config.num_key_value_heads //= TP_SIZE


def load_rank_model(
    model_id: str,
    rank: int,
    *,
    num_hidden_layers: int | None,
    vocab_size: int | None,
) -> PreTrainedModel:
    """Load FP16 weights, optionally shrink validation scope, then shard one rank."""
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
    if vocab_size is not None:
        if not 1 <= vocab_size <= model.config.vocab_size:
            raise ValueError(
                f"vocab_size must be within [1, {model.config.vocab_size}], got {vocab_size}"
            )
        model.resize_token_embeddings(vocab_size, mean_resizing=False)
    apply_tp_sharding(model, rank)
    return model.eval()


def make_stage_inputs(
    model: PreTrainedModel,
    spec: StageSpec,
    device: torch.device,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Create deterministic tensors for one rank-local static graph."""
    if spec.query_length <= 0 or spec.kv_capacity <= 0:
        raise ValueError("query_length and kv_capacity must be positive")
    if not 0 <= spec.valid_kv_length < spec.kv_capacity:
        raise ValueError("valid_kv_length must be within the KV buffer")
    if spec.valid_kv_length + spec.query_length > spec.kv_capacity:
        raise ValueError("query chunk exceeds KV buffer capacity")

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
    past_key = torch.zeros(cache_shape, dtype=model.dtype, device=device)
    past_value = torch.zeros_like(past_key)
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
    """Return static external-data export settings."""
    return OnnxConfig(
        opset_version=21,
        optimize=False,
        dynamic=False,
        external_data=True,
    )


def _hcom_schema() -> OpSchema:
    parameter = OpSchema.FormalParameter
    attribute = OpSchema.Attribute
    return OpSchema(
        HCOM_ALL_GATHER_OP,
        "",
        MDC_ONNX_OPSET,
        doc="Gather equal tensors from an HCCL group along the leading dimension.",
        inputs=[parameter("x", "T")],
        outputs=[parameter("y", "T")],
        type_constraints=[
            (
                "T",
                ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"],
                "Supported floating-point tensor types.",
            )
        ],
        attributes=[
            attribute("fusion", helper.make_attribute("fusion", 0), "Fusion mode."),
            attribute(
                "fusion_id",
                helper.make_attribute("fusion_id", -1),
                "Fusion identifier.",
            ),
            attribute(
                "group",
                helper.make_attribute("group", HCOM_GROUP),
                "HCCL communication group.",
            ),
            attribute(
                "rank_size",
                OpSchema.AttrType.INT,
                "Number of ranks.",
                required=True,
            ),
        ],
    )


def register_hcom_schema() -> None:
    """Register process-local default-domain HcomAllGather schema."""
    register_schema_objects((_hcom_schema(),))


def _name_scopes(node: onnx.NodeProto) -> str:
    metadata = {item.key: item.value for item in node.metadata_props}
    return metadata.get("pkg.torch.onnx.name_scopes", "")


def inject_tp_all_reduce(model: onnx.ModelProto, num_hidden_layers: int) -> None:
    """Insert HcomAllGather plus ReduceSum after each rowwise TP projection."""
    graph = model.graph
    axes_name = "tp_all_reduce_axes"
    existing_names = {initializer.name for initializer in graph.initializer}
    if axes_name in existing_names:
        raise ValueError(f"initializer name collision: {axes_name}")
    graph.initializer.append(
        helper.make_tensor(
            axes_name,
            TensorProto.INT64,
            dims=[1],
            vals=[0],
        )
    )

    rewritten: list[onnx.NodeProto] = []
    injected = 0
    for node in graph.node:
        rewritten.append(node)
        scopes = _name_scopes(node)
        is_rowwise = (
            ".self_attn.o_proj" in scopes or ".mlp.down_proj" in scopes
        ) and node.op_type in {"MatMul", "Gemm"}
        if not is_rowwise:
            continue
        if len(node.output) != 1:
            raise ValueError(f"rowwise projection {node.name!r} must have one output")
        original_output = node.output[0]
        local_output = f"{original_output}__tp_local"
        gathered_output = f"{original_output}__tp_gathered"
        node.output[0] = local_output
        suffix = f"layer_{injected // 2}_{'attention' if injected % 2 == 0 else 'mlp'}"
        rewritten.append(
            helper.make_node(
                HCOM_ALL_GATHER_OP,
                [local_output],
                [gathered_output],
                name=f"HcomAllGather_{suffix}",
                domain="",
                fusion=0,
                fusion_id=-1,
                group=HCOM_GROUP,
                rank_size=TP_SIZE,
            )
        )
        rewritten.append(
            helper.make_node(
                "ReduceSum",
                [gathered_output, axes_name],
                [original_output],
                name=f"ReduceSum_{suffix}",
                domain="",
                keepdims=1,
            )
        )
        injected += 1

    expected = 2 * num_hidden_layers
    if injected != expected:
        raise ValueError(f"Expected {expected} rowwise TP projections, found {injected}")
    del graph.node[:]
    graph.node.extend(rewritten)


def _inline_small_constants(model: onnx.ModelProto, base_dir: Path) -> None:
    """Load small constants required by graph transforms."""
    for tensor in model.graph.initializer:
        if not uses_external_data(tensor):
            continue
        length = ExternalDataInfo(tensor).length
        if length is None or length > INLINE_CONSTANT_LIMIT:
            continue
        load_external_data_for_tensor(tensor, str(base_dir))
        tensor.data_location = TensorProto.DEFAULT
        del tensor.external_data[:]


def _adapt_and_save(
    program: ONNXProgram,
    path: Path,
    num_hidden_layers: int,
) -> onnx.ModelProto:
    """Inject collectives, run MDC adaptation, and save one external-data graph."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data_path = path.parent / f"{path.name}.data"
    path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)
    model: onnx.ModelProto | None = None
    try:
        optimizer.optimize(program.model)
        program.save(path, external_data=True)
        model = onnx.load(path, load_external_data=False)
        _inline_small_constants(model, path.parent)
        inject_tp_all_reduce(model, num_hidden_layers)
        with chdir(path.parent):
            OnnxAdapter(
                AdapterConfig(
                    fuse_fused_infer_attention_score=False,
                    show_progress=False,
                )
            )(model)
        onnx.save_model(model, path)
        onnx.checker.check_model(path, full_check=True)
        return model
    except BaseException:
        del model
        gc.collect()
        path.unlink(missing_ok=True)
        data_path.unlink(missing_ok=True)
        raise


def export_stage(
    module: ChunkedQwen3,
    inputs: dict[str, torch.Tensor],
    path: Path,
) -> onnx.ModelProto:
    """Export and save one rank-local static stage."""
    program = OnnxExporter().export(module, inputs, export_config())
    if not isinstance(program, ONNXProgram):
        raise TypeError(f"Expected ONNXProgram, got {type(program).__name__}")
    return _adapt_and_save(program, path, module.num_hidden_layers)


def _attribute_values(node: onnx.NodeProto) -> dict[str, object]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def validate_export(
    model: onnx.ModelProto,
    path: Path,
    *,
    num_hidden_layers: int,
) -> None:
    """Validate TP communication structure and serialized external data."""
    operators = [node.op_type for node in model.graph.node]
    if "ReduceScatter" in operators:
        raise ValueError("ReduceScatter must not remain in exported graph")
    hcom_nodes = [
        node for node in model.graph.node if node.op_type == HCOM_ALL_GATHER_OP
    ]
    expected = 2 * num_hidden_layers
    if len(hcom_nodes) != expected:
        raise ValueError(f"Expected {expected} HcomAllGather nodes, got {len(hcom_nodes)}")
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
    }
    for node in hcom_nodes:
        attributes = _attribute_values(node)
        expected_attributes = {
            "fusion": 0,
            "fusion_id": -1,
            "group": HCOM_GROUP.encode(),
            "rank_size": TP_SIZE,
        }
        if attributes != expected_attributes:
            raise ValueError(f"Invalid {node.name} attributes: {attributes}")
        consumers = [
            candidate
            for candidate in model.graph.node
            if node.output[0] in candidate.input
        ]
        if len(consumers) != 1 or consumers[0].op_type != "ReduceSum":
            raise ValueError(f"{node.name} must feed exactly one ReduceSum")
        if producers.get(node.input[0]) is None:
            raise ValueError(f"{node.name} has no rank-local producer")

    if not path.is_file():
        raise FileNotFoundError(path)
    data_path = path.parent / f"{path.name}.data"
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not any(uses_external_data(tensor) for tensor in model.graph.initializer):
        raise ValueError("exported model has no external initializers")
    onnx.checker.check_model(path, full_check=True)


def run_export(
    model_id: str,
    output_dir: Path,
    *,
    num_hidden_layers: int | None,
    vocab_size: int | None,
    prefill_length: int,
    kv_capacity: int,
) -> None:
    """Sequentially export rank0 and rank1 without distributed initialization."""
    if prefill_length <= 0:
        raise ValueError("prefill_length must be positive")
    if kv_capacity <= prefill_length:
        raise ValueError("kv_capacity must be greater than prefill_length")
    register_hcom_schema()
    device = select_device()
    specs = (
        StageSpec("prefill", prefill_length, 0, kv_capacity),
        StageSpec("decode", 1, prefill_length, kv_capacity),
    )

    for rank in range(TP_SIZE):
        layer_scope = "full" if num_hidden_layers is None else f"{num_hidden_layers}-layer"
        print(f"Loading {layer_scope} rank{rank} model from {model_id} on {device}")
        model = load_rank_model(
            model_id,
            rank,
            num_hidden_layers=num_hidden_layers,
            vocab_size=vocab_size,
        ).to(device)
        module = ChunkedQwen3(model).eval()
        rank_dir = output_dir / f"rank{rank}"
        for seed, spec in enumerate(specs):
            print(f"Exporting rank{rank} {spec.name}")
            inputs = make_stage_inputs(model, spec, device, seed=seed)
            path = rank_dir / f"{spec.name}.onnx"
            graph = export_stage(module, inputs, path)
            validate_export(
                graph,
                path,
                num_hidden_layers=module.num_hidden_layers,
            )
            del graph, inputs
            gc.collect()
        del module, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/qwen3_4b_fp16_tp2"),
        help="Directory for rank-local ONNX graphs",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Limit loaded layers; use 1 for fast validation",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help=f"Retain leading vocabulary rows; use {VOCAB_SIZE} for fast validation",
    )
    parser.add_argument(
        "--prefill-length",
        type=int,
        default=PREFILL_LENGTH,
        help="Static prefill query length",
    )
    parser.add_argument(
        "--kv-capacity",
        type=int,
        default=KV_CAPACITY,
        help="Static physical KV-cache capacity",
    )
    return parser.parse_args()


def main() -> int:
    """Run sequential TP rank export."""
    args = parse_args()
    run_export(
        args.model,
        args.output_dir,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=args.vocab_size,
        prefill_length=args.prefill_length,
        kv_capacity=args.kv_capacity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
