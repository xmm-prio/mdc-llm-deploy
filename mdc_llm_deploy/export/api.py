"""Static ATen FX export and transactional decode conversion."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.fx import GraphModule, Node
from torch.fx.graph import CodeGen
from torch.fx.node import map_arg

from ..errors import GraphStateError, UnsupportedPatternError
from ..graph import (
    GRAPH_SCHEMA_VERSION,
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    QuantizedTarget,
    TensorAbi,
    infer_model_kind,
    metadata,
    set_metadata,
    transactional_update,
)

_OUTPUT_NAMES = ("logits", "present.0.key", "present.0.value")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_abi(name: str, tensor: Tensor) -> TensorAbi:
    return TensorAbi(name, _dtype_name(tensor.dtype), tuple(int(item) for item in tensor.shape))


def _flatten_nodes(value: Any) -> tuple[Node, ...]:
    result: list[Node] = []

    def collect(item: Any) -> Any:
        if isinstance(item, Node):
            result.append(item)
        return item

    map_arg(value, collect)
    return tuple(result)


def _node_target(node: Node) -> str:
    target = node.target
    if hasattr(target, "_schema"):
        return str(target._schema.name)
    return str(target)


def _module_sources(node: Node) -> tuple[str, ...]:
    stack = node.meta.get("nn_module_stack", {})
    result: list[str] = []
    if isinstance(stack, Mapping):
        for key, value in stack.items():
            result.append(str(key))
            if isinstance(value, tuple) and value:
                result.append(str(value[0]))
            elif isinstance(value, str):
                result.append(value)
    return tuple(result)


def _belongs_to(node: Node, fqn: str) -> bool:
    if not fqn:
        return False
    normalized = fqn.replace(".", "_")
    return any(
        source == fqn
        or source.endswith(f".{fqn}")
        or source.endswith(f"_{normalized}")
        or fqn in source
        for source in _module_sources(node)
    )


def _structural_module_kind(module: nn.Module) -> str | None:
    children = dict(module.named_children())
    if {"q_proj", "k_proj", "v_proj", "o_proj"} <= children.keys():
        return "attention"
    if {"router", "experts", "shared_expert"} <= children.keys():
        return "moe"
    buffers = dict(module.named_buffers(recurse=False))
    if "inv_freq" in buffers:
        return "rope"
    parameters = dict(module.named_parameters(recurse=False))
    if (
        set(parameters) == {"weight"}
        and hasattr(module, "epsilon")
        and parameters["weight"].ndim >= 1
    ):
        return "rms_norm"
    return None


def _operator_signature(kind: str, node: Node) -> bool:
    target = _node_target(node)
    if kind == "rms_norm":
        return "aten::rsqrt" in target
    if kind == "rope":
        return "aten::cos" in target or "aten::sin" in target
    if kind == "attention":
        return "aten::_softmax" in target or "aten::softmax" in target
    if kind == "moe":
        return "aten::topk" in target
    return False


def _discover_boundaries(model: nn.Module, graph: GraphModule) -> tuple[FusionBoundary, ...]:
    """Discover semantic boundaries from module structure and captured operators."""
    discovered: list[tuple[str, str, tuple[str, ...]]] = []
    graph_nodes = tuple(graph.graph.nodes)
    for fqn, module in model.named_modules():
        kind = _structural_module_kind(module)
        if kind is None:
            continue
        owned = tuple(node.name for node in graph_nodes if _belongs_to(node, fqn))
        if not owned:
            owned = tuple(node.name for node in graph_nodes if _operator_signature(kind, node))
        if owned:
            discovered.append((kind, fqn, owned))

    present = {item[0] for item in discovered}
    for kind in ("rms_norm", "rope", "attention", "moe"):
        if kind in present:
            continue
        nodes = tuple(node.name for node in graph_nodes if _operator_signature(kind, node))
        if nodes:
            discovered.append((kind, f"<graph:{kind}>", nodes))
    claimed: set[str] = set()
    result: list[FusionBoundary] = []
    for kind, fqn, nodes in sorted(
        discovered,
        key=lambda item: (-item[1].count("."), item[0], item[1]),
    ):
        owned = tuple(node for node in nodes if node not in claimed)
        if not owned:
            continue
        claimed.update(owned)
        result.append(FusionBoundary(kind, fqn, owned))
    return tuple(sorted(result, key=lambda item: (item.kind, item.fqn)))


def _output_abi(graph: GraphModule) -> tuple[TensorAbi, ...]:
    output_node = next(node for node in graph.graph.nodes if node.op == "output")
    result: list[TensorAbi] = []
    for index, node in enumerate(_flatten_nodes(output_node.args[0])):
        tensor = node.meta.get("val")
        if isinstance(tensor, Tensor):
            name = _OUTPUT_NAMES[index] if index < len(_OUTPUT_NAMES) else f"output.{index}"
            result.append(_tensor_abi(name, tensor))
    if not result:
        raise UnsupportedPatternError("ATen export did not preserve output tensor metadata")
    return tuple(result)


def _model_properties(model: nn.Module, graph: GraphModule) -> dict[str, Any]:
    config = getattr(model, "config", None)
    properties: dict[str, Any] = {
        "opset": 18,
        "source": "torch.export",
        "dialect": "ATEN",
        "rms_norm_epsilon": getattr(config, "rms_norm_eps", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": getattr(config, "head_dim", None),
        "rope_theta": getattr(config, "rope_theta", None),
        "moe_intermediate_size": getattr(config, "moe_intermediate_size", None),
        "num_experts": getattr(config, "num_experts", None),
        "num_shared_experts": getattr(config, "num_shared_experts", None),
    }
    properties["aten_node_count"] = sum(
        node.op == "call_function" and "aten::" in _node_target(node)
        for node in graph.graph.nodes
    )
    return properties


def _validate_aten_graph(graph: GraphModule) -> None:
    forbidden = [
        node.name
        for node in graph.graph.nodes
        if node.op in {"call_module", "call_method"}
        or (
            node.op == "call_function"
            and "aten::" not in _node_target(node)
            and node.target is not operator.getitem
        )
    ]
    if forbidden:
        raise UnsupportedPatternError(
            f"Export must produce a functional ATen graph; found {forbidden[:4]}"
        )


def export(
    model: nn.Module,
    example_inputs: Mapping[str, Tensor],
) -> GraphModule:
    """Export an eval-mode model to a static, functional ATen FX graph."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    if model.training:
        raise ValueError("model must be in eval mode")
    if not example_inputs:
        raise ValueError("example_inputs must not be empty")
    if not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in example_inputs.items()
    ):
        raise TypeError("example_inputs must map strings to tensors")
    devices = {value.device for value in example_inputs.values()}
    devices.update(parameter.device for parameter in model.parameters())
    devices.update(buffer.device for buffer in model.buffers())
    if len(devices) > 1:
        raise ValueError("model parameters and example inputs must use one device")
    try:
        exported = torch.export.export(
            model,
            args=(),
            kwargs=dict(example_inputs),
            strict=False,
        )
        graph = cast(GraphModule, exported.module())
    except Exception as error:
        raise UnsupportedPatternError(f"ATen export failed: {error}") from error
    _validate_aten_graph(graph)
    object.__setattr__(
        graph,
        "_mdc_model_kind",
        getattr(type(model), "model_kind", None) or infer_model_kind(graph),
    )
    input_abi = tuple(_tensor_abi(name, value) for name, value in example_inputs.items())
    input_ids = next((item for item in input_abi if item.name == "input_ids"), None)
    if input_ids is None or len(input_ids.shape) != 2:
        raise UnsupportedPatternError("Static export requires rank-2 input_ids")
    value = GraphMetadata(
        schema_version=GRAPH_SCHEMA_VERSION,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind=str(graph._mdc_model_kind),
        input_abi=input_abi,
        output_abi=_output_abi(graph),
        boundaries=_discover_boundaries(model, graph),
        sequence_length=input_ids.shape[1],
        properties=_model_properties(model, graph),
    )
    set_metadata(graph, value)
    return graph


def _replace_static_sequence(value: Any, sequence: int) -> Any:
    if type(value) is int and value == sequence:
        return 1
    if isinstance(value, tuple):
        return tuple(_replace_static_sequence(item, sequence) for item in value)
    if isinstance(value, list):
        return [_replace_static_sequence(item, sequence) for item in value]
    if isinstance(value, dict):
        return {key: _replace_static_sequence(item, sequence) for key, item in value.items()}
    return value


def _cache_target(value: GraphMetadata, edge: str) -> QuantizedTarget | None:
    matches = [
        target
        for target in value.quantized_targets
        if target.target_type == "attention" and target.fqn.rsplit(".", 1)[-1] == edge
    ]
    if len(matches) > 1:
        raise UnsupportedPatternError(f"Decode conversion found multiple attention {edge} targets")
    return matches[0] if matches else None


def _register_tensor(candidate: GraphModule, name: str, value: Tensor) -> str:
    candidate.register_buffer(name, value, persistent=True)
    return name


def _qparam_tensor(
    target: QuantizedTarget,
    sequence: int,
    *,
    current: bool,
    zero_point: bool,
) -> Tensor:
    raw = target.zero_point if zero_point else target.scale
    dtype = torch.float32
    if len(raw) == 1:
        return torch.tensor(raw[0], dtype=dtype)
    if len(raw) != sequence:
        raise UnsupportedPatternError(
            f"Decode {target.fqn} parameters must be scalar or have {sequence} positions"
        )
    if current:
        return torch.tensor(raw[-1], dtype=dtype)
    return torch.tensor(raw, dtype=dtype).reshape(1, 1, sequence, 1)


def _insert_cache_quantization(
    candidate: GraphModule,
    current: Node,
    past: Node,
    target: QuantizedTarget | None,
    edge: str,
    sequence: int,
) -> tuple[Node, Node]:
    graph = candidate.graph
    if target is None:
        present = graph.call_function(
            torch.ops.aten.cat.default,
            args=([past, current], 2),
        )
        return present, present
    if target.bits != 8:
        raise UnsupportedPatternError("Decode cache supports only float or INT8")
    scale_name = _register_tensor(
        candidate,
        f"_mdc_{edge}_current_scale",
        _qparam_tensor(target, sequence, current=True, zero_point=False),
    )
    zero_name = _register_tensor(
        candidate,
        f"_mdc_{edge}_current_zero_point",
        _qparam_tensor(target, sequence, current=True, zero_point=True),
    )
    full_scale_name = _register_tensor(
        candidate,
        f"_mdc_{edge}_cache_scale",
        _qparam_tensor(target, sequence, current=False, zero_point=False),
    )
    full_zero_name = _register_tensor(
        candidate,
        f"_mdc_{edge}_cache_zero_point",
        _qparam_tensor(target, sequence, current=False, zero_point=True),
    )
    scale = graph.get_attr(scale_name)
    zero = graph.get_attr(zero_name)
    current_fp32 = graph.call_function(
        torch.ops.aten.to.dtype,
        args=(current, torch.float32),
    )
    divided = graph.call_function(torch.ops.aten.div.Tensor, args=(current_fp32, scale))
    rounded = graph.call_function(torch.ops.aten.round.default, args=(divided,))
    shifted = graph.call_function(torch.ops.aten.add.Tensor, args=(rounded, zero))
    clamped = graph.call_function(
        torch.ops.aten.clamp.default,
        args=(shifted, -128, 127),
    )
    quantized = graph.call_function(
        torch.ops.aten.to.dtype,
        args=(clamped, torch.int8),
    )
    present = graph.call_function(
        torch.ops.aten.cat.default,
        args=([past, quantized], 2),
    )
    full_scale = graph.get_attr(full_scale_name)
    full_zero = graph.get_attr(full_zero_name)
    present_fp32 = graph.call_function(
        torch.ops.aten.to.dtype,
        args=(present, torch.float32),
    )
    centered = graph.call_function(
        torch.ops.aten.sub.Tensor,
        args=(present_fp32, full_zero),
    )
    dequantized = graph.call_function(
        torch.ops.aten.mul.Tensor,
        args=(centered, full_scale),
    )
    source_dtype = current.meta.get("val")
    if isinstance(source_dtype, Tensor) and source_dtype.dtype != torch.float32:
        dequantized = graph.call_function(
            torch.ops.aten.to.dtype,
            args=(dequantized, source_dtype.dtype),
        )
    return present, dequantized


def _replace_attention_cache_users(current: Node, replacement: Node, output: Node) -> None:
    for user in tuple(current.users):
        if user is output:
            continue
        target = _node_target(user)
        if "repeat_interleave" in target:
            user.replace_input_with(current, replacement)


def _rewrite_position_nodes(candidate: GraphModule, sequence: int) -> None:
    for node in tuple(candidate.graph.nodes):
        if node.op != "call_function" or "aten::arange" not in _node_target(node):
            continue
        tensor = node.meta.get("val")
        kwargs: dict[str, Any] = {"dtype": torch.int64}
        if isinstance(tensor, Tensor):
            kwargs["device"] = tensor.device
        with candidate.graph.inserting_before(node):
            position = candidate.graph.call_function(
                torch.ops.aten.full.default,
                args=([1], sequence - 1),
                kwargs=kwargs,
            )
        node.replace_all_uses_with(position)
        candidate.graph.erase_node(node)


def _remove_prefill_causal_mask(candidate: GraphModule) -> None:
    for node in tuple(candidate.graph.nodes):
        if node.op != "call_function" or "masked_fill" not in _node_target(node):
            continue
        if node.args and isinstance(node.args[0], Node):
            node.replace_all_uses_with(node.args[0])
            candidate.graph.erase_node(node)


def convert_to_decode(graph: GraphModule) -> GraphModule:
    """Atomically rewrite a static prefill ATen graph into one-token decode."""
    current = metadata(graph)
    if not current.stage.is_prefill:
        raise GraphStateError("convert_to_decode requires a prefill graph")
    if current.sequence_length < 2:
        raise UnsupportedPatternError("Decode conversion requires sequence length >= 2")
    if not any(boundary.kind == "attention" for boundary in current.boundaries):
        raise UnsupportedPatternError("Decode conversion requires an attention boundary")

    def mutate(candidate: GraphModule) -> None:
        value = metadata(candidate)
        output = next(node for node in candidate.graph.nodes if node.op == "output")
        outputs = list(_flatten_nodes(output.args[0]))
        if len(outputs) < 3:
            raise UnsupportedPatternError(
                "Decode conversion requires logits, key, and value outputs"
            )
        current_key, current_value = outputs[1:3]
        if not any("repeat_interleave" in _node_target(user) for user in current_key.users):
            raise UnsupportedPatternError("Decode conversion cannot locate key attention use")
        if not any("repeat_interleave" in _node_target(user) for user in current_value.users):
            raise UnsupportedPatternError("Decode conversion cannot locate value attention use")
        cache_use = next(
            node
            for node in candidate.graph.nodes
            if node in current_key.users or node in current_value.users
            if "repeat_interleave" in _node_target(node)
        )

        first_compute = next(
            node for node in candidate.graph.nodes if node.op not in {"placeholder", "get_attr"}
        )
        with candidate.graph.inserting_before(first_compute):
            past_key = candidate.graph.placeholder("past_key_values_0_key")
            past_value = candidate.graph.placeholder("past_key_values_0_value")
        with candidate.graph.inserting_before(cache_use):
            present_key, attention_key = _insert_cache_quantization(
                candidate,
                current_key,
                past_key,
                _cache_target(value, "key"),
                "key",
                value.sequence_length,
            )
            present_value, attention_value = _insert_cache_quantization(
                candidate,
                current_value,
                past_value,
                _cache_target(value, "value"),
                "value",
                value.sequence_length,
            )
        _replace_attention_cache_users(current_key, attention_key, output)
        _replace_attention_cache_users(current_value, attention_value, output)
        output_values = outputs.copy()
        output_values[1] = present_key
        output_values[2] = present_value
        output.args = (tuple(output_values),)
        candidate.graph.set_codegen(CodeGen())  # type: ignore[no-untyped-call]

        _rewrite_position_nodes(candidate, value.sequence_length)
        _remove_prefill_causal_mask(candidate)
        for node in candidate.graph.nodes:
            if node.op in {"placeholder", "get_attr", "output"}:
                continue
            node.args = _replace_static_sequence(node.args, value.sequence_length)
            node.kwargs = _replace_static_sequence(dict(node.kwargs), value.sequence_length)

        key_target = _cache_target(value, "key")
        value_target = _cache_target(value, "value")
        original_inputs = tuple(
            replace(
                item,
                shape=tuple(1 if dimension == value.sequence_length else dimension for dimension in item.shape),
            )
            for item in value.input_abi
        )
        cache_shape = (1, int(value.properties.get("num_key_value_heads") or 2), value.sequence_length - 1, int(value.properties.get("head_dim") or 16))
        updated_inputs = (
            *original_inputs,
            TensorAbi(
                "past_key_values.0.key",
                "int8" if key_target is not None else value.output_abi[1].dtype,
                cache_shape,
            ),
            TensorAbi(
                "past_key_values.0.value",
                "int8" if value_target is not None else value.output_abi[2].dtype,
                cache_shape,
            ),
        )
        updated_outputs = tuple(
            replace(item, shape=(1, 1, item.shape[-1]))
            if item.name == "logits"
            else replace(
                item,
                dtype=(
                    "int8"
                    if (item.name.endswith(".key") and key_target is not None)
                    or (item.name.endswith(".value") and value_target is not None)
                    else item.dtype
                ),
            )
            for item in value.output_abi
        )
        properties = dict(value.properties)
        properties.update(
            {
                "decode_rewrite": True,
                "cache_layout": "BNSD",
                "cache_length": value.sequence_length - 1,
                "query_length": 1,
                "position_ids": (value.sequence_length - 1,),
                "mask_semantics": "all cached and current tokens visible",
            }
        )
        next_stage = (
            GraphStage.QUANTIZED_DECODE
            if value.stage.is_quantized
            else GraphStage.FLOAT_DECODE
        )
        set_metadata(
            candidate,
            replace(
                value,
                stage=next_stage,
                input_abi=updated_inputs,
                output_abi=updated_outputs,
                absolute_position=value.sequence_length - 1,
                properties=properties,
            ),
        )

    updated = transactional_update(graph, mutate)
    updated.recompile()
    return updated
