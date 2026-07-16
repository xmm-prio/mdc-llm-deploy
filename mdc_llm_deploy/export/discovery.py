"""Metadata discovery for captured ATen FX graphs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.fx import GraphModule

from ..errors import UnsupportedPatternError
from ..fx_inspection import flatten_nodes, node_target
from ..fx_ownership import node_belongs_to
from ..graph_types import FusionBoundary, TensorAbi
from ..onnx_protocol import MDC_ONNX_OPSET

__all__ = ["DiscoveryResult", "discover_metadata"]


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Metadata discovered from model structure, inputs, and captured graph."""

    input_abi: tuple[TensorAbi, ...]
    output_abi: tuple[TensorAbi, ...]
    boundaries: tuple[FusionBoundary, ...]
    properties: dict[str, Any]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_abi(name: str, tensor: Tensor) -> TensorAbi:
    return TensorAbi(
        name,
        _dtype_name(tensor.dtype),
        tuple(int(item) for item in tensor.shape),
    )


def _structural_module_kind(module: nn.Module) -> str | None:
    children = dict(module.named_children())
    if {"q_proj", "k_proj", "v_proj", "o_proj"} <= children.keys():
        return "attention"
    if (
        "gate" in children
        and "expert_weights" in dict(module.named_parameters(recurse=False))
    ):
        return "moe"
    buffers = dict(module.named_buffers(recurse=False))
    if "inv_freq" in buffers or "_mdc_rotary" in buffers:
        return "rope"
    parameters = dict(module.named_parameters(recurse=False))
    if (
        set(parameters) == {"weight"}
        and hasattr(module, "epsilon")
        and parameters["weight"].ndim >= 1
    ):
        return "rms_norm"
    return None


def _strict_fqn_parent(parent: str, child: str) -> bool:
    return bool(parent) and child.startswith(f"{parent}.")


def _resolve_boundary_overlaps(
    discovered: list[tuple[str, str, tuple[str, ...]]],
) -> tuple[FusionBoundary, ...]:
    node_sets = [set(nodes) for _, _, nodes in discovered]
    for index, (_, fqn, _) in enumerate(discovered):
        for other_index in range(index + 1, len(discovered)):
            overlap = node_sets[index] & node_sets[other_index]
            if not overlap:
                continue
            other_fqn = discovered[other_index][1]
            if not (
                _strict_fqn_parent(fqn, other_fqn)
                or _strict_fqn_parent(other_fqn, fqn)
            ):
                raise UnsupportedPatternError(
                    "Overlapping fusion boundaries must have a strict FQN "
                    f"parent-child relationship: {fqn!r}, {other_fqn!r}"
                )

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


def _discover_boundaries(
    model: nn.Module,
    graph: GraphModule,
) -> tuple[FusionBoundary, ...]:
    """Discover semantic boundaries from module structure and FX ownership."""
    discovered: list[tuple[str, str, tuple[str, ...]]] = []
    graph_nodes = tuple(graph.graph.nodes)
    for fqn, module in model.named_modules():
        kind = _structural_module_kind(module)
        if kind is None:
            continue
        owned = tuple(node.name for node in graph_nodes if node_belongs_to(node, fqn))
        if owned:
            discovered.append((kind, fqn, owned))
    return _resolve_boundary_overlaps(discovered)


def _output_abi(graph: GraphModule) -> tuple[TensorAbi, ...]:
    output_node = next(node for node in graph.graph.nodes if node.op == "output")
    result: list[TensorAbi] = []
    for index, node in enumerate(flatten_nodes(output_node.args[0])):
        tensor = node.meta.get("val")
        if isinstance(tensor, Tensor):
            if index == 0:
                name = "logits"
            else:
                cache_index = index - 1
                layer_id, cache_kind = divmod(cache_index, 2)
                name = f"present.{layer_id}.{'key' if cache_kind == 0 else 'value'}"
            result.append(_tensor_abi(name, tensor))
    if not result:
        raise UnsupportedPatternError(
            "ATen export did not preserve output tensor metadata"
        )
    return tuple(result)


def _model_properties(model: nn.Module, graph: GraphModule) -> dict[str, Any]:
    config = getattr(model, "config", None)
    properties: dict[str, Any] = {
        "opset": MDC_ONNX_OPSET,
        "source": "torch.export",
        "dialect": "ATEN",
        "mask_mode": getattr(
            getattr(model, "export_config", None),
            "mask_mode",
            "causal",
        ),
        "rms_norm_epsilon": getattr(config, "rms_norm_eps", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": getattr(config, "head_dim", None),
        "rope_theta": getattr(config, "rope_theta", None),
        "moe_intermediate_size": getattr(config, "moe_intermediate_size", None),
        "num_experts": getattr(config, "num_experts", None),
        "num_experts_per_tok": getattr(config, "num_experts_per_tok", None),
    }
    properties["aten_node_count"] = sum(
        node.op == "call_function" and "aten::" in node_target(node)
        for node in graph.graph.nodes
    )
    return properties


def discover_metadata(
    model: nn.Module,
    graph: GraphModule,
    example_inputs: Mapping[str, Tensor],
) -> DiscoveryResult:
    """Discover graph metadata without mutating model or graph."""
    return DiscoveryResult(
        input_abi=tuple(
            _tensor_abi(name, value) for name, value in example_inputs.items()
        ),
        output_abi=_output_abi(graph),
        boundaries=_discover_boundaries(model, graph),
        properties=_model_properties(model, graph),
    )
