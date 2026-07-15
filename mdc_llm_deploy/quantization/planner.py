"""Compile configuration selectors into graph target work items."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from ..config import ActivationSpec, QuantizationConfig, WeightSpec
from ..config.modifiers import Modifier
from ..errors import QuantizationConfigError
from ..graph import metadata
from .selectors import effective_selector, selected


@dataclass(frozen=True, slots=True)
class TargetPlan:
    """One selected quantization target."""

    fqn: str
    target_type: str
    algorithm: str
    modifier_index: int
    parameter_name: str | None
    weight: WeightSpec | None
    activation: ActivationSpec | None
    percdamp: float = 0.01
    actorder: bool = True
    block_size: int = 128


def _is_moe_parameter(name: str) -> bool:
    lowered = name.lower()
    return "experts." in lowered or "shared_expert" in lowered


def _linear_weight_names(graph: GraphModule) -> frozenset[str]:
    """Return parameters consumed by ATen linear nodes."""
    return frozenset(
        str(weight.target)
        for node in graph.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.linear.default
        and len(node.args) >= 2
        and isinstance((weight := node.args[1]), Node)
        and weight.op == "get_attr"
    )


def _linear_parameters(graph: GraphModule) -> tuple[str, ...]:
    linear_weights = _linear_weight_names(graph)
    return tuple(
        name
        for name, parameter in graph.named_parameters()
        if name in linear_weights and parameter.ndim == 2 and not _is_moe_parameter(name)
    )


def _moe_parameters(graph: GraphModule) -> tuple[str, ...]:
    linear_weights = _linear_weight_names(graph)
    return tuple(
        name
        for name, parameter in graph.named_parameters()
        if name in linear_weights and parameter.ndim == 2 and _is_moe_parameter(name)
    )


def _append_parameter_targets(
    result: list[TargetPlan],
    names: tuple[str, ...],
    target_type: str,
    modifier: Modifier,
    modifier_index: int,
    root: QuantizationConfig,
) -> None:
    target = getattr(modifier, target_type)
    if target is None:
        return
    include, exclude = effective_selector(
        root.include,
        root.exclude,
        modifier.include,
        modifier.exclude,
    )
    for parameter_name in names:
        fqn = parameter_name.removesuffix(".weight")
        if selected(fqn, include, exclude):
            result.append(
                TargetPlan(
                    fqn=fqn,
                    target_type=target_type,
                    algorithm=modifier.type,
                    modifier_index=modifier_index,
                    parameter_name=parameter_name,
                    weight=target.weight,
                    activation=target.activation,
                    percdamp=getattr(modifier, "percdamp", 0.01),
                    actorder=getattr(modifier, "actorder", True),
                    block_size=getattr(modifier, "block_size", 128),
                )
            )


def _append_attention_targets(
    result: list[TargetPlan],
    graph: GraphModule,
    modifier: Modifier,
    modifier_index: int,
    root: QuantizationConfig,
) -> None:
    target = getattr(modifier, "attention", None)
    if target is None:
        return
    include, exclude = effective_selector(
        root.include,
        root.exclude,
        modifier.include,
        modifier.exclude,
    )
    attention_fqns = tuple(
        boundary.fqn
        for boundary in metadata(graph).boundaries
        if boundary.kind == "attention"
    )
    for attention_fqn in attention_fqns:
        for edge in ("query", "key", "value", "score"):
            activation = getattr(target, edge)
            fqn = f"{attention_fqn}.{edge}"
            if activation is not None and selected(fqn, include, exclude):
                result.append(
                    TargetPlan(
                        fqn=fqn,
                        target_type="attention",
                        algorithm=modifier.type,
                        modifier_index=modifier_index,
                        parameter_name=None,
                        weight=None,
                        activation=activation,
                        percdamp=getattr(modifier, "percdamp", 0.01),
                        actorder=getattr(modifier, "actorder", True),
                        block_size=getattr(modifier, "block_size", 128),
                    )
                )


def plan_quantization(
    graph: GraphModule,
    config: QuantizationConfig,
) -> tuple[TargetPlan, ...]:
    """Build a deterministic, overlap-free target plan."""
    result: list[TargetPlan] = []
    linear = _linear_parameters(graph)
    moe = _moe_parameters(graph)
    for index, modifier in enumerate(config.modifiers):
        _append_parameter_targets(result, linear, "linear", modifier, index, config)
        _append_parameter_targets(result, moe, "moe", modifier, index, config)
        _append_attention_targets(result, graph, modifier, index, config)
    ownership: dict[str, int] = {}
    for target in result:
        previous = ownership.get(target.fqn)
        if previous is not None:
            raise QuantizationConfigError(
                f"Target {target.fqn!r} is selected by modifiers "
                f"{previous} and {target.modifier_index}"
            )
        ownership[target.fqn] = target.modifier_index
    return tuple(result)
