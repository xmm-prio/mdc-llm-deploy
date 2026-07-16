"""Compile configuration selectors into graph target work items."""

from __future__ import annotations

from dataclasses import dataclass

from torch.fx import GraphModule

from ...errors import QuantizationConfigError
from ...graph.fx.inspection import linear_weight_name
from ...graph.lifecycle import metadata
from ..config import ActivationSpec, QuantizationConfig, WeightSpec
from ..config.modifiers import (
    GPTQ_ACTORDER_DEFAULT,
    GPTQ_BLOCK_SIZE_DEFAULT,
    GPTQ_PERCDAMP_DEFAULT,
    Modifier,
)
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
    percdamp: float = GPTQ_PERCDAMP_DEFAULT
    actorder: bool = GPTQ_ACTORDER_DEFAULT
    block_size: int = GPTQ_BLOCK_SIZE_DEFAULT


def _is_moe_parameter(name: str) -> bool:
    lowered = name.lower()
    return (
        "experts." in lowered
        or "shared_expert" in lowered
        or lowered.endswith(".expert_weights")
    )


def _linear_weight_names(graph: GraphModule) -> frozenset[str]:
    """Return parameters consumed by ATen linear nodes."""
    return frozenset(
        weight_name
        for node in graph.graph.nodes
        if (weight_name := linear_weight_name(node)) is not None
    )


def _linear_parameters(graph: GraphModule) -> tuple[str, ...]:
    linear_weights = _linear_weight_names(graph)
    return tuple(
        name
        for name, parameter in graph.named_parameters(remove_duplicate=False)
        if name in linear_weights and parameter.ndim == 2 and not _is_moe_parameter(name)
    )


def _moe_parameters(graph: GraphModule) -> tuple[str, ...]:
    linear_weights = _linear_weight_names(graph)
    return tuple(
        name
        for name, parameter in graph.named_parameters(remove_duplicate=False)
        if parameter.ndim == 2
        and _is_moe_parameter(name)
        and (
            name in linear_weights
            or name.endswith(".expert_weights")
        )
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
                    percdamp=getattr(
                        modifier,
                        "percdamp",
                        GPTQ_PERCDAMP_DEFAULT,
                    ),
                    actorder=getattr(
                        modifier,
                        "actorder",
                        GPTQ_ACTORDER_DEFAULT,
                    ),
                    block_size=getattr(
                        modifier,
                        "block_size",
                        GPTQ_BLOCK_SIZE_DEFAULT,
                    ),
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
                        percdamp=getattr(
                            modifier,
                            "percdamp",
                            GPTQ_PERCDAMP_DEFAULT,
                        ),
                        actorder=getattr(
                            modifier,
                            "actorder",
                            GPTQ_ACTORDER_DEFAULT,
                        ),
                        block_size=getattr(
                            modifier,
                            "block_size",
                            GPTQ_BLOCK_SIZE_DEFAULT,
                        ),
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
