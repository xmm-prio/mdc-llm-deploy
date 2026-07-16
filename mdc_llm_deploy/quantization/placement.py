"""Alias-aware placement contracts for quantization."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.fx import GraphModule

from ..errors import QuantizationConfigError
from ..placement import PlacementSnapshot, capture_placement
from .planner import TargetPlan


@dataclass(frozen=True, slots=True)
class AliasTargetGroup:
    """Targets sharing one parameter plus every name bound to it."""

    targets: tuple[TargetPlan, ...]
    parameter_names: tuple[str, ...]


def _owner_and_name(model: nn.Module, fqn: str) -> tuple[nn.Module, str]:
    owner_name, separator, local_name = fqn.rpartition(".")
    owner = model.get_submodule(owner_name) if separator else model
    return owner, local_name


def _contract(target: TargetPlan) -> tuple[object, ...]:
    return (
        target.target_type,
        target.algorithm,
        target.weight,
        target.percdamp,
        target.actorder,
        target.block_size,
    )


def group_alias_targets(
    graph: GraphModule,
    plan: tuple[TargetPlan, ...],
) -> tuple[AliasTargetGroup, ...]:
    """Group selected targets by parameter identity without losing aliases."""
    parameters = dict(graph.named_parameters(remove_duplicate=False))
    names_by_identity: dict[int, list[str]] = {}
    for name, parameter in parameters.items():
        names_by_identity.setdefault(id(parameter), []).append(name)

    grouped: dict[tuple[str, object], list[TargetPlan]] = {}
    order: list[tuple[str, object]] = []
    for target in plan:
        key: tuple[str, object]
        if target.parameter_name is None:
            key = ("activation", target.fqn)
        else:
            try:
                key = ("parameter", id(parameters[target.parameter_name]))
            except KeyError as error:
                raise QuantizationConfigError(
                    f"Target parameter disappeared: {target.parameter_name}"
                ) from error
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(target)

    result: list[AliasTargetGroup] = []
    for key in order:
        targets = tuple(grouped[key])
        if key[0] == "parameter":
            identity = key[1]
            if not isinstance(identity, int):
                raise RuntimeError("Invalid parameter alias identity")
            contracts = {_contract(target) for target in targets}
            if len(contracts) != 1:
                names = sorted(target.fqn for target in targets)
                raise QuantizationConfigError(
                    f"Aliased targets require one quantization contract: {names}"
                )
            parameter_names = tuple(
                sorted(names_by_identity[identity])
            )
        else:
            parameter_names = ()
        result.append(AliasTargetGroup(targets, parameter_names))
    return tuple(result)


def restore_parameter_aliases(
    graph: GraphModule,
    group: AliasTargetGroup,
) -> None:
    """Bind every alias name to the representative materialized parameter."""
    if not group.parameter_names:
        return
    representative_name = group.targets[0].parameter_name
    if representative_name is None:
        return
    owner, local_name = _owner_and_name(graph, representative_name)
    parameter = owner._parameters[local_name]
    for name in group.parameter_names:
        alias_owner, alias_name = _owner_and_name(graph, name)
        alias_owner._parameters[alias_name] = parameter


def validate_quantized_placement(
    before: PlacementSnapshot,
    graph: GraphModule,
) -> None:
    """Allow quantized dtypes while preserving names, devices, and aliases."""
    after = capture_placement(graph)
    before_by_fqn = before.by_fqn
    after_by_fqn = after.by_fqn
    missing = sorted(before_by_fqn.keys() - after_by_fqn.keys())
    if missing:
        raise QuantizationConfigError(
            f"Quantization removed resident tensors: {missing}"
        )
    changed_devices = sorted(
        name
        for name, placement in before_by_fqn.items()
        if after_by_fqn[name].device != placement.device
    )
    if changed_devices:
        raise QuantizationConfigError(
            f"Quantization changed tensor devices: {changed_devices}"
        )
    original_names = before_by_fqn.keys()
    projected_aliases = tuple(
        sorted(
            (
                tuple(name for name in group if name in original_names)
                for group in after.alias_groups
                if sum(name in original_names for name in group) > 1
            ),
            key=lambda names: names[0],
        )
    )
    if projected_aliases != before.alias_groups:
        raise QuantizationConfigError("Quantization changed parameter aliases")
    if after.hf_device_map != before.hf_device_map:
        raise QuantizationConfigError("Quantization changed hf_device_map")
