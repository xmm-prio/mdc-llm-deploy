"""Plan calibration artifacts required by quantization materialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..config import ActivationSpec
from .planner import TargetPlan


@dataclass(frozen=True, slots=True)
class CalibrationRequirement:
    """Immutable artifacts required for one logical target."""

    activation_specs: frozenset[ActivationSpec]
    full_samples: bool


@dataclass(frozen=True, slots=True)
class CalibrationPlan:
    """Immutable calibration artifact requirements by target FQN."""

    requirements: Mapping[str, CalibrationRequirement]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirements",
            MappingProxyType(dict(self.requirements)),
        )

    @property
    def requires_collection(self) -> bool:
        """Return whether materialization requires calibration artifacts."""
        return bool(self.requirements)

    @property
    def required_fqns(self) -> frozenset[str]:
        """Return target FQNs that require calibration."""
        return frozenset(self.requirements)


def _requires_full_samples(target: TargetPlan) -> bool:
    return (
        target.algorithm == "gptq"
        and target.parameter_name is not None
    ) or (
        target.target_type == "moe"
        and target.parameter_name is not None
        and target.parameter_name.endswith(".expert_weights")
    )


def plan_calibration(targets: tuple[TargetPlan, ...]) -> CalibrationPlan:
    """Derive calibration artifact requirements from target plans."""
    requirements: dict[str, CalibrationRequirement] = {}
    for target in targets:
        full_samples = _requires_full_samples(target)
        if target.activation is None and not full_samples:
            continue
        previous = requirements.get(
            target.fqn,
            CalibrationRequirement(frozenset(), False),
        )
        activation_specs = previous.activation_specs
        if target.activation is not None:
            activation_specs = activation_specs | {target.activation}
        requirements[target.fqn] = CalibrationRequirement(
            activation_specs=frozenset(activation_specs),
            full_samples=previous.full_samples or full_samples,
        )
    return CalibrationPlan(requirements)
