"""Plan calibration samples required by quantization materialization."""

from __future__ import annotations

from dataclasses import dataclass

from .planner import TargetPlan


@dataclass(frozen=True, slots=True)
class CalibrationPlan:
    """Immutable set of calibration sample names to retain."""

    required_fqns: frozenset[str]


def _requires_calibration(target: TargetPlan) -> bool:
    return (
        target.activation is not None
        or (
            target.algorithm == "gptq"
            and target.parameter_name is not None
        )
        or (
            target.target_type == "moe"
            and target.parameter_name is not None
            and target.parameter_name.endswith(".expert_weights")
        )
    )


def plan_calibration(targets: tuple[TargetPlan, ...]) -> CalibrationPlan:
    """Derive all calibration samples required by a target plan."""
    return CalibrationPlan(
        frozenset(
            target.fqn for target in targets if _requires_calibration(target)
        )
    )
