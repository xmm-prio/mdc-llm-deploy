"""Structured execution-backend capability reporting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

BackendImplementation = Literal["reference", "accelerated", "unavailable"]
_EXECUTION_DISPATCHES = ("CPU", "CUDA", "PrivateUse1")


@dataclass(frozen=True, slots=True)
class OperatorBackendStatus:
    """Implementation status for one operator and execution dispatch."""

    operator: str
    dispatch_key: str
    implementation: BackendImplementation
    registered: bool


def backend_status_snapshot(
    operator: str,
    implementations: Mapping[str, BackendImplementation],
) -> tuple[OperatorBackendStatus, ...]:
    """Build a stable immutable capability snapshot."""
    return tuple(
        OperatorBackendStatus(
            operator=operator,
            dispatch_key=dispatch_key,
            implementation=implementations.get(dispatch_key, "unavailable"),
            registered=dispatch_key in implementations,
        )
        for dispatch_key in _EXECUTION_DISPATCHES
    )
