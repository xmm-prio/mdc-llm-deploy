"""Stable contracts shared by MDC ONNX fusion passes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import onnx


@dataclass(frozen=True, slots=True)
class FusionPassResult:
    """Statistics returned by one fusion pass."""

    pass_name: str
    fused_count: int
    fused_node_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.pass_name:
            raise ValueError("pass_name must not be empty")
        if self.fused_count < 0:
            raise ValueError("fused_count must not be negative")
        if self.fused_count != len(self.fused_node_names):
            raise ValueError("fused_count must equal the number of fused node names")

    @property
    def changed(self) -> bool:
        """Return whether this pass changed the graph."""
        return self.fused_count > 0


@dataclass(frozen=True, slots=True)
class FusionReport:
    """Aggregate immutable results for a future fusion orchestrator."""

    pass_results: tuple[FusionPassResult, ...] = ()

    def __post_init__(self) -> None:
        names = [result.pass_name for result in self.pass_results]
        if len(names) != len(set(names)):
            raise ValueError("pass_results must contain unique pass names")

    @property
    def total_fused_count(self) -> int:
        """Return the total number of fused subgraphs."""
        return sum(result.fused_count for result in self.pass_results)

    @property
    def counts(self) -> Mapping[str, int]:
        """Return immutable fused counts keyed by pass name."""
        return MappingProxyType(
            {result.pass_name: result.fused_count for result in self.pass_results}
        )


class FusionPass(Protocol):
    """Structural contract implemented by an independent fusion pass."""

    @property
    def name(self) -> str:
        """Return the stable pass name."""
        ...

    def apply(self, model: onnx.ModelProto) -> FusionPassResult:
        """Rewrite supported subgraphs in place and return pass statistics."""
        ...


__all__ = ["FusionPass", "FusionPassResult", "FusionReport"]
