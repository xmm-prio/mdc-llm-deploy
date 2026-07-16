"""Torch-independent graph metadata value types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .immutable_json import freeze_json

GRAPH_METADATA_KEY = "mdc_llm_deploy"
GRAPH_SCHEMA_VERSION = 1
ABI_DTYPES = frozenset(
    {
        "bool",
        "bfloat16",
        "float16",
        "float32",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint64",
    }
)
BOUNDARY_KINDS = frozenset(
    {"attention", "linear", "moe", "rms_norm", "rope"}
)


class GraphStage(StrEnum):
    """Supported graph lifecycle stages."""

    FLOAT_PREFILL = "FLOAT_PREFILL"
    QUANTIZED_PREFILL = "QUANTIZED_PREFILL"
    FLOAT_DECODE = "FLOAT_DECODE"
    QUANTIZED_DECODE = "QUANTIZED_DECODE"

    @property
    def is_prefill(self) -> bool:
        """Return whether this stage represents prefill."""
        return self in {self.FLOAT_PREFILL, self.QUANTIZED_PREFILL}

    @property
    def is_quantized(self) -> bool:
        """Return whether this stage represents a quantized graph."""
        return self in {
            self.QUANTIZED_PREFILL,
            self.QUANTIZED_DECODE,
        }


@dataclass(frozen=True, slots=True)
class TensorAbi:
    """Static tensor ABI entry."""

    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FusionBoundary:
    """Named graph fusion boundary."""

    kind: str
    fqn: str
    nodes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QuantizedTarget:
    """Materialized quantization state for one target."""

    fqn: str
    target_type: str
    algorithm: str
    bits: int
    granularity: str
    symmetric: bool
    scale: tuple[float, ...]
    zero_point: tuple[int, ...]
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class GraphMetadata:
    """Private, versioned metadata attached to each exported FX graph."""

    schema_version: int
    stage: GraphStage
    model_kind: str
    input_abi: tuple[TensorAbi, ...]
    output_abi: tuple[TensorAbi, ...]
    boundaries: tuple[FusionBoundary, ...] = ()
    quantized_targets: tuple[QuantizedTarget, ...] = ()
    config_fingerprint: str | None = None
    sequence_length: int = 3072
    absolute_position: int | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Capture properties as a deeply immutable snapshot."""
        object.__setattr__(
            self,
            "properties",
            freeze_json(self.properties),
        )
