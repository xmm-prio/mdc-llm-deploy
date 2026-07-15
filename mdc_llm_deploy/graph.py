"""Versioned graph contract and transactional state transitions."""

from __future__ import annotations

import copy
import math
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar

from .capabilities import (
    Algorithm,
    Artifact,
    Capability,
    MaskMode,
    ModelKind,
    Phase,
    Target,
    require_capability,
)
from .errors import GraphStateError, UnsupportedPatternError

if TYPE_CHECKING:
    from torch.fx import GraphModule

GRAPH_METADATA_KEY = "mdc_llm_deploy"
GRAPH_SCHEMA_VERSION = 1
_ABI_DTYPES = {
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
_BOUNDARY_KINDS = {"attention", "linear", "moe", "rms_norm", "rope"}


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
        return self in {self.QUANTIZED_PREFILL, self.QUANTIZED_DECODE}


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
    properties: dict[str, Any] = field(default_factory=dict)


def metadata(graph: GraphModule) -> GraphMetadata:
    """Return and validate graph metadata."""
    value = graph.meta.get(GRAPH_METADATA_KEY)
    if not isinstance(value, GraphMetadata):
        raise GraphStateError("Graph does not carry MDC metadata")
    validate_metadata(value)
    return value


def set_metadata(graph: GraphModule, value: GraphMetadata) -> None:
    """Attach validated graph metadata."""
    validate_metadata(value)
    graph.meta[GRAPH_METADATA_KEY] = value


def validate_metadata(value: GraphMetadata) -> None:
    """Validate cross-module graph contract invariants."""
    if not isinstance(value, GraphMetadata):
        raise GraphStateError("Metadata must use GraphMetadata")
    if isinstance(value.schema_version, bool) or not isinstance(value.schema_version, int):
        raise GraphStateError("schema_version must be an integer")
    if value.schema_version != GRAPH_SCHEMA_VERSION:
        raise GraphStateError(
            f"Unsupported graph metadata version: {value.schema_version}"
        )
    if not isinstance(value.stage, GraphStage):
        raise GraphStateError("stage must use GraphStage")
    try:
        model_kind = ModelKind(value.model_kind)
    except (TypeError, ValueError) as error:
        raise GraphStateError(f"Unsupported model kind: {value.model_kind!r}") from error
    if (
        isinstance(value.sequence_length, bool)
        or not isinstance(value.sequence_length, int)
        or value.sequence_length <= 0
    ):
        raise GraphStateError("sequence_length must be positive")
    _validate_abi("Input", value.input_abi, required=True)
    _validate_abi("Output", value.output_abi, required=False)
    _validate_boundaries(value.boundaries)
    algorithms = _validate_quantized_targets(value.quantized_targets)
    _validate_properties(value.properties)

    if value.stage.is_prefill:
        if value.absolute_position is not None:
            raise GraphStateError("Prefill graph must not carry absolute_position")
    elif value.absolute_position != value.sequence_length - 1:
        raise GraphStateError(
            "Decode graph absolute_position must equal sequence_length - 1"
        )

    if value.stage.is_quantized:
        if not value.quantized_targets:
            raise GraphStateError("Quantized graph must carry target metadata")
        if not _valid_fingerprint(value.config_fingerprint):
            raise GraphStateError(
                "Quantized graph must carry a lowercase SHA-256 config fingerprint"
            )
    elif value.quantized_targets or value.config_fingerprint is not None:
        raise GraphStateError(
            "Float graph must not carry quantized targets or a config fingerprint"
        )

    phase = Phase.PREFILL if value.stage.is_prefill else Phase.DECODE
    for target in value.quantized_targets:
        try:
            require_capability(
                model_kind,
                Algorithm(target.algorithm),
                Target(target.target_type),
                phase,
                MaskMode.MASKED,
                Artifact.FX,
            )
        except (TypeError, ValueError, UnsupportedPatternError) as error:
            raise GraphStateError(str(error)) from error
    if not value.stage.is_quantized and algorithms:
        raise GraphStateError("Float graph must not declare quantization algorithms")


def _validate_abi(
    label: str,
    entries: tuple[TensorAbi, ...],
    *,
    required: bool,
) -> None:
    """Validate one ordered tensor ABI."""
    if not isinstance(entries, tuple) or not all(
        isinstance(item, TensorAbi) for item in entries
    ):
        raise GraphStateError(f"{label} ABI must be a tuple of TensorAbi")
    if required and not entries:
        raise GraphStateError(f"{label} ABI must not be empty")
    names: list[str] = []
    for item in entries:
        if not isinstance(item.name, str) or not item.name:
            raise GraphStateError(f"{label} ABI names must be non-empty strings")
        if item.dtype not in _ABI_DTYPES:
            raise GraphStateError(f"Unsupported ABI dtype: {item.dtype!r}")
        if not isinstance(item.shape, tuple) or not item.shape:
            raise GraphStateError(f"{label} ABI shapes must be non-empty tuples")
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in item.shape
        ):
            raise GraphStateError(f"{label} ABI shapes must be static and positive")
        names.append(item.name)
    if len(names) != len(set(names)):
        raise GraphStateError(f"{label} ABI names must be unique")


def _validate_boundaries(boundaries: tuple[FusionBoundary, ...]) -> None:
    """Validate fusion boundary names and ownership."""
    if not isinstance(boundaries, tuple) or not all(
        isinstance(item, FusionBoundary) for item in boundaries
    ):
        raise GraphStateError("boundaries must be a tuple of FusionBoundary")
    identities: set[tuple[str, str]] = set()
    claimed_nodes: set[str] = set()
    for boundary in boundaries:
        if boundary.kind not in _BOUNDARY_KINDS:
            raise GraphStateError(f"Unsupported fusion boundary kind: {boundary.kind!r}")
        if not isinstance(boundary.fqn, str) or not boundary.fqn:
            raise GraphStateError("Fusion boundary FQN must be non-empty")
        identity = (boundary.kind, boundary.fqn)
        if identity in identities:
            raise GraphStateError("Fusion boundary kind/FQN pairs must be unique")
        identities.add(identity)
        if not isinstance(boundary.nodes, tuple) or not all(
            isinstance(node, str) and node for node in boundary.nodes
        ):
            raise GraphStateError("Fusion boundary nodes must be non-empty strings")
        if len(boundary.nodes) != len(set(boundary.nodes)):
            raise GraphStateError("Fusion boundary nodes must be unique")
        overlap = claimed_nodes.intersection(boundary.nodes)
        if overlap:
            raise GraphStateError(
                f"Fusion boundary nodes have multiple owners: {sorted(overlap)}"
            )
        claimed_nodes.update(boundary.nodes)


def _validate_quantized_targets(
    targets: tuple[QuantizedTarget, ...],
) -> set[Algorithm]:
    """Validate materialized quantization records."""
    if not isinstance(targets, tuple) or not all(
        isinstance(item, QuantizedTarget) for item in targets
    ):
        raise GraphStateError(
            "quantized_targets must be a tuple of QuantizedTarget"
        )
    names: set[str] = set()
    algorithms: set[Algorithm] = set()
    for item in targets:
        if not isinstance(item.fqn, str) or not item.fqn:
            raise GraphStateError("Quantized target FQN must be non-empty")
        if item.fqn in names:
            raise GraphStateError("Quantized target FQNs must be unique")
        names.add(item.fqn)
        try:
            target = Target(item.target_type)
            algorithm = Algorithm(item.algorithm)
        except (TypeError, ValueError) as error:
            raise GraphStateError(
                f"Unsupported quantized target contract: {item.target_type}/{item.algorithm}"
            ) from error
        if algorithm is Algorithm.FP16:
            raise GraphStateError("FP16 does not create quantized target metadata")
        algorithms.add(algorithm)
        if isinstance(item.bits, bool) or item.bits not in {4, 8}:
            raise GraphStateError("Quantized target bits must be 4 or 8")
        if algorithm is Algorithm.GPTQ:
            expected_bits = 4 if target is Target.LINEAR else 8
            if item.bits != expected_bits:
                raise GraphStateError(
                    f"GPTQ {target.value} targets must use {expected_bits} bits"
                )
        allowed_granularity = {
            Target.LINEAR: {"per_tensor", "per_channel", "per_token"},
            Target.ATTENTION: {"per_tensor", "per_token"},
            Target.MOE: {"per_tensor"},
        }[target]
        if item.granularity not in allowed_granularity:
            raise GraphStateError(
                f"Unsupported {target.value} granularity: {item.granularity!r}"
            )
        if not isinstance(item.symmetric, bool):
            raise GraphStateError("Quantized target symmetric must be a bool")
        if not isinstance(item.scale, tuple) or not item.scale:
            raise GraphStateError("Quantized target scale must be a non-empty tuple")
        if any(
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale <= 0
            for scale in item.scale
        ):
            raise GraphStateError("Quantized target scales must be finite and positive")
        if not isinstance(item.zero_point, tuple) or len(item.zero_point) != len(item.scale):
            raise GraphStateError("Scale and zero_point lengths must match")
        qmin, qmax = (-(2 ** (item.bits - 1)), 2 ** (item.bits - 1) - 1)
        if any(
            isinstance(zero_point, bool)
            or not isinstance(zero_point, int)
            or zero_point < qmin
            or zero_point > qmax
            for zero_point in item.zero_point
        ):
            raise GraphStateError("Quantized target zero points are out of range")
        if item.symmetric and any(item.zero_point):
            raise GraphStateError("Symmetric quantization requires zero_point=0")
        if item.fallback_reason is not None and (
            algorithm is not Algorithm.GPTQ
            or not isinstance(item.fallback_reason, str)
            or not item.fallback_reason
        ):
            raise GraphStateError("Only GPTQ may carry a non-empty fallback reason")
    return algorithms


def _validate_properties(properties: dict[str, Any]) -> None:
    """Validate metadata extension properties as finite JSON-like data."""
    if not isinstance(properties, dict) or not all(
        isinstance(key, str) and key for key in properties
    ):
        raise GraphStateError("properties must be a string-keyed dictionary")

    def visit(item: Any) -> bool:
        if item is None or isinstance(item, (str, bool, int)):
            return True
        if isinstance(item, float):
            return math.isfinite(item)
        if isinstance(item, (list, tuple)):
            return all(visit(value) for value in item)
        if isinstance(item, dict):
            return all(
                isinstance(key, str) and key and visit(value)
                for key, value in item.items()
            )
        return False

    if not visit(properties):
        raise GraphStateError("properties must contain finite JSON-compatible values")


def _valid_fingerprint(value: str | None) -> bool:
    """Return whether value is a canonical SHA-256 string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits.lower() for character in value)
        and value == value.lower()
    )


def validate_graph(graph: GraphModule) -> GraphMetadata:
    """Validate graph structure and all attached cross-module metadata."""
    if not hasattr(graph, "graph") or not hasattr(graph, "meta"):
        raise TypeError("graph must be a torch.fx.GraphModule")
    graph.graph.lint()  # type: ignore[no-untyped-call]
    value = metadata(graph)
    nodes = tuple(graph.graph.nodes)
    names = [node.name for node in nodes]
    if len(names) != len(set(names)):
        raise GraphStateError("FX node names must be unique")
    placeholders = tuple(node for node in nodes if node.op == "placeholder")
    if len(placeholders) != len(value.input_abi):
        raise GraphStateError("FX placeholders do not match input ABI cardinality")
    output_nodes = tuple(node for node in nodes if node.op == "output")
    if len(output_nodes) != 1:
        raise GraphStateError("FX graph must contain exactly one output node")
    raw_output = output_nodes[0].args[0]
    output_count = len(raw_output) if isinstance(raw_output, (tuple, list)) else 1
    if value.output_abi and output_count != len(value.output_abi):
        raise GraphStateError("FX outputs do not match output ABI cardinality")
    return value


def validate_capability_request(
    value: GraphMetadata,
    *,
    mask_mode: MaskMode | str,
    artifact: Artifact | str,
) -> tuple[Capability, ...]:
    """Validate one requested lowering against metadata and the central matrix."""
    validate_metadata(value)
    try:
        requested_artifact = Artifact(artifact)
    except (TypeError, ValueError) as error:
        raise UnsupportedPatternError(f"Unsupported artifact: {artifact!r}") from error
    phase = Phase.PREFILL if value.stage.is_prefill else Phase.DECODE
    if not value.quantized_targets:
        return (
            require_capability(
                value.model_kind,
                Algorithm.FP16,
                None,
                phase,
                mask_mode,
                artifact,
            ),
        )
    result: list[Capability] = []
    seen: set[tuple[str, str]] = set()
    for target in value.quantized_targets:
        identity = (target.algorithm, target.target_type)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(
            require_capability(
                value.model_kind,
                target.algorithm,
                target.target_type,
                phase,
                mask_mode,
                artifact,
            )
        )
    if requested_artifact is not Artifact.FX and any(
        target.bits == 4 for target in value.quantized_targets
    ):
        raise UnsupportedPatternError("W4 is FX-only and does not support ONNX or ATC")
    return tuple(result)


T = TypeVar("T", bound="GraphModule")


def transactional_update(graph: T, mutator: Callable[[T], None]) -> T:  # noqa: UP047
    """Validate a candidate then atomically replace state, preserving identity."""
    validate_graph(graph)
    candidate = copy.deepcopy(graph)
    mutator(candidate)
    candidate.graph.lint()  # type: ignore[no-untyped-call]
    candidate.recompile()
    validate_graph(candidate)
    object.__setattr__(graph, "__dict__", candidate.__dict__)
    return graph


def infer_model_kind(graph: GraphModule) -> str:
    """Infer Tiny model kind from module metadata."""
    kind = getattr(graph, "_mdc_model_kind", None)
    if kind in {"dense", "moe"}:
        return str(kind)
    module_names = tuple(name.lower() for name, _ in graph.named_modules())
    return "moe" if any("expert" in name or "moe" in name for name in module_names) else "dense"


def require_boundaries(value: GraphMetadata, *kinds: str) -> None:
    """Require every requested fusion boundary kind."""
    present = {boundary.kind for boundary in value.boundaries}
    missing = set(kinds) - present
    if missing:
        joined = ", ".join(sorted(missing))
        raise UnsupportedPatternError(f"Missing fusion boundaries: {joined}")
