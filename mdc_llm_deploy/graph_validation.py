"""Low-level validation for graph metadata value objects."""

from __future__ import annotations

import math
import string

from .capabilities import Algorithm, Target, gptq_bits_for
from .errors import GraphStateError
from .graph_types import (
    ABI_DTYPES,
    BOUNDARY_KINDS,
    FusionBoundary,
    QuantizedTarget,
    TensorAbi,
)


def validate_abi(
    label: str,
    entries: tuple[TensorAbi, ...],
    *,
    required: bool,
) -> None:
    """Validate one ordered tensor ABI."""
    if not isinstance(entries, tuple) or not all(
        isinstance(item, TensorAbi) for item in entries
    ):
        raise GraphStateError(
            f"{label} ABI must be a tuple of TensorAbi"
        )
    if required and not entries:
        raise GraphStateError(f"{label} ABI must not be empty")
    names: list[str] = []
    for item in entries:
        if not isinstance(item.name, str) or not item.name:
            raise GraphStateError(
                f"{label} ABI names must be non-empty strings"
            )
        if item.dtype not in ABI_DTYPES:
            raise GraphStateError(
                f"Unsupported ABI dtype: {item.dtype!r}"
            )
        if not isinstance(item.shape, tuple) or not item.shape:
            raise GraphStateError(
                f"{label} ABI shapes must be non-empty tuples"
            )
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in item.shape
        ):
            raise GraphStateError(
                f"{label} ABI shapes must be static and positive"
            )
        names.append(item.name)
    if len(names) != len(set(names)):
        raise GraphStateError(
            f"{label} ABI names must be unique"
        )


def validate_boundaries(
    boundaries: tuple[FusionBoundary, ...],
) -> None:
    """Validate fusion boundary names and ownership."""
    if not isinstance(boundaries, tuple) or not all(
        isinstance(item, FusionBoundary) for item in boundaries
    ):
        raise GraphStateError(
            "boundaries must be a tuple of FusionBoundary"
        )
    identities: set[tuple[str, str]] = set()
    claimed_nodes: set[str] = set()
    for boundary in boundaries:
        if (
            not isinstance(boundary.kind, str)
            or boundary.kind not in BOUNDARY_KINDS
        ):
            raise GraphStateError(
                "Unsupported fusion boundary kind: "
                f"{boundary.kind!r}"
            )
        if not isinstance(boundary.fqn, str) or not boundary.fqn:
            raise GraphStateError(
                "Fusion boundary FQN must be non-empty"
            )
        identity = (boundary.kind, boundary.fqn)
        if identity in identities:
            raise GraphStateError(
                "Fusion boundary kind/FQN pairs must be unique"
            )
        identities.add(identity)
        if not isinstance(boundary.nodes, tuple) or not all(
            isinstance(node, str) and node
            for node in boundary.nodes
        ):
            raise GraphStateError(
                "Fusion boundary nodes must be non-empty strings"
            )
        if len(boundary.nodes) != len(set(boundary.nodes)):
            raise GraphStateError(
                "Fusion boundary nodes must be unique"
            )
        overlap = claimed_nodes.intersection(boundary.nodes)
        if overlap:
            raise GraphStateError(
                "Fusion boundary nodes have multiple owners: "
                f"{sorted(overlap)}"
            )
        claimed_nodes.update(boundary.nodes)


def validate_quantized_targets(
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
            raise GraphStateError(
                "Quantized target FQN must be non-empty"
            )
        if item.fqn in names:
            raise GraphStateError(
                "Quantized target FQNs must be unique"
            )
        names.add(item.fqn)
        try:
            target = Target(item.target_type)
            algorithm = Algorithm(item.algorithm)
        except (TypeError, ValueError) as error:
            raise GraphStateError(
                "Unsupported quantized target contract: "
                f"{item.target_type}/{item.algorithm}"
            ) from error
        if algorithm is Algorithm.FP16:
            raise GraphStateError(
                "FP16 does not create quantized target metadata"
            )
        algorithms.add(algorithm)
        if isinstance(item.bits, bool) or item.bits not in {4, 8}:
            raise GraphStateError(
                "Quantized target bits must be 4 or 8"
            )
        if algorithm is Algorithm.GPTQ:
            expected_bits = gptq_bits_for(target)
            if item.bits != expected_bits:
                raise GraphStateError(
                    f"GPTQ {target.value} targets must use "
                    f"{expected_bits} bits"
                )
        allowed_granularity = {
            Target.LINEAR: {
                "per_tensor",
                "per_channel",
                "per_token",
            },
            Target.ATTENTION: {"per_tensor", "per_token"},
            Target.MOE: {"per_tensor"},
        }[target]
        if item.granularity not in allowed_granularity:
            raise GraphStateError(
                f"Unsupported {target.value} granularity: "
                f"{item.granularity!r}"
            )
        if not isinstance(item.symmetric, bool):
            raise GraphStateError(
                "Quantized target symmetric must be a bool"
            )
        if not isinstance(item.scale, tuple) or not item.scale:
            raise GraphStateError(
                "Quantized target scale must be a non-empty tuple"
            )
        if any(
            type(scale) is not float
            or not math.isfinite(scale)
            or scale <= 0
            for scale in item.scale
        ):
            raise GraphStateError(
                "Quantized target scales must be finite and positive"
            )
        if (
            not isinstance(item.zero_point, tuple)
            or len(item.zero_point) != len(item.scale)
        ):
            raise GraphStateError(
                "Scale and zero_point lengths must match"
            )
        qmin = -(2 ** (item.bits - 1))
        qmax = 2 ** (item.bits - 1) - 1
        if any(
            isinstance(zero_point, bool)
            or not isinstance(zero_point, int)
            or zero_point < qmin
            or zero_point > qmax
            for zero_point in item.zero_point
        ):
            raise GraphStateError(
                "Quantized target zero points are out of range"
            )
        if item.symmetric and any(item.zero_point):
            raise GraphStateError(
                "Symmetric quantization requires zero_point=0"
            )
        if item.fallback_reason is not None and (
            algorithm is not Algorithm.GPTQ
            or not isinstance(item.fallback_reason, str)
            or not item.fallback_reason
        ):
            raise GraphStateError(
                "Only GPTQ may carry a non-empty fallback reason"
            )
    return algorithms


def valid_fingerprint(value: str | None) -> bool:
    """Return whether value is a canonical SHA-256 string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in string.hexdigits.lower()
            for character in value
        )
        and value == value.lower()
    )
