"""Torch-independent, versioned graph contract."""

from __future__ import annotations

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
from .graph_types import (
    GRAPH_METADATA_KEY as GRAPH_METADATA_KEY,
)
from .graph_types import (
    GRAPH_SCHEMA_VERSION as GRAPH_SCHEMA_VERSION,
)
from .graph_types import (
    FusionBoundary as FusionBoundary,
)
from .graph_types import (
    GraphMetadata as GraphMetadata,
)
from .graph_types import (
    GraphStage as GraphStage,
)
from .graph_types import (
    QuantizedTarget as QuantizedTarget,
)
from .graph_types import (
    TensorAbi as TensorAbi,
)
from .graph_validation import (
    valid_fingerprint,
    validate_abi,
    validate_boundaries,
    validate_quantized_targets,
)
from .immutable_json import (
    FrozenJsonMapping as FrozenJsonMapping,
)
from .immutable_json import (
    validate_json_mapping,
)
from .model_properties import NormalizationProperties


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
    validate_abi("Input", value.input_abi, required=True)
    validate_abi("Output", value.output_abi, required=False)
    validate_boundaries(value.boundaries)
    algorithms = validate_quantized_targets(value.quantized_targets)
    validate_json_mapping(value.properties)

    if value.stage.is_prefill:
        if value.absolute_position is not None:
            raise GraphStateError("Prefill graph must not carry absolute_position")
    else:
        if type(value.absolute_position) is not int:
            raise GraphStateError("Decode graph absolute_position must be an integer")
        if value.absolute_position != value.sequence_length - 1:
            raise GraphStateError(
                "Decode graph absolute_position must equal sequence_length - 1"
            )

    if value.stage.is_quantized:
        if not value.quantized_targets:
            raise GraphStateError("Quantized graph must carry target metadata")
        if not valid_fingerprint(value.config_fingerprint):
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
    try:
        normalization = NormalizationProperties.from_properties(
            value.properties
        )
    except ValueError as error:
        raise GraphStateError(str(error)) from error
    if (
        requested_artifact is not Artifact.FX
        and normalization.rms_norm_epsilon is not None
        and normalization.rms_norm_epsilon != 1e-6
    ):
        raise UnsupportedPatternError(
            "MDC ONNX and ATC require RmsNorm epsilon=1e-6"
        )
    phase = Phase.PREFILL if value.stage.is_prefill else Phase.DECODE
    result: list[Capability] = []
    if not value.quantized_targets:
        result.append(
            require_capability(
                value.model_kind,
                Algorithm.FP16,
                None,
                phase,
                mask_mode,
                artifact,
            )
        )
    else:
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
    if requested_artifact is not Artifact.FX and any(
        target.target_type == Target.ATTENTION.value
        and target.fqn.rsplit(".", 1)[-1] in {"query", "score"}
        and not target.symmetric
        for target in value.quantized_targets
    ):
        raise UnsupportedPatternError(
            "Asymmetric attention query/score is FX-only"
        )
    if requested_artifact is not Artifact.FX:
        require_boundaries(value, "attention")
    return tuple(result)


def require_boundaries(value: GraphMetadata, *kinds: str) -> None:
    """Require every requested fusion boundary kind."""
    present = {boundary.kind for boundary in value.boundaries}
    missing = set(kinds) - present
    if missing:
        joined = ", ".join(sorted(missing))
        raise UnsupportedPatternError(f"Missing fusion boundaries: {joined}")
