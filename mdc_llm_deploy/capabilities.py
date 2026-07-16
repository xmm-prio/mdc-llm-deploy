"""Central capability matrix for graph, export, and validation stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .errors import UnsupportedPatternError


class ModelKind(StrEnum):
    """Supported model architecture families."""

    DENSE = "dense"
    MOE = "moe"


class Algorithm(StrEnum):
    """Supported numerical algorithms."""

    FP16 = "fp16"
    MINMAX = "minmax"
    GPTQ = "gptq"


class Target(StrEnum):
    """Supported quantization target families."""

    LINEAR = "linear"
    ATTENTION = "attention"
    MOE = "moe"


class Phase(StrEnum):
    """Supported inference phases."""

    PREFILL = "prefill"
    DECODE = "decode"


class MaskMode(StrEnum):
    """Supported attention mask modes."""

    MASKED = "masked"
    MASKLESS = "maskless"


class Artifact(StrEnum):
    """Validation artifacts ordered by lowering depth."""

    FX = "fx"
    ONNX = "onnx"
    ATC = "atc"


@dataclass(frozen=True, slots=True)
class Capability:
    """One supported model conversion combination."""

    model: ModelKind
    algorithm: Algorithm
    target: Target | None
    phase: Phase
    mask_mode: MaskMode
    artifacts: frozenset[Artifact]

    def supports(self, artifact: Artifact | str) -> bool:
        """Return whether this combination reaches the requested artifact."""
        return Artifact(artifact) in self.artifacts


def gptq_bits_for(target: Target | str) -> int:
    """Return the fixed GPTQ bit width for one target family."""
    return {
        Target.LINEAR: 4,
        Target.MOE: 8,
    }[Target(target)]


def gptq_granularity_for(target: Target | str) -> str:
    """Return the fixed GPTQ granularity for one target family."""
    return {
        Target.LINEAR: "per_channel",
        Target.MOE: "per_tensor",
    }[Target(target)]


_FULL_ARTIFACTS = frozenset({Artifact.FX, Artifact.ONNX, Artifact.ATC})
_FX_ONLY = frozenset({Artifact.FX})


def _build_matrix() -> tuple[Capability, ...]:
    result: list[Capability] = []
    for model in ModelKind:
        for phase in Phase:
            for mask_mode in MaskMode:
                result.append(
                    Capability(
                        model=model,
                        algorithm=Algorithm.FP16,
                        target=None,
                        phase=phase,
                        mask_mode=mask_mode,
                        artifacts=_FULL_ARTIFACTS,
                    )
                )
                minmax_targets = (
                    (Target.LINEAR, Target.ATTENTION)
                    if model is ModelKind.DENSE
                    else (Target.LINEAR, Target.ATTENTION, Target.MOE)
                )
                result.extend(
                    Capability(
                        model=model,
                        algorithm=Algorithm.MINMAX,
                        target=target,
                        phase=phase,
                        mask_mode=mask_mode,
                        artifacts=_FULL_ARTIFACTS,
                    )
                    for target in minmax_targets
                )

    # GPTQ is an FX-only numerical path. Mask mode remains explicit so every
    # requested combination is deterministic, but it never grants ONNX/ATC.
    for model, targets in (
        (ModelKind.DENSE, (Target.LINEAR,)),
        (ModelKind.MOE, (Target.LINEAR, Target.MOE)),
    ):
        for phase in Phase:
            for mask_mode in MaskMode:
                result.extend(
                    Capability(
                        model=model,
                        algorithm=Algorithm.GPTQ,
                        target=target,
                        phase=phase,
                        mask_mode=mask_mode,
                        artifacts=_FX_ONLY,
                    )
                    for target in targets
                )
    return tuple(result)


CAPABILITY_MATRIX = _build_matrix()
CapabilityKey = tuple[
    ModelKind,
    Algorithm,
    Target | None,
    Phase,
    MaskMode,
]


def _capability_key(item: Capability) -> CapabilityKey:
    return (
        item.model,
        item.algorithm,
        item.target,
        item.phase,
        item.mask_mode,
    )


_CAPABILITY_INDEX = {
    _capability_key(item): item for item in CAPABILITY_MATRIX
}
if len(_CAPABILITY_INDEX) != len(CAPABILITY_MATRIX):
    raise RuntimeError(
        "Capability matrix contains duplicate combinations"
    )


def capability_for(
    model: ModelKind | str,
    algorithm: Algorithm | str,
    target: Target | str | None,
    phase: Phase | str,
    mask_mode: MaskMode | str,
) -> Capability | None:
    """Return the exact matrix entry, or None when unsupported."""
    try:
        requested: CapabilityKey = (
            ModelKind(model),
            Algorithm(algorithm),
            None if target is None else Target(target),
            Phase(phase),
            MaskMode(mask_mode),
        )
    except (TypeError, ValueError):
        return None
    return _CAPABILITY_INDEX.get(requested)


def require_capability(
    model: ModelKind | str,
    algorithm: Algorithm | str,
    target: Target | str | None,
    phase: Phase | str,
    mask_mode: MaskMode | str,
    artifact: Artifact | str,
) -> Capability:
    """Return a supported capability or raise a stable contract error."""
    try:
        requested_artifact = Artifact(artifact)
    except (TypeError, ValueError) as error:
        raise UnsupportedPatternError(f"Unsupported artifact: {artifact!r}") from error
    item = capability_for(model, algorithm, target, phase, mask_mode)
    if item is None:
        raise UnsupportedPatternError(
            "Unsupported capability: "
            f"model={model}, algorithm={algorithm}, target={target}, "
            f"phase={phase}, mask_mode={mask_mode}"
        )
    if not item.supports(requested_artifact):
        if item.algorithm is Algorithm.GPTQ:
            raise UnsupportedPatternError("GPTQ is FX-only and does not support ONNX or ATC")
        raise UnsupportedPatternError(
            f"Capability does not support artifact={requested_artifact.value}"
        )
    return item
