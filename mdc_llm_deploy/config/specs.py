"""Immutable tensor and target quantization specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from ..errors import QuantizationConfigError

WeightGranularity = Literal["per_tensor", "per_channel"]
ActivationGranularity = Literal["per_tensor", "per_token"]
ActivationMode = Literal["static", "dynamic"]
QUANTIZATION_BITS = (4, 8)
WEIGHT_GRANULARITIES = ("per_tensor", "per_channel")
ACTIVATION_GRANULARITIES = ("per_tensor", "per_token")
ACTIVATION_MODES = ("static", "dynamic")
ATTENTION_EDGES = ("query", "key", "value", "score")


def _strict_fields(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(repr(name) for name in unknown))
        raise QuantizationConfigError(f"{context} contains unknown fields: {names}")


def _required(value: Mapping[str, Any], fields: set[str], context: str) -> None:
    missing = fields - set(value)
    if missing:
        names = ", ".join(sorted(repr(name) for name in missing))
        raise QuantizationConfigError(f"{context} is missing required fields: {names}")


def _plain_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QuantizationConfigError(f"{context} must be an integer")
    return value


def _plain_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise QuantizationConfigError(f"{context} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class WeightSpec:
    """Weight quantization specification."""

    bits: Literal[4, 8]
    granularity: WeightGranularity
    symmetric: bool = True

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], context: str = "weight") -> WeightSpec:
        """Parse a strict mapping."""
        _strict_fields(value, {"bits", "granularity", "symmetric"}, context)
        _required(value, {"bits", "granularity"}, context)
        bits = _plain_int(value["bits"], f"{context}.bits")
        granularity = value["granularity"]
        symmetric = _plain_bool(value.get("symmetric", True), f"{context}.symmetric")
        if bits not in QUANTIZATION_BITS:
            raise QuantizationConfigError(f"{context}.bits must be 4 or 8")
        if granularity not in WEIGHT_GRANULARITIES:
            raise QuantizationConfigError(
                f"{context}.granularity must be per_tensor or per_channel"
            )
        return cls(
            bits=cast(Literal[4, 8], bits),
            granularity=cast(WeightGranularity, granularity),
            symmetric=symmetric,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "bits": self.bits,
            "granularity": self.granularity,
            "symmetric": self.symmetric,
        }


@dataclass(frozen=True, slots=True)
class ActivationSpec:
    """Activation quantization specification."""

    bits: Literal[4, 8]
    granularity: ActivationGranularity
    mode: ActivationMode
    symmetric: bool = True

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], context: str = "activation"
    ) -> ActivationSpec:
        """Parse a strict mapping."""
        _strict_fields(value, {"bits", "granularity", "mode", "symmetric"}, context)
        _required(value, {"bits", "granularity", "mode"}, context)
        bits = _plain_int(value["bits"], f"{context}.bits")
        granularity = value["granularity"]
        mode = value["mode"]
        symmetric = _plain_bool(value.get("symmetric", True), f"{context}.symmetric")
        if bits not in QUANTIZATION_BITS:
            raise QuantizationConfigError(f"{context}.bits must be 4 or 8")
        if granularity not in ACTIVATION_GRANULARITIES:
            raise QuantizationConfigError(
                f"{context}.granularity must be per_tensor or per_token"
            )
        if mode not in ACTIVATION_MODES:
            raise QuantizationConfigError(f"{context}.mode must be static or dynamic")
        return cls(
            bits=cast(Literal[4, 8], bits),
            granularity=cast(ActivationGranularity, granularity),
            mode=cast(ActivationMode, mode),
            symmetric=symmetric,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "bits": self.bits,
            "granularity": self.granularity,
            "mode": self.mode,
            "symmetric": self.symmetric,
        }


def _optional_spec(
    value: Mapping[str, Any],
    key: str,
    parser: Any,
    context: str,
) -> Any:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, Mapping):
        raise QuantizationConfigError(f"{context}.{key} must be an object or null")
    return parser(item, f"{context}.{key}")


@dataclass(frozen=True, slots=True)
class LinearSpec:
    """Linear target specification."""

    weight: WeightSpec | None = None
    activation: ActivationSpec | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], context: str = "linear") -> LinearSpec:
        """Parse a strict mapping."""
        _strict_fields(value, {"weight", "activation"}, context)
        return cls(
            weight=_optional_spec(value, "weight", WeightSpec.from_dict, context),
            activation=_optional_spec(
                value, "activation", ActivationSpec.from_dict, context
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "weight": None if self.weight is None else self.weight.to_dict(),
            "activation": (
                None if self.activation is None else self.activation.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class AttentionSpec:
    """Attention edge quantization specification."""

    query: ActivationSpec | None = None
    key: ActivationSpec | None = None
    value: ActivationSpec | None = None
    score: ActivationSpec | None = None

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], context: str = "attention"
    ) -> AttentionSpec:
        """Parse a strict mapping."""
        keys = set(ATTENTION_EDGES)
        _strict_fields(value, keys, context)
        return cls(
            **{
                key: _optional_spec(value, key, ActivationSpec.from_dict, context)
                for key in keys
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            key: None if value is None else value.to_dict()
            for key, value in (
                ("query", self.query),
                ("key", self.key),
                ("value", self.value),
                ("score", self.score),
            )
        }


@dataclass(frozen=True, slots=True)
class MoeSpec:
    """MoE expert target specification."""

    weight: WeightSpec | None = None
    activation: ActivationSpec | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], context: str = "moe") -> MoeSpec:
        """Parse a strict mapping."""
        _strict_fields(value, {"weight", "activation"}, context)
        return cls(
            weight=_optional_spec(value, "weight", WeightSpec.from_dict, context),
            activation=_optional_spec(
                value, "activation", ActivationSpec.from_dict, context
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "weight": None if self.weight is None else self.weight.to_dict(),
            "activation": (
                None if self.activation is None else self.activation.to_dict()
            ),
        }
