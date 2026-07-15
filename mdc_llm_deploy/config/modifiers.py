"""Immutable MinMax and GPTQ modifier definitions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import QuantizationConfigError
from .specs import (
    AttentionSpec,
    LinearSpec,
    MoeSpec,
    _plain_bool,
    _plain_int,
    _strict_fields,
)


def _patterns(value: Any, context: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise QuantizationConfigError(f"{context} must be a list of strings or null")
    return tuple(value)


def _target(
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
class MinMaxModifier:
    """MinMax quantization modifier."""

    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    linear: LinearSpec | None = None
    attention: AttentionSpec | None = None
    moe: MoeSpec | None = None
    type: str = "minmax"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MinMaxModifier:
        """Parse a strict MinMax modifier."""
        context = "minmax modifier"
        allowed = {"type", "include", "exclude", "linear", "attention", "moe"}
        _strict_fields(value, allowed, context)
        if value.get("type") != "minmax":
            raise QuantizationConfigError("MinMax modifier type must be 'minmax'")
        return cls(
            include=_patterns(value.get("include"), f"{context}.include"),
            exclude=_patterns(value.get("exclude"), f"{context}.exclude"),
            linear=_target(value, "linear", LinearSpec.from_dict, context),
            attention=_target(value, "attention", AttentionSpec.from_dict, context),
            moe=_target(value, "moe", MoeSpec.from_dict, context),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "type": self.type,
            "include": None if self.include is None else list(self.include),
            "exclude": None if self.exclude is None else list(self.exclude),
            "linear": None if self.linear is None else self.linear.to_dict(),
            "attention": (
                None if self.attention is None else self.attention.to_dict()
            ),
            "moe": None if self.moe is None else self.moe.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GptqModifier:
    """GPTQ quantization modifier."""

    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    linear: LinearSpec | None = None
    moe: MoeSpec | None = None
    percdamp: float = 0.01
    actorder: bool = True
    block_size: int = 128
    type: str = "gptq"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GptqModifier:
        """Parse a strict GPTQ modifier."""
        context = "gptq modifier"
        allowed = {
            "type",
            "include",
            "exclude",
            "linear",
            "attention",
            "moe",
            "percdamp",
            "actorder",
            "block_size",
        }
        _strict_fields(value, allowed, context)
        if value.get("type") != "gptq":
            raise QuantizationConfigError("GPTQ modifier type must be 'gptq'")
        if value.get("attention") is not None:
            raise QuantizationConfigError("GPTQ does not support attention")
        raw_damp = value.get("percdamp", 0.01)
        if isinstance(raw_damp, bool) or not isinstance(raw_damp, (int, float)):
            raise QuantizationConfigError("gptq modifier.percdamp must be a number")
        percdamp = float(raw_damp)
        if not math.isfinite(percdamp) or percdamp < 0:
            raise QuantizationConfigError(
                "gptq modifier.percdamp must be finite and non-negative"
            )
        actorder = _plain_bool(
            value.get("actorder", True), "gptq modifier.actorder"
        )
        block_size = _plain_int(
            value.get("block_size", 128), "gptq modifier.block_size"
        )
        if block_size <= 0:
            raise QuantizationConfigError(
                "gptq modifier.block_size must be positive"
            )
        linear = _target(value, "linear", LinearSpec.from_dict, context)
        moe = _target(value, "moe", MoeSpec.from_dict, context)
        if linear is None and moe is None:
            raise QuantizationConfigError("GPTQ requires linear or moe")
        for name, target, granularity in (
            ("linear", linear, "per_channel"),
            ("moe", moe, "per_tensor"),
        ):
            if target is not None:
                if target.weight is None:
                    raise QuantizationConfigError(f"GPTQ {name} requires weight")
                if target.weight.granularity != granularity:
                    raise QuantizationConfigError(
                        f"GPTQ {name} weight granularity must be {granularity}"
                    )
                if not target.weight.symmetric:
                    raise QuantizationConfigError(
                        f"GPTQ {name} weight must be symmetric"
                    )
        return cls(
            include=_patterns(value.get("include"), f"{context}.include"),
            exclude=_patterns(value.get("exclude"), f"{context}.exclude"),
            linear=linear,
            moe=moe,
            percdamp=percdamp,
            actorder=actorder,
            block_size=block_size,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "type": self.type,
            "include": None if self.include is None else list(self.include),
            "exclude": None if self.exclude is None else list(self.exclude),
            "linear": None if self.linear is None else self.linear.to_dict(),
            "moe": None if self.moe is None else self.moe.to_dict(),
            "percdamp": self.percdamp,
            "actorder": self.actorder,
            "block_size": self.block_size,
        }


Modifier = MinMaxModifier | GptqModifier
