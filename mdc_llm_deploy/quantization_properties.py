"""Typed readers for quantization data stored in graph properties."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ActivationQuantizationParameters:
    """Validated activation quantization parameters for one target."""

    bits: int
    granularity: str
    mode: str | None
    symmetric: bool
    scale: tuple[float, ...]
    zero_point: tuple[int, ...]

    @classmethod
    def for_target(
        cls,
        properties: Mapping[str, Any],
        fqn: str,
    ) -> ActivationQuantizationParameters | None:
        """Read parameters for a target, returning None when absent."""
        all_parameters = properties.get("activation_qparams")
        if not isinstance(all_parameters, Mapping):
            return None
        raw = all_parameters.get(fqn)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Activation qparams for {fqn!r} must be a mapping"
            )
        return cls._from_mapping(raw, fqn)

    @classmethod
    def _from_mapping(
        cls,
        raw: Mapping[str, Any],
        fqn: str,
    ) -> ActivationQuantizationParameters:
        bits = raw.get("bits")
        granularity = raw.get("granularity")
        mode = raw.get("mode")
        symmetric = raw.get("symmetric")
        scale = raw.get("scale")
        zero_point = raw.get("zero_point")
        if type(bits) is not int or bits <= 0:
            raise ValueError(
                f"Activation qparams for {fqn!r} require positive integer bits"
            )
        if not isinstance(granularity, str) or not granularity:
            raise ValueError(
                f"Activation qparams for {fqn!r} require granularity"
            )
        if mode is not None and (
            not isinstance(mode, str) or not mode
        ):
            raise ValueError(
                f"Activation qparams for {fqn!r} have invalid mode"
            )
        if type(symmetric) is not bool:
            raise ValueError(
                f"Activation qparams for {fqn!r} require boolean symmetric"
            )
        if not isinstance(scale, tuple) or not scale:
            raise ValueError(
                f"Activation qparams for {fqn!r} require scale values"
            )
        if not isinstance(zero_point, tuple) or not zero_point:
            raise ValueError(
                f"Activation qparams for {fqn!r} require zero-point values"
            )
        if len(scale) != len(zero_point):
            raise ValueError(
                f"Activation qparams for {fqn!r} have mismatched parameter lengths"
            )
        return cls(
            bits=bits,
            granularity=granularity,
            mode=mode,
            symmetric=symmetric,
            scale=tuple(float(item) for item in scale),
            zero_point=tuple(int(item) for item in zero_point),
        )


__all__ = ["ActivationQuantizationParameters"]
