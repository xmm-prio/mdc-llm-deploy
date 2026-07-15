"""Strict loading, serialization, schema, and fingerprinting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import QuantizationConfigError
from .modifiers import GptqModifier, MinMaxModifier, Modifier
from .specs import _strict_fields


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QuantizationConfigError(f"Config JSON contains duplicate field: {key!r}")
        result[key] = value
    return result


def _selector(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise QuantizationConfigError(f"{context} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Ordered quantization configuration with stable canonical serialization."""

    modifiers: tuple[Modifier, ...]
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QuantizationConfig:
        """Parse a strict JSON-compatible mapping."""
        if not isinstance(value, Mapping):
            raise QuantizationConfigError("config must be an object")
        _strict_fields(value, {"modifiers", "include", "exclude"}, "config")
        raw_modifiers = value.get("modifiers")
        if not isinstance(raw_modifiers, list):
            raise QuantizationConfigError("config.modifiers must be a list")
        modifiers: list[Modifier] = []
        for index, raw_modifier in enumerate(raw_modifiers):
            if not isinstance(raw_modifier, Mapping):
                raise QuantizationConfigError(
                    f"config.modifiers[{index}] must be an object"
                )
            modifier_type = raw_modifier.get("type")
            if modifier_type == "minmax":
                modifiers.append(MinMaxModifier.from_dict(raw_modifier))
            elif modifier_type == "gptq":
                modifiers.append(GptqModifier.from_dict(raw_modifier))
            else:
                raise QuantizationConfigError(
                    f"config.modifiers[{index}].type must be minmax or gptq"
                )
        return cls(
            modifiers=tuple(modifiers),
            include=_selector(value.get("include", []), "config.include"),
            exclude=_selector(value.get("exclude", []), "config.exclude"),
        )

    @classmethod
    def load(
        cls,
        value: QuantizationConfig | Mapping[str, Any] | str | Path,
    ) -> QuantizationConfig:
        """Load an existing config, mapping, or UTF-8 JSON file."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        if not isinstance(value, (str, Path)):
            raise QuantizationConfigError(
                "config must be QuantizationConfig, mapping, str, or Path"
            )
        path = Path(value)
        try:
            raw = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_json_object,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise QuantizationConfigError(f"Cannot load config: {error}") from error
        if not isinstance(raw, Mapping):
            raise QuantizationConfigError("Config JSON root must be an object")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Return parsed values including all defaults."""
        return {
            "include": list(self.include),
            "exclude": list(self.exclude),
            "modifiers": [modifier.to_dict() for modifier in self.modifiers],
        }

    def to_json_string(self) -> str:
        """Serialize readable JSON with one trailing newline."""
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    @property
    def fingerprint(self) -> str:
        """Return canonical SHA-256 fingerprint."""
        canonical = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(canonical).hexdigest()


def generate_schema() -> dict[str, Any]:
    """Generate package JSON Schema from configuration contracts."""
    def nullable(name: str) -> dict[str, Any]:
        return {"anyOf": [{"$ref": f"#/$defs/{name}"}, {"type": "null"}]}

    weight = {
        "type": "object",
        "additionalProperties": False,
        "required": ["bits", "granularity"],
        "properties": {
            "bits": {"enum": [4, 8]},
            "granularity": {"enum": ["per_tensor", "per_channel"]},
            "symmetric": {"type": "boolean", "default": True},
        },
    }
    activation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["bits", "granularity", "mode"],
        "properties": {
            "bits": {"enum": [4, 8]},
            "granularity": {"enum": ["per_tensor", "per_token"]},
            "mode": {"enum": ["static", "dynamic"]},
            "symmetric": {"type": "boolean", "default": True},
        },
    }
    linear = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "weight": nullable("weight"),
            "activation": nullable("activation"),
        },
    }
    attention = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            edge: nullable("activation")
            for edge in ("query", "key", "value", "score")
        },
    }
    moe = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "weight": nullable("weight"),
            "activation": nullable("activation"),
        },
    }
    gptq_linear_weight = {
        "type": "object",
        "additionalProperties": False,
        "required": ["bits", "granularity"],
        "properties": {
            "bits": {"enum": [4, 8]},
            "granularity": {"const": "per_channel"},
            "symmetric": {"const": True, "default": True},
        },
    }
    gptq_moe_weight = {
        "type": "object",
        "additionalProperties": False,
        "required": ["bits", "granularity"],
        "properties": {
            "bits": {"enum": [4, 8]},
            "granularity": {"const": "per_tensor"},
            "symmetric": {"const": True, "default": True},
        },
    }
    gptq_linear = {
        "type": "object",
        "additionalProperties": False,
        "required": ["weight"],
        "properties": {
            "weight": {"$ref": "#/$defs/gptqLinearWeight"},
            "activation": nullable("activation"),
        },
    }
    gptq_moe = {
        "type": "object",
        "additionalProperties": False,
        "required": ["weight"],
        "properties": {
            "weight": {"$ref": "#/$defs/gptqMoeWeight"},
            "activation": nullable("activation"),
        },
    }
    selector_properties: dict[str, Any] = {
        "include": {"type": ["array", "null"], "items": {"type": "string"}},
        "exclude": {"type": ["array", "null"], "items": {"type": "string"}},
    }
    minmax = {
        "type": "object",
        "additionalProperties": False,
        "required": ["type"],
        "properties": {
            "type": {"const": "minmax"},
            **selector_properties,
            "linear": nullable("linear"),
            "attention": nullable("attention"),
            "moe": nullable("moe"),
        },
    }
    gptq = {
        "type": "object",
        "additionalProperties": False,
        "required": ["type"],
        "properties": {
            "type": {"const": "gptq"},
            **selector_properties,
            "linear": nullable("gptqLinear"),
            "attention": {"type": "null"},
            "moe": nullable("gptqMoe"),
            "percdamp": {"type": "number", "minimum": 0, "default": 0.01},
            "actorder": {"type": "boolean", "default": True},
            "block_size": {"type": "integer", "minimum": 1, "default": 128},
        },
        "anyOf": [
            {
                "required": ["linear"],
                "properties": {"linear": {"$ref": "#/$defs/gptqLinear"}},
            },
            {
                "required": ["moe"],
                "properties": {"moe": {"$ref": "#/$defs/gptqMoe"}},
            },
        ],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://mdc-llm-deploy/schema/quantization-config-0.1.0.json",
        "title": "MDC LLM Deploy quantization configuration",
        "type": "object",
        "additionalProperties": False,
        "required": ["modifiers"],
        "properties": {
            "include": {"type": "array", "items": {"type": "string"}, "default": []},
            "exclude": {"type": "array", "items": {"type": "string"}, "default": []},
            "modifiers": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"$ref": "#/$defs/minmax"},
                        {"$ref": "#/$defs/gptq"},
                    ]
                },
            },
        },
        "$defs": {
            "weight": weight,
            "activation": activation,
            "linear": linear,
            "attention": attention,
            "moe": moe,
            "gptqLinearWeight": gptq_linear_weight,
            "gptqMoeWeight": gptq_moe_weight,
            "gptqLinear": gptq_linear,
            "gptqMoe": gptq_moe,
            "minmax": minmax,
            "gptq": gptq,
        },
    }


def schema_json() -> str:
    """Return deterministic packaged schema text."""
    return (
        json.dumps(
            generate_schema(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
