"""JSON Schema generation for quantization configuration."""

from __future__ import annotations

import json
from typing import Any

from ..capabilities import (
    Target,
    gptq_bits_for,
    gptq_granularity_for,
)
from .modifiers import (
    GPTQ_ACTORDER_DEFAULT,
    GPTQ_BLOCK_SIZE_DEFAULT,
    GPTQ_BLOCK_SIZE_MINIMUM,
    GPTQ_PERCDAMP_DEFAULT,
    GPTQ_PERCDAMP_MINIMUM,
)
from .specs import (
    ACTIVATION_GRANULARITIES,
    ACTIVATION_MODES,
    ATTENTION_EDGES,
    QUANTIZATION_BITS,
    WEIGHT_GRANULARITIES,
)


def _nullable(name: str) -> dict[str, Any]:
    return {"anyOf": [{"$ref": f"#/$defs/{name}"}, {"type": "null"}]}


def _required_ref(
    field: str,
    definition: str,
) -> dict[str, Any]:
    return {
        "required": [field],
        "properties": {
            field: {"$ref": f"#/$defs/{definition}"}
        },
    }


def _core_definitions() -> dict[str, dict[str, Any]]:
    weight = {
        "type": "object",
        "additionalProperties": False,
        "required": ["bits", "granularity"],
        "properties": {
            "bits": {"enum": list(QUANTIZATION_BITS)},
            "granularity": {
                "enum": list(WEIGHT_GRANULARITIES)
            },
            "symmetric": {"type": "boolean", "default": True},
        },
    }
    activation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["bits", "granularity", "mode"],
        "properties": {
            "bits": {"enum": list(QUANTIZATION_BITS)},
            "granularity": {
                "enum": list(ACTIVATION_GRANULARITIES)
            },
            "mode": {"enum": list(ACTIVATION_MODES)},
            "symmetric": {"type": "boolean", "default": True},
        },
    }
    return {
        "weight": weight,
        "activation": activation,
        "linear": {
            "type": "object",
            "additionalProperties": False,
            "anyOf": [
                _required_ref("weight", "weight"),
                _required_ref("activation", "activation"),
            ],
            "properties": {
                "weight": _nullable("weight"),
                "activation": _nullable("activation"),
            },
        },
        "attention": {
            "type": "object",
            "additionalProperties": False,
            "anyOf": [
                _required_ref(edge, "activation")
                for edge in ATTENTION_EDGES
            ],
            "properties": {
                edge: _nullable("activation")
                for edge in ATTENTION_EDGES
            },
        },
        "moe": {
            "type": "object",
            "additionalProperties": False,
            "anyOf": [
                _required_ref("weight", "weight"),
                _required_ref("activation", "activation"),
            ],
            "properties": {
                "weight": _nullable("weight"),
                "activation": _nullable("activation"),
            },
        },
    }


def _gptq_definitions() -> dict[str, dict[str, Any]]:
    linear_weight = {
        "type": "object",
        "additionalProperties": False,
        "required": ["bits", "granularity"],
        "properties": {
            "bits": {
                "const": gptq_bits_for(Target.LINEAR)
            },
            "granularity": {
                "const": gptq_granularity_for(Target.LINEAR)
            },
            "symmetric": {"const": True, "default": True},
        },
    }
    moe_weight = {
        "type": "object",
        "additionalProperties": False,
        "required": ["bits", "granularity"],
        "properties": {
            "bits": {"const": gptq_bits_for(Target.MOE)},
            "granularity": {
                "const": gptq_granularity_for(Target.MOE)
            },
            "symmetric": {"const": True, "default": True},
        },
    }
    return {
        "gptqLinearWeight": linear_weight,
        "gptqMoeWeight": moe_weight,
        "gptqLinear": {
            "type": "object",
            "additionalProperties": False,
            "required": ["weight"],
            "properties": {
                "weight": {"$ref": "#/$defs/gptqLinearWeight"},
                "activation": _nullable("activation"),
            },
        },
        "gptqMoe": {
            "type": "object",
            "additionalProperties": False,
            "required": ["weight"],
            "properties": {
                "weight": {"$ref": "#/$defs/gptqMoeWeight"},
                "activation": _nullable("activation"),
            },
        },
    }


def _modifier_definitions() -> dict[str, dict[str, Any]]:
    selector_properties: dict[str, Any] = {
        "include": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
        "exclude": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
    }
    return {
        "minmax": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type"],
            "anyOf": [
                _required_ref("linear", "linear"),
                _required_ref("attention", "attention"),
                _required_ref("moe", "moe"),
            ],
            "properties": {
                "type": {"const": "minmax"},
                **selector_properties,
                "linear": _nullable("linear"),
                "attention": _nullable("attention"),
                "moe": _nullable("moe"),
            },
        },
        "gptq": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type"],
            "properties": {
                "type": {"const": "gptq"},
                **selector_properties,
                "linear": _nullable("gptqLinear"),
                "attention": {"type": "null"},
                "moe": _nullable("gptqMoe"),
                "percdamp": {
                    "type": "number",
                    "minimum": GPTQ_PERCDAMP_MINIMUM,
                    "default": GPTQ_PERCDAMP_DEFAULT,
                },
                "actorder": {
                    "type": "boolean",
                    "default": GPTQ_ACTORDER_DEFAULT,
                },
                "block_size": {
                    "type": "integer",
                    "minimum": GPTQ_BLOCK_SIZE_MINIMUM,
                    "default": GPTQ_BLOCK_SIZE_DEFAULT,
                },
            },
            "anyOf": [
                {
                    "required": ["linear"],
                    "properties": {
                        "linear": {"$ref": "#/$defs/gptqLinear"}
                    },
                },
                {
                    "required": ["moe"],
                    "properties": {
                        "moe": {"$ref": "#/$defs/gptqMoe"}
                    },
                },
            ],
        },
    }


def generate_schema() -> dict[str, Any]:
    """Generate package JSON Schema from configuration contracts."""
    definitions = {
        **_core_definitions(),
        **_gptq_definitions(),
        **_modifier_definitions(),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (
            "https://mdc-llm-deploy/schema/"
            "quantization-config-0.1.0.json"
        ),
        "title": "MDC LLM Deploy quantization configuration",
        "type": "object",
        "additionalProperties": False,
        "required": ["modifiers"],
        "properties": {
            "include": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
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
        "$defs": definitions,
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
