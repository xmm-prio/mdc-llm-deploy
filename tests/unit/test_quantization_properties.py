"""Tests for typed graph quantization property readers."""

from __future__ import annotations

import pytest

from mdc_llm_deploy.quantization_properties import (
    ActivationQuantizationParameters,
)


def _properties() -> dict[str, object]:
    return {
        "activation_qparams": {
            "model.proj": {
                "bits": 8,
                "granularity": "per_tensor",
                "mode": "static",
                "symmetric": True,
                "scale": (0.25,),
                "zero_point": (0,),
            }
        }
    }


def test_activation_qparams_parse_target_contract() -> None:
    result = ActivationQuantizationParameters.for_target(
        _properties(),
        "model.proj",
    )

    assert result == ActivationQuantizationParameters(
        bits=8,
        granularity="per_tensor",
        mode="static",
        symmetric=True,
        scale=(0.25,),
        zero_point=(0,),
    )


def test_activation_qparams_return_none_when_target_absent() -> None:
    assert (
        ActivationQuantizationParameters.for_target(
            _properties(),
            "model.missing",
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bits", True),
        ("granularity", ""),
        ("mode", 1),
        ("symmetric", 1),
        ("scale", ()),
        ("zero_point", ()),
        ("zero_point", (0, 1)),
    ],
)
def test_activation_qparams_reject_malformed_contract(
    field: str,
    value: object,
) -> None:
    properties = _properties()
    parameters = properties["activation_qparams"]
    assert isinstance(parameters, dict)
    target = parameters["model.proj"]
    assert isinstance(target, dict)
    target[field] = value

    with pytest.raises(ValueError, match=r"model\.proj"):
        ActivationQuantizationParameters.for_target(
            properties,
            "model.proj",
        )
