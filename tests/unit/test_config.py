from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mdc_llm_deploy.config import QuantizationConfig, schema_json
from mdc_llm_deploy.errors import MdcDeployError, QuantizationConfigError

ROOT = Path(__file__).parents[2]
CONFIG_PATHS = sorted((ROOT / "configs").glob("*.json"))
EXPECTED_FINGERPRINTS = {
    "gptq-linear-w4a8.json": "5269c98570f4c92e88f27c54495bb3b4b4031b023e7ed0a19cfbb8c9099221f6",
    "gptq-moe-w8a8.json": "1ebf0f890afbe93761412b50464036aebd3633f22e7764694c2eaba483743ab0",
    "minmax-attention-a8.json": "0ebf1ef6fd92ded5630904dbef0a66099103e3759813bf6c17098dd8b77dd089",
    "minmax-linear-w8a8.json": "02475a243821b584878f48b48020e9e334a3b47fc76f1b2c0929645accd78f49",
    "minmax-moe-w8a8.json": "6edcd65e9781e48d0f9615c8c83a63ff31116b7e44a2d6e76fd0a0b5708ec3d2",
}


def test_all_release_configs_load_and_round_trip() -> None:
    assert [path.name for path in CONFIG_PATHS] == [
        "gptq-linear-w4a8.json",
        "gptq-moe-w8a8.json",
        "minmax-attention-a8.json",
        "minmax-linear-w8a8.json",
        "minmax-moe-w8a8.json",
    ]
    for path in CONFIG_PATHS:
        config = QuantizationConfig.load(path)
        reparsed = QuantizationConfig.from_dict(json.loads(config.to_json_string()))
        assert reparsed == config
        assert reparsed.fingerprint == config.fingerprint
        assert len(config.fingerprint) == 64
        assert config.fingerprint == config.fingerprint.lower()
        assert config.fingerprint == EXPECTED_FINGERPRINTS[path.name]


def test_fingerprint_ignores_order_whitespace_and_implicit_defaults(tmp_path: Path) -> None:
    implicit = {
        "modifiers": [
            {
                "linear": {
                    "activation": {
                        "mode": "static",
                        "granularity": "per_tensor",
                        "bits": 8,
                    },
                    "weight": {"granularity": "per_channel", "bits": 8},
                },
                "type": "minmax",
            }
        ]
    }
    explicit = {
        "exclude": [],
        "include": [],
        "modifiers": [
            {
                "attention": None,
                "exclude": None,
                "include": None,
                "linear": {
                    "activation": {
                        "bits": 8,
                        "granularity": "per_tensor",
                        "mode": "static",
                        "symmetric": True,
                    },
                    "weight": {
                        "bits": 8,
                        "granularity": "per_channel",
                        "symmetric": True,
                    },
                },
                "moe": None,
                "type": "minmax",
            }
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(implicit, separators=(", ", ": ")), encoding="utf-8")

    first = QuantizationConfig.load(path)
    second = QuantizationConfig.from_dict(explicit)
    assert first.fingerprint == second.fingerprint
    assert first.to_json_string() == second.to_json_string()
    assert first.to_json_string().endswith("\n")
    assert not first.to_json_string().endswith("\n\n")


@pytest.mark.parametrize(
    "value",
    [
        {"modifiers": [], "unknown": True},
        {"modifiers": [{"type": "minmax", "unknown": True}]},
        {
            "modifiers": [
                {"type": "minmax", "linear": {"weight": {"bits": 8, "granularity": "x"}}}
            ]
        },
        {
            "modifiers": [
                {
                    "type": "gptq",
                    "attention": {},
                    "linear": {"weight": {"bits": 4, "granularity": "per_channel"}},
                }
            ]
        },
        {
            "modifiers": [
                {
                    "type": "gptq",
                    "linear": {"weight": {"bits": 4, "granularity": "per_tensor"}},
                }
            ]
        },
        {"modifiers": [{"type": "gptq", "percdamp": float("nan")}]},
        {"modifiers": [{"type": "gptq", "linear": None}]},
        {"modifiers": "minmax"},
        {"modifiers": [], "include": None},
    ],
    ids=(
        "unknown-root-field",
        "unknown-modifier-field",
        "invalid-granularity",
        "gptq-attention-target",
        "gptq-per-tensor-linear",
        "non-finite-percdamp",
        "gptq-without-linear-target",
        "modifiers-not-list",
        "null-root-include",
    ),
)
def test_invalid_config_fails_without_mutating_input(value: dict[str, object]) -> None:
    original = copy.deepcopy(value)
    with pytest.raises(QuantizationConfigError):
        QuantizationConfig.from_dict(value)
    assert value == original


def test_config_is_frozen_and_load_is_idempotent() -> None:
    config = QuantizationConfig.from_dict({"modifiers": []})
    assert QuantizationConfig.load(config) is config
    with pytest.raises(FrozenInstanceError):
        config.include = ("changed",)  # type: ignore[misc]


def test_duplicate_json_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"modifiers":[],"modifiers":[]}', encoding="utf-8")
    with pytest.raises(QuantizationConfigError, match="duplicate field"):
        QuantizationConfig.load(path)


def test_public_config_and_exception_exports() -> None:
    import mdc_llm_deploy

    assert mdc_llm_deploy.QuantizationConfig is QuantizationConfig
    assert issubclass(QuantizationConfigError, MdcDeployError)
    assert issubclass(QuantizationConfigError, ValueError)
    assert mdc_llm_deploy.__version__ == "0.1.0"


def test_packaged_schema_matches_generator() -> None:
    packaged = ROOT / "mdc_llm_deploy" / "config" / "schema.json"
    assert packaged.read_text(encoding="utf-8") == schema_json()


def test_config_import_does_not_import_torch() -> None:
    command = (
        "import sys; import mdc_llm_deploy.config; "
        "raise SystemExit(1 if 'torch' in sys.modules else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
