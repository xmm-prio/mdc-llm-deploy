from __future__ import annotations

import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import onnx
import pytest
import torch

from mdc_llm_deploy.custom_ops import (
    create_onnx_export_profile,
    registered_operators,
)
from mdc_llm_deploy.custom_ops.apply_rotary_pos_emb import apply_rotary_pos_emb
from mdc_llm_deploy.custom_ops.fused_infer_attention_score import (
    fused_infer_attention_score,
)
from mdc_llm_deploy.custom_ops.moe_expert import moe_expert
from mdc_llm_deploy.custom_ops.rms_norm import rms_norm

_OPERATOR_NAMES = (
    "apply_rotary_pos_emb",
    "rms_norm",
    "fused_infer_attention_score",
    "moe_expert",
)
_ONNX_NAMES = (
    "ApplyRotaryPosEmb",
    "NPURmsNorm",
    "FusedInferAttentionScore",
    "MoeExpert",
)


class _CombinedModel(torch.nn.Module):
    def forward(
        self,
        rope_query: torch.Tensor,
        rope_key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        norm_input: torch.Tensor,
        gamma: torch.Tensor,
        attention_query: torch.Tensor,
        attention_key: torch.Tensor,
        attention_value: torch.Tensor,
        moe_input: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
        quant_scales: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        rotated_query, rotated_key = apply_rotary_pos_emb(
            rope_query, rope_key, cos, sin, 1, "half"
        )
        normalized, rstd = rms_norm(norm_input, gamma, 1e-5)
        attention, softmax_lse = fused_infer_attention_score(
            attention_query,
            attention_key,
            attention_value,
            num_heads=4,
            scale=0.5,
            input_layout="BNSD",
            num_key_value_heads=2,
        )
        expert = cast(
            torch.Tensor,
            moe_expert(
                moe_input,
                topk_ids,
                topk_weight,
                expert_weights,
                quant_scales,
            ),
        )
        return (
            rotated_query,
            rotated_key,
            normalized,
            rstd,
            attention,
            softmax_lse,
            expert,
        )


def _combined_inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(127)
    angles = torch.randn(1, 3, 1, 8, generator=generator)
    return (
        torch.randn(1, 3, 4, 8, generator=generator),
        torch.randn(1, 3, 2, 8, generator=generator),
        angles.cos(),
        angles.sin(),
        torch.randn(2, 3, 16, generator=generator),
        torch.randn(16, generator=generator),
        torch.randn(1, 4, 1, 8, dtype=torch.float16, generator=generator),
        torch.randn(1, 2, 5, 8, dtype=torch.float16, generator=generator),
        torch.randn(1, 2, 5, 8, dtype=torch.float16, generator=generator),
        torch.randint(-8, 8, (1, 256), dtype=torch.int8, generator=generator),
        torch.tensor([[0, 1]], dtype=torch.int16),
        torch.tensor([[0.375, 0.625]], dtype=torch.float16),
        torch.randint(
            -8,
            8,
            (2 * 3 * 128, 256),
            dtype=torch.int8,
            generator=generator,
        ),
        torch.rand(9, dtype=torch.float32, generator=generator),
    )


@pytest.fixture(scope="module")
def combined_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("combined_custom_ops") / "combined.onnx"
    profile = create_onnx_export_profile(*_OPERATOR_NAMES)
    torch.onnx.export(
        _CombinedModel().eval(),
        _combined_inputs(),
        path,
        opset_version=18,
        dynamo=True,
        verbose=False,
        external_data=False,
        optimize=False,
        custom_translation_table=dict(profile.custom_translation_table),
        input_names=[
            "rope_query",
            "rope_key",
            "cos",
            "sin",
            "norm_input",
            "gamma",
            "attention_query",
            "attention_key",
            "attention_value",
            "moe_input",
            "topk_ids",
            "topk_weight",
            "expert_weights",
            "quant_scales",
        ],
        output_names=[
            "rotated_query",
            "rotated_key",
            "normalized",
            "rstd",
            "attention",
            "softmax_lse",
            "expert",
        ],
    )
    return path


@pytest.mark.integration
def test_combined_dynamo_export_has_exact_default_domain_abis(
    combined_model_path: Path,
) -> None:
    model = onnx.load(combined_model_path)
    onnx.checker.check_model(model, full_check=True)
    assert [(item.domain, item.version) for item in model.opset_import] == [("", 18)]

    nodes = {node.op_type: node for node in model.graph.node if node.op_type in _ONNX_NAMES}
    assert tuple(nodes) == _ONNX_NAMES
    assert all(node.domain == "" for node in nodes.values())
    assert tuple(nodes["ApplyRotaryPosEmb"].input) == (
        "rope_query",
        "rope_key",
        "cos",
        "sin",
    )
    assert tuple(nodes["ApplyRotaryPosEmb"].output) == (
        "rotated_query",
        "rotated_key",
    )
    assert tuple(nodes["NPURmsNorm"].input) == ("norm_input", "gamma")
    assert tuple(nodes["NPURmsNorm"].output) == ("normalized", "rstd")
    assert tuple(nodes["FusedInferAttentionScore"].input) == (
        "attention_query",
        "attention_key",
        "attention_value",
    )
    assert tuple(nodes["FusedInferAttentionScore"].output) == ("attention",)
    assert tuple(nodes["MoeExpert"].input) == (
        "moe_input",
        "topk_ids",
        "topk_weight",
        "expert_weights",
        "quant_scales",
    )
    assert tuple(nodes["MoeExpert"].output) == ("expert",)

    attributes = {
        name: {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        for name, node in nodes.items()
    }
    assert attributes["ApplyRotaryPosEmb"] == {
        "layout": 1,
        "rotary_mode": b"half",
    }
    assert attributes["NPURmsNorm"] == {"epsilon": pytest.approx(1e-5)}
    assert attributes["FusedInferAttentionScore"] == {
        "input_layout": b"BNSD",
        "num_heads": 4,
        "num_key_value_heads": 2,
        "scale": pytest.approx(0.5),
    }
    assert attributes["MoeExpert"] == {}


@pytest.mark.integration
def test_real_profiles_preserve_order_deduplicate_and_support_concurrency() -> None:
    permutations = tuple(itertools.permutations(_OPERATOR_NAMES))

    with ThreadPoolExecutor(max_workers=8) as executor:
        profiles = tuple(
            executor.map(
                lambda names: create_onnx_export_profile(*names, *names),
                permutations * 2,
            )
        )

    for names, profile in zip(permutations * 2, profiles, strict=True):
        assert tuple(profile.operators) == names
        assert len(profile.custom_translation_table) == len(_OPERATOR_NAMES)


@pytest.mark.integration
@pytest.mark.parametrize("selected", _OPERATOR_NAMES)
def test_new_process_loads_only_selected_plugin_and_schema(selected: str) -> None:
    script = f"""
import importlib
import onnx
from mdc_llm_deploy.custom_ops import (
    create_onnx_export_profile,
    registered_operators,
)

operators = {dict(zip(_OPERATOR_NAMES, _ONNX_NAMES, strict=True))!r}
selected = {selected!r}
importlib.import_module(f"mdc_llm_deploy.custom_ops.{{selected}}")
if tuple(entry.plugin.name for entry in registered_operators()) != ({selected!r},):
    raise SystemExit(1)
for plugin_name, onnx_name in operators.items():
    try:
        onnx.defs.get_schema(onnx_name, 18, "")
    except onnx.defs.SchemaError:
        exists = False
    else:
        exists = True
    if exists:
        raise SystemExit(2)
create_onnx_export_profile({selected!r}, {selected!r})
for plugin_name, onnx_name in operators.items():
    try:
        schema = onnx.defs.get_schema(onnx_name, 18, "")
    except onnx.defs.SchemaError:
        schema = None
    if (schema is not None) != (plugin_name == {selected!r}):
        raise SystemExit(3)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_serialized_model_requires_profile_again_in_new_process(
    combined_model_path: Path,
) -> None:
    script = f"""
import importlib
import onnx

path = {str(combined_model_path)!r}
try:
    onnx.checker.check_model(path, full_check=True)
except onnx.checker.ValidationError:
    pass
else:
    raise SystemExit(1)

from mdc_llm_deploy.custom_ops import create_onnx_export_profile
names = {_OPERATOR_NAMES!r}
for name in reversed(names):
    importlib.import_module(f"mdc_llm_deploy.custom_ops.{{name}}")
create_onnx_export_profile(*reversed(names))
onnx.checker.check_model(path, full_check=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_all_real_plugins_are_loaded_once() -> None:
    names = tuple(entry.plugin.name for entry in registered_operators())
    assert all(names.count(name) == 1 for name in _OPERATOR_NAMES)
