"""End-to-end tests for planning, calibration, MinMax, and GPTQ."""

from __future__ import annotations

import copy

import pytest
import torch

from mdc_llm_deploy.config import QuantizationConfig
from mdc_llm_deploy.errors import QuantizationConfigError
from mdc_llm_deploy.export import export
from mdc_llm_deploy.graph import GraphStage, metadata
from mdc_llm_deploy.models import TinyQwen3Dense, TinyQwen3Moe
from mdc_llm_deploy.quantization import (
    calculate_qparams,
    oneshot,
    plan_quantization,
    quantize,
)


def _inputs(sequence: int = 8) -> dict[str, torch.Tensor]:
    return {"input_ids": torch.arange(sequence).reshape(1, sequence) % 128}


def _graph(model: torch.nn.Module | None = None) -> torch.fx.GraphModule:
    return export(model or TinyQwen3Dense().eval(), _inputs())


def test_minmax_zero_rule_and_ties_to_even() -> None:
    zeros = torch.zeros(4, dtype=torch.float32)
    scale, zero_point = calculate_qparams(zeros, bits=8, symmetric=True)

    assert scale.item() == 1.0
    assert zero_point.item() == 0
    assert torch.equal(quantize(zeros, bits=8, symmetric=True).values, torch.zeros(4, dtype=torch.int8))

    # qmax=7 and max=7 produce scale=1, exposing exact half-way values.
    values = quantize(
        torch.tensor([-7.0, -2.5, -1.5, 1.5, 2.5, 7.0]),
        bits=4,
        symmetric=True,
    ).values
    assert values.tolist() == [-7, -2, -2, 2, 2, 7]


def test_planner_selects_only_aten_linear_parameters() -> None:
    graph = _graph()
    config = QuantizationConfig.load("configs/minmax-linear-w8a8.json")

    plan = plan_quantization(graph, config)

    assert len(plan) == 8
    assert "embed_tokens" not in {item.fqn for item in plan}
    assert "lm_head" in {item.fqn for item in plan}
    assert all(item.parameter_name and item.parameter_name.endswith(".weight") for item in plan)


def test_minmax_linear_materializes_independent_reference() -> None:
    graph = _graph()
    before = dict(graph.named_parameters())["lm_head.weight"].detach().clone()
    expected = quantize(before, bits=8, symmetric=True, axis=0)

    same = oneshot(
        graph,
        "configs/minmax-linear-w8a8.json",
        [_inputs()],
    )

    assert same is graph
    actual = dict(graph.named_parameters())["lm_head.weight"]
    torch.testing.assert_close(actual, expected.dequantized, rtol=0, atol=0)
    value = metadata(graph)
    target = next(item for item in value.quantized_targets if item.fqn == "lm_head")
    assert value.stage is GraphStage.QUANTIZED_PREFILL
    assert target.scale == tuple(float(item) for item in expected.scale.reshape(-1))
    assert target.zero_point == tuple(int(item) for item in expected.zero_point.reshape(-1))
    assert value.properties["fake_quant"] is True
    assert value.properties["activation_qparams"]["lm_head"]["bits"] == 8
    assert len(value.properties["quantized_integer_sha256"]["lm_head"]) == 64


def test_attention_and_moe_materialization_contracts() -> None:
    attention = _graph()
    oneshot(attention, "configs/minmax-attention-a8.json", [_inputs()])
    attention_value = metadata(attention)

    assert {item.fqn.rsplit(".", 1)[-1] for item in attention_value.quantized_targets} == {
        "query",
        "key",
        "value",
        "score",
    }
    assert len(attention_value.properties["activation_qparams"]) == 4

    moe = _graph(TinyQwen3Moe().eval())
    oneshot(moe, "configs/minmax-moe-w8a8.json", [_inputs()])
    moe_value = metadata(moe)

    assert len(moe_value.quantized_targets) == 15
    assert all(item.target_type == "moe" for item in moe_value.quantized_targets)
    assert len(moe_value.properties["moe_quant_parameter_order"]) == 21
    assert moe_value.properties["moe_quant_parameter_order"][0] == "input"


def test_gptq_is_deterministic_and_records_limited_fallback() -> None:
    first = _graph()
    second = _graph()
    config = {
        "modifiers": [
            {
                "type": "gptq",
                "include": ["lm_head"],
                "linear": {
                    "weight": {
                        "bits": 4,
                        "granularity": "per_channel",
                        "symmetric": True,
                    }
                },
                "percdamp": 0.0,
            }
        ]
    }

    oneshot(first, config, [_inputs()])
    oneshot(second, config, [_inputs()])

    first_value = metadata(first)
    target = first_value.quantized_targets[0]
    assert target.fqn == "lm_head"
    assert target.bits == 4
    assert target.fallback_reason == "cholesky_failed:_LinAlgError"
    assert first_value.properties["gptq_fallbacks"] == {
        "lm_head": "cholesky_failed:_LinAlgError"
    }
    torch.testing.assert_close(
        dict(first.named_parameters())["lm_head.weight"],
        dict(second.named_parameters())["lm_head.weight"],
        rtol=0,
        atol=0,
    )


def test_quantization_failures_preserve_graph_and_parameters() -> None:
    graph = _graph()
    before_graph = str(graph.graph)
    before_metadata = copy.deepcopy(metadata(graph))
    before_parameters = {
        name: parameter.detach().clone() for name, parameter in graph.named_parameters()
    }

    with pytest.raises(QuantizationConfigError, match="Calibration keys"):
        oneshot(
            graph,
            "configs/minmax-linear-w8a8.json",
            [{"wrong": _inputs()["input_ids"]}],
        )

    assert str(graph.graph) == before_graph
    assert metadata(graph) == before_metadata
    for name, parameter in graph.named_parameters():
        torch.testing.assert_close(parameter, before_parameters[name], rtol=0, atol=0)


def test_overlapping_modifiers_are_rejected_before_mutation() -> None:
    graph = _graph()
    config = {
        "modifiers": [
            {
                "type": "minmax",
                "include": ["lm_head"],
                "linear": {
                    "weight": {
                        "bits": 8,
                        "granularity": "per_channel",
                        "symmetric": True,
                    }
                },
            },
            {
                "type": "minmax",
                "include": ["lm_head"],
                "linear": {
                    "weight": {
                        "bits": 8,
                        "granularity": "per_channel",
                        "symmetric": True,
                    }
                },
            },
        ]
    }

    with pytest.raises(QuantizationConfigError, match="selected by modifiers"):
        oneshot(graph, config, [_inputs()])

    assert metadata(graph).stage is GraphStage.FLOAT_PREFILL
