"""End-to-end tests for planning, calibration, MinMax, and GPTQ."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterator

import pytest
import torch
from torch.fx import Graph, GraphModule

import mdc_llm_deploy.quantization.materialization as quantization_materialization
from mdc_llm_deploy.errors import QuantizationConfigError
from mdc_llm_deploy.export import export
from mdc_llm_deploy.graph.lifecycle import (
    GraphMetadata,
    GraphStage,
    TensorAbi,
    metadata,
    set_metadata,
)
from mdc_llm_deploy.quantization import (
    calculate_qparams,
    oneshot,
    plan_quantization,
    quantize,
)
from mdc_llm_deploy.quantization.algorithms.math import (
    GPTQ_FALLBACK_CHOLESKY_FAILED,
    GPTQ_FALLBACK_NON_FINITE_HESSIAN,
    GptqFallbackError,
    gptq_weight_quantize,
)
from mdc_llm_deploy.quantization.calibration import collect_calibration_samples
from mdc_llm_deploy.quantization.config import QuantizationConfig
from mdc_llm_deploy.quantization.planning import (
    CalibrationPlan,
    plan_calibration,
)
from tests.support.models.qwen3 import dense_model, moe_model

pytestmark = pytest.mark.integration


def _inputs(sequence: int = 8) -> dict[str, torch.Tensor]:
    return {"input_ids": torch.arange(sequence).reshape(1, sequence) % 128}


def _graph(model: torch.nn.Module | None = None) -> torch.fx.GraphModule:
    return export(model or dense_model(8), _inputs())


def _integer_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()


def _independent_clipped_quantize(
    weight: torch.Tensor,
    *,
    bits: int,
    per_channel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    ratios = torch.tensor(
        [0.5 + index * 0.5 / 19 for index in range(20)],
        dtype=torch.float32,
    )
    assert ratios.numel() == 20
    parameter_shape = (weight.shape[0], 1) if per_channel else (1, 1)
    best_error = torch.full(parameter_shape, torch.inf)
    best_integer = torch.empty_like(weight, dtype=torch.int8)
    best_scale = torch.ones(parameter_shape, dtype=torch.float32)
    bounds = (
        weight.float().abs().amax(dim=1, keepdim=True)
        if per_channel
        else weight.float().abs().amax().reshape(1, 1)
    )
    for ratio in ratios:
        candidate_scale = torch.where(
            bounds == 0,
            torch.ones_like(bounds),
            bounds * ratio / qmax,
        )
        candidate_integer = torch.round(weight.float() / candidate_scale).clamp(qmin, qmax)
        candidate_dequantized = candidate_integer * candidate_scale
        squared_error = (weight.float() - candidate_dequantized).square()
        candidate_error = (
            squared_error.mean(dim=1, keepdim=True)
            if per_channel
            else squared_error.mean().reshape(1, 1)
        )
        improved = candidate_error < best_error
        best_error = torch.where(improved, candidate_error, best_error)
        best_scale = torch.where(improved, candidate_scale, best_scale)
        best_integer = torch.where(
            improved.expand_as(candidate_integer),
            candidate_integer.to(torch.int8),
            best_integer,
        )
    return best_integer, best_scale


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
    config = QuantizationConfig.load("configs/quantization/minmax-linear-w8a8.json")

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
        "configs/quantization/minmax-linear-w8a8.json",
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
    assert "algorithms" not in value.properties
    assert "gptq" not in value.properties
    assert value.properties["activation_qparams"]["lm_head"]["bits"] == 8
    assert len(value.properties["quantized_integer_sha256"]["lm_head"]) == 64


def test_materialization_captures_candidate_parameters_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    config = QuantizationConfig.load(
        "configs/quantization/minmax-linear-w8a8.json"
    )
    plan = plan_quantization(graph, config)
    capture_calls = 0
    snapshot_calls = 0
    capture_active = False
    original_capture = (
        quantization_materialization.MaterializationContext.capture
    )
    original_named_parameters = GraphModule.named_parameters

    def capture_spy(
        cls: type[quantization_materialization.MaterializationContext],
        candidate: GraphModule,
    ) -> quantization_materialization.MaterializationContext:
        del cls
        nonlocal capture_active, capture_calls
        capture_calls += 1
        capture_active = True
        try:
            return original_capture(candidate)
        finally:
            capture_active = False

    def named_parameters_spy(
        self: GraphModule,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        nonlocal snapshot_calls
        if capture_active:
            assert remove_duplicate is False
            snapshot_calls += 1
        return original_named_parameters(
            self,
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    monkeypatch.setattr(
        quantization_materialization.MaterializationContext,
        "capture",
        classmethod(capture_spy),
    )
    monkeypatch.setattr(
        GraphModule,
        "named_parameters",
        named_parameters_spy,
    )

    oneshot(graph, config, [_inputs()])

    assert len(plan) > 1
    assert capture_calls == 1
    assert snapshot_calls == 1


def test_attention_and_moe_materialization_contracts() -> None:
    attention = _graph()
    oneshot(attention, "configs/quantization/minmax-attention-a8.json", [_inputs()])
    attention_value = metadata(attention)

    assert {item.fqn.rsplit(".", 1)[-1] for item in attention_value.quantized_targets} == {
        "query",
        "key",
        "value",
        "score",
    }
    assert len(attention_value.properties["activation_qparams"]) == 4

    moe = _graph(moe_model(8))
    oneshot(moe, "configs/quantization/minmax-moe-w8a8.json", [_inputs()])
    moe_value = metadata(moe)

    assert len(moe_value.quantized_targets) == 1
    assert all(item.target_type == "moe" for item in moe_value.quantized_targets)
    assert moe_value.quantized_targets[0].fqn.endswith(".expert_weights")
    block = moe.get_submodule("model.layers.0.mlp")
    assert block.expert_weights.dtype is torch.int8
    assert block.quant_scales.shape == (4, 3)
    qparams = moe_value.properties["activation_qparams"][
        moe_value.quantized_targets[0].fqn
    ]
    assert len(qparams["intermediate_scale"]) == 4
    assert "moe_quant_parameter_order" not in moe_value.properties


@pytest.mark.parametrize(("bits", "per_channel"), [(4, True), (8, False)])
def test_gptq_clip_search_matches_independent_20_ratio_reference(
    bits: int,
    per_channel: bool,
) -> None:
    weight = torch.tensor(
        [
            [4.0, -3.0, 1.7, -0.2],
            [0.25, -0.75, 1.25, -1.75],
        ],
        dtype=torch.float32,
    )
    activations = torch.eye(weight.shape[1], dtype=torch.float32)
    expected_integer, expected_scale = _independent_clipped_quantize(
        weight,
        bits=bits,
        per_channel=per_channel,
    )

    actual = gptq_weight_quantize(
        weight,
        activations,
        bits=bits,
        percdamp=0.01,
        actorder=True,
        block_size=128,
        per_channel=per_channel,
    )

    assert torch.equal(actual.values, expected_integer)
    torch.testing.assert_close(actual.scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.dequantized,
        expected_integer.float() * expected_scale,
        rtol=0,
        atol=0,
    )


def test_dense_gptq_json_fx_path_materializes_w4a8() -> None:
    graph = _graph()

    same = oneshot(graph, "configs/quantization/gptq-linear-w4a8.json", [_inputs()])

    value = metadata(graph)
    targets = value.quantized_targets
    assert same is graph
    assert value.stage is GraphStage.QUANTIZED_PREFILL
    assert len(targets) == 8
    assert {item.fqn.rsplit(".", 1)[-1] for item in targets} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    }
    assert all(
        item.algorithm == "gptq"
        and item.target_type == "linear"
        and item.bits == 4
        and item.granularity == "per_channel"
        for item in targets
    )
    assert set(value.properties["activation_qparams"]) == {item.fqn for item in targets}
    assert value.properties["gptq_fallbacks"] == {}
    for target in targets:
        parameter = dict(graph.named_parameters())[f"{target.fqn}.weight"]
        scale = torch.tensor(target.scale, dtype=torch.float32).reshape(-1, 1)
        integer = torch.round(parameter.float() / scale).clamp(-8, 7).to(torch.int8)
        torch.testing.assert_close(
            parameter,
            (integer.float() * scale).to(parameter.dtype),
            rtol=0,
            atol=0,
        )
        assert value.properties["quantized_integer_sha256"][target.fqn] == _integer_sha256(
            integer
        )


def test_moe_gptq_rejects_packed_expert_weights() -> None:
    graph = _graph(moe_model(8))

    with pytest.raises(
        QuantizationConfigError,
        match="does not support packed MoeExpert",
    ):
        oneshot(graph, "configs/quantization/gptq-moe-w8a8.json", [_inputs()])


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
    assert target.fallback_reason == GPTQ_FALLBACK_CHOLESKY_FAILED
    assert first_value.properties["gptq_fallbacks"] == {
        "lm_head": GPTQ_FALLBACK_CHOLESKY_FAILED
    }
    torch.testing.assert_close(
        dict(first.named_parameters())["lm_head.weight"],
        dict(second.named_parameters())["lm_head.weight"],
        rtol=0,
        atol=0,
    )


def test_gptq_non_finite_hessian_has_stable_fallback_reason() -> None:
    weight = torch.ones((1, 2), dtype=torch.float32)
    activations = torch.full((2, 2), 2e20, dtype=torch.float32)

    with pytest.raises(GptqFallbackError) as captured:
        gptq_weight_quantize(weight, activations, bits=4)

    assert captured.value.reason == GPTQ_FALLBACK_NON_FINITE_HESSIAN


@pytest.mark.parametrize("error", [RuntimeError("unexpected"), ValueError("unexpected")])
def test_gptq_does_not_swallow_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    graph = _graph()
    before = dict(graph.named_parameters())["lm_head.weight"].detach().clone()
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
            }
        ]
    }

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(
        quantization_materialization,
        "gptq_weight_quantize",
        fail,
    )
    with pytest.raises(type(error), match="unexpected"):
        oneshot(graph, config, [_inputs()])

    assert metadata(graph).stage is GraphStage.FLOAT_PREFILL
    torch.testing.assert_close(
        dict(graph.named_parameters())["lm_head.weight"],
        before,
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
            "configs/quantization/minmax-linear-w8a8.json",
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


def _tied_minmax_config() -> dict[str, object]:
    return {
        "modifiers": [
            {
                "type": "minmax",
                "include": ["first", "second"],
                "linear": {
                    "weight": {
                        "bits": 8,
                        "granularity": "per_channel",
                        "symmetric": True,
                    }
                },
            }
        ]
    }


def _tied_moe_minmax_config() -> dict[str, object]:
    return {
        "modifiers": [
            {
                "type": "minmax",
                "include": ["experts.0", "experts.1"],
                "moe": {
                    "weight": {
                        "bits": 8,
                        "granularity": "per_tensor",
                        "symmetric": True,
                    }
                },
            }
        ]
    }


def _tied_ordinary_moe_graph() -> GraphModule:
    root = torch.nn.Module()
    root.add_module(
        "experts",
        torch.nn.ModuleList(
            [
                torch.nn.Linear(4, 4, bias=False),
                torch.nn.Linear(4, 4, bias=False),
            ]
        ),
    )
    root.experts[1].weight = root.experts[0].weight
    graph = Graph()
    value = graph.placeholder("input_ids")
    first_weight = graph.get_attr("experts.0.weight")
    first = graph.call_function(
        torch.ops.aten.linear.default,
        (value, first_weight, None),
    )
    second_weight = graph.get_attr("experts.1.weight")
    second = graph.call_function(
        torch.ops.aten.linear.default,
        (value, second_weight, None),
    )
    graph.output(graph.call_function(torch.ops.aten.add.Tensor, (first, second)))
    module = GraphModule(root, graph)
    set_metadata(
        module,
        GraphMetadata(
            schema_version=1,
            stage=GraphStage.FLOAT_PREFILL,
            model_kind="moe",
            input_abi=(TensorAbi("input_ids", "float32", (1, 4)),),
            output_abi=(TensorAbi("output", "float32", (1, 4)),),
            sequence_length=4,
        ),
    )
    return module


def test_oneshot_materializes_tied_parameter_once_and_preserves_aliases() -> None:
    class TiedLinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(4, 4, bias=False)
            self.second = torch.nn.Linear(4, 4, bias=False)
            self.second.weight = self.first.weight

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            values = input_ids.float()
            return self.first(values) + self.second(values)

    inputs = {"input_ids": torch.arange(4).reshape(1, 4)}
    graph = export(TiedLinearModel().eval(), inputs)
    before_device = graph.first.weight.device

    oneshot(graph, _tied_minmax_config(), [inputs])

    assert graph.first.weight is graph.second.weight
    assert graph.first.weight.device == before_device
    assert {target.fqn for target in metadata(graph).quantized_targets} == {
        "second"
    }


def test_minmax_tied_ordinary_moe_needs_no_calibration_samples() -> None:
    graph = _tied_ordinary_moe_graph()
    config = QuantizationConfig.load(_tied_moe_minmax_config())
    plan = plan_quantization(graph, config)
    inputs = {"input_ids": torch.arange(4, dtype=torch.float32).reshape(1, 4)}

    assert {target.fqn for target in plan} == {"experts.0", "experts.1"}
    assert plan_calibration(plan).required_fqns == frozenset()

    same = oneshot(graph, config, [inputs])

    assert same is graph
    assert (
        graph.get_submodule("experts.0").weight
        is graph.get_submodule("experts.1").weight
    )
    assert {target.fqn for target in metadata(graph).quantized_targets} == {
        "experts.0",
        "experts.1",
    }


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_oneshot_aggregates_each_target_on_its_resident_device() -> None:
    class ResidentLinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(4, 4, bias=False).to("cuda:0")
            self.second = torch.nn.Linear(4, 4, bias=False).to("cuda:1")
            self.hf_device_map = {
                "first": "cuda:0",
                "second": "cuda:1",
            }

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            hidden = self.first(input_ids.float().to("cuda:0"))
            return self.second(hidden.to("cuda:1"))

    inputs = {
        "input_ids": torch.arange(4, device="cuda:0").reshape(1, 4)
    }
    graph = export(ResidentLinearModel().eval(), inputs)
    samples = collect_calibration_samples(
        graph,
        [inputs, inputs],
        CalibrationPlan(frozenset({"first", "second"})),
    )

    assert samples["first"].device == torch.device("cuda:0")
    assert samples["second"].device == torch.device("cuda:1")

    oneshot(graph, _tied_minmax_config(), [inputs])

    assert graph.first.weight.device == torch.device("cuda:0")
    assert graph.second.weight.device == torch.device("cuda:1")
