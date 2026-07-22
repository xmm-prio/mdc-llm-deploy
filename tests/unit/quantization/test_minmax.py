from __future__ import annotations

import pytest
import torch
from torch import nn

from mdc_llm_deploy.quantization import (
    MinMaxConfig,
    MinMaxLinear,
    QuantizationState,
    TargetSelector,
    calibrate,
    convert,
    prepare,
    quantization_state,
    quantize,
)
from mdc_llm_deploy.quantization.api import load_quantized_state_dict
from mdc_llm_deploy.quantization.minmax.observer import QuantizationParameters


def _supported_configs() -> list[MinMaxConfig]:
    configs: list[MinMaxConfig] = []
    for weight, activation in ((True, False), (False, True), (True, True)):
        weight_options = (
            [(granularity, symmetric) for granularity in ("per_tensor", "per_channel") for symmetric in (True, False)]
            if weight
            else [("per_tensor", True)]
        )
        activation_options = (
            [(granularity, symmetric) for granularity in ("per_tensor", "per_token") for symmetric in (True, False)]
            if activation
            else [("per_tensor", True)]
        )
        for weight_granularity, weight_symmetric in weight_options:
            for activation_granularity, activation_symmetric in activation_options:
                configs.append(
                    MinMaxConfig(
                        weight=weight,
                        activation=activation,
                        weight_granularity=weight_granularity,  # type: ignore[arg-type]
                        activation_granularity=activation_granularity,  # type: ignore[arg-type]
                        weight_symmetric=weight_symmetric,
                        activation_symmetric=activation_symmetric,
                    )
                )
    return configs


def test_minmax_linear_uses_per_tensor_symmetric_weight_fake_quantization() -> None:
    model = nn.Sequential(nn.Linear(2, 2, bias=True))
    original_weight = nn.Parameter(torch.tensor([[1.0, 0.5], [-1.0, 0.1]]))
    original_bias = nn.Parameter(torch.tensor([0.25, -0.25]))
    model[0].weight = original_weight
    model[0].bias = original_bias
    inputs = torch.tensor([[2.0, -1.0]])

    returned = quantize(model, MinMaxConfig())
    output = model(inputs)

    assert returned is model
    assert isinstance(model[0], MinMaxLinear)
    assert model[0].weight is original_weight
    assert model[0].bias is original_bias
    torch.testing.assert_close(model[0].weight_scale, torch.tensor(1.0 / 127.0))
    expected_weight = torch.round(original_weight.detach() * 127.0).clamp(-128, 127) / 127.0
    expected = torch.nn.functional.linear(inputs, expected_weight, original_bias)
    torch.testing.assert_close(output, expected)
    assert model[0].weight_zero_point is None


def test_public_minmax_linear_constructor_preserves_qparam_contract() -> None:
    source = nn.Linear(2, 2, bias=False)
    source.weight = nn.Parameter(
        torch.tensor([[1.0, 0.5], [-1.0, 0.25]], dtype=torch.float32)
    )
    inputs = torch.tensor([[1.0, -1.0]])
    expected = source(inputs)

    module = MinMaxLinear(
        source,
        weight_qparams=QuantizationParameters(
            scale=torch.tensor([[0.5], [0.25]], dtype=torch.float64),
            zero_point=torch.zeros((2, 1), dtype=torch.int16),
        ),
        activation_qparams=QuantizationParameters(
            scale=torch.tensor(0.25, dtype=torch.float64),
            zero_point=None,
        ),
        weight_axis=0,
        activation_axis=None,
    )

    assert module.weight is source.weight
    assert module.weight_axis == 0
    assert module.activation_axis is None
    assert module.weight_scale.dtype is torch.float32
    assert module.weight_zero_point.dtype is torch.int8
    assert module.activation_scale.dtype is torch.float32
    assert set(dict(module.named_buffers(remove_duplicate=False))) == {
        "weight_scale",
        "weight_zero_point",
        "weight_qdq_scale",
        "weight_qdq_zero_point",
        "activation_scale",
        "activation_qdq_scale",
    }
    assert set(module.state_dict()) == {
        "weight",
        "weight_scale",
        "weight_zero_point",
        "activation_scale",
    }
    torch.testing.assert_close(module(inputs), expected)


def test_zero_weight_range_uses_unit_scale() -> None:
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.zero_()

    quantize(model, MinMaxConfig())

    assert isinstance(model[0], MinMaxLinear)
    torch.testing.assert_close(model[0].weight_scale, torch.tensor(1.0))
    torch.testing.assert_close(model(torch.ones(1, 2)), torch.zeros(1, 2))


def test_selector_only_converts_matching_linear_modules() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    config = MinMaxConfig(targets=TargetSelector(include=("0",)))

    quantize(model, config)

    assert isinstance(model[0], MinMaxLinear)
    assert type(model[1]) is nn.Linear


def test_shared_linear_aliases_use_one_replacement() -> None:
    class SharedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = nn.Linear(2, 2)
            self.left = shared
            self.right = shared

    model = SharedModel()

    quantize(model, MinMaxConfig())

    assert isinstance(model.left, MinMaxLinear)
    assert model.left is model.right


def test_shared_alias_selection_conflict_is_atomic() -> None:
    class SharedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = nn.Linear(2, 2)
            self.left = shared
            self.right = shared

    model = SharedModel()
    original = model.left
    config = MinMaxConfig(targets=TargetSelector(include=("left",)))

    with pytest.raises(ValueError, match="consistent selection"):
        quantize(model, config)

    assert model.left is original
    assert model.right is original


def test_tied_parameters_remain_tied_between_distinct_linears() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    model[1].weight = model[0].weight
    tied_weight = model[0].weight

    quantize(model, MinMaxConfig())

    assert isinstance(model[0], MinMaxLinear)
    assert isinstance(model[1], MinMaxLinear)
    assert model[0].weight is tied_weight
    assert model[1].weight is tied_weight


def test_quantized_parameters_are_serialized_without_session_state() -> None:
    model = nn.Sequential(nn.Linear(2, 2))

    quantize(model, MinMaxConfig())

    assert set(model.state_dict()) == {"0.weight", "0.bias", "0.weight_scale"}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("config", _supported_configs())
def test_all_24_configurations_support_eager_dtypes(
    config: MinMaxConfig,
    dtype: torch.dtype,
) -> None:
    model = nn.Sequential(nn.Linear(4, 3, dtype=dtype))
    inputs = torch.linspace(-2, 2, 24, dtype=dtype).reshape(2, 3, 4)
    batches = [((inputs,), {})] if config.activation else ()

    quantize(model, config, batches)
    output = model(inputs)

    assert isinstance(model[0], MinMaxLinear)
    assert output.dtype is dtype
    assert bool(torch.isfinite(output).all())
    for scale in (model[0].weight_scale, model[0].activation_scale):
        if scale is not None:
            assert scale.dtype is torch.float32
            assert bool(torch.isfinite(scale).all())
            assert bool((scale > 0).all())
    assert (model[0].weight_zero_point is None) is (
        not config.weight or config.weight_symmetric
    )
    assert (model[0].activation_zero_point is None) is (
        not config.activation or config.activation_symmetric
    )


def test_per_channel_asymmetric_weight_qparams_follow_int8_formula() -> None:
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    model[0].weight = nn.Parameter(
        torch.tensor([[-1.0, 0.0, 3.0], [2.0, 4.0, 6.0]])
    )

    quantize(
        model,
        MinMaxConfig(weight_granularity="per_channel", weight_symmetric=False),
    )

    assert isinstance(model[0], MinMaxLinear)
    expected_scale = torch.tensor([[4.0 / 255.0], [6.0 / 255.0]])
    expected_zero_point = torch.round(
        torch.tensor([[-128.0], [-128.0]]) - torch.tensor([[-1.0], [0.0]]) / expected_scale
    ).clamp(-128, 127).to(torch.int8)
    torch.testing.assert_close(model[0].weight_scale, expected_scale)
    torch.testing.assert_close(model[0].weight_zero_point, expected_zero_point)
    assert model[0].weight_axis == 0


def test_per_token_observer_streams_by_position_and_freezes_broadcast_shape() -> None:
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    first = torch.tensor([[[1.0, 2.0], [-4.0, 1.0]]])
    second = torch.tensor([[[3.0, -2.0], [2.0, 8.0]]])
    config = MinMaxConfig(
        weight=False,
        activation=True,
        activation_granularity="per_token",
    )

    quantize(model, config, [((first,), {}), ((second,), {})])

    assert isinstance(model[0], MinMaxLinear)
    torch.testing.assert_close(
        model[0].activation_scale,
        torch.tensor([[[3.0 / 127.0], [8.0 / 127.0]]]),
    )
    assert model[0].activation_axis == -2


def test_per_token_rejects_calibration_and_inference_shape_changes() -> None:
    config = MinMaxConfig(
        weight=False,
        activation=True,
        activation_granularity="per_token",
    )
    model = nn.Sequential(nn.Linear(2, 2))
    prepare(model, config)

    with pytest.raises(ValueError, match="axis length changed"):
        calibrate(
            model,
            [
                ((torch.ones(1, 2, 2),), {}),
                ((torch.ones(1, 3, 2),), {}),
            ],
        )

    model = nn.Sequential(nn.Linear(2, 2))
    quantize(model, config, [((torch.ones(1, 2, 2),), {})])
    with pytest.raises(ValueError, match="activation length changed"):
        model(torch.ones(1, 3, 2))
    with pytest.raises(ValueError, match="activation rank changed"):
        model(torch.ones(2, 2))


def test_convert_requires_activation_coverage_for_every_target() -> None:
    class PartialModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = nn.Linear(2, 2)
            self.unused = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.used(inputs)

    model = PartialModel()
    config = MinMaxConfig(weight=False, activation=True)
    prepare(model, config)
    calibrate(model, [((torch.ones(1, 2),), {})])

    with pytest.raises(RuntimeError, match=r"did not cover.*unused"):
        convert(model)

    assert type(model.used) is nn.Linear
    assert type(model.unused) is nn.Linear
    assert quantization_state(model) is QuantizationState.CALIBRATED


def test_load_quantized_state_dict_strictly_restores_converted_structure() -> None:
    torch.manual_seed(7)
    config = MinMaxConfig(
        weight=True,
        activation=True,
        weight_granularity="per_channel",
        activation_granularity="per_token",
        weight_symmetric=False,
        activation_symmetric=False,
    )
    source = nn.Sequential(nn.Linear(4, 3))
    inputs = torch.randn(2, 5, 4)
    quantize(source, config, [((inputs,), {})])
    expected = source(inputs)
    checkpoint = {name: value.clone() for name, value in source.state_dict().items()}

    restored = nn.Sequential(nn.Linear(4, 3))
    returned = load_quantized_state_dict(restored, config, checkpoint)

    assert returned is restored
    assert isinstance(restored[0], MinMaxLinear)
    assert quantization_state(restored) is QuantizationState.CONVERTED
    assert restored[0].activation_scale.shape == (1, 5, 1)
    torch.testing.assert_close(restored(inputs), expected)
    assert set(restored.state_dict()) == set(checkpoint)


def test_load_quantized_state_dict_rejects_qparam_shape_without_mutation() -> None:
    config = MinMaxConfig(weight_granularity="per_channel")
    source = nn.Sequential(nn.Linear(2, 2))
    quantize(source, config)
    checkpoint = {name: value.clone() for name, value in source.state_dict().items()}
    checkpoint["0.weight_scale"] = torch.ones(3, 1)
    restored = nn.Sequential(nn.Linear(2, 2))
    original = restored[0]

    with pytest.raises(ValueError, match="weight scale shape"):
        load_quantized_state_dict(restored, config, checkpoint)

    assert restored[0] is original
    assert quantization_state(restored) is QuantizationState.UNPREPARED


def test_load_quantized_state_dict_rejects_inconsistent_alias_qparams_atomically() -> None:
    class SharedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = nn.Linear(2, 2)
            self.left = shared
            self.right = shared

    source = SharedModel()
    quantize(source, MinMaxConfig())
    checkpoint = {name: value.clone() for name, value in source.state_dict().items()}
    checkpoint["right.weight_scale"] = checkpoint["right.weight_scale"] * 2
    restored = SharedModel()
    original = restored.left

    with pytest.raises(ValueError, match="inconsistent weight scales"):
        load_quantized_state_dict(restored, MinMaxConfig(), checkpoint)

    assert restored.left is original
    assert restored.right is original
    assert quantization_state(restored) is QuantizationState.UNPREPARED


def test_load_quantized_state_dict_rejects_zero_point_dtype_atomically() -> None:
    config = MinMaxConfig(weight_symmetric=False)
    source = nn.Sequential(nn.Linear(2, 2))
    quantize(source, config)
    checkpoint = {name: value.clone() for name, value in source.state_dict().items()}
    checkpoint["0.weight_zero_point"] = checkpoint["0.weight_zero_point"].to(torch.int32)
    restored = nn.Sequential(nn.Linear(2, 2))
    original = restored[0]

    with pytest.raises(TypeError, match=r"weight zero-point must use torch\.int8"):
        load_quantized_state_dict(restored, config, checkpoint)

    assert restored[0] is original
    assert quantization_state(restored) is QuantizationState.UNPREPARED


def test_load_quantized_state_dict_rejects_zero_point_shape_atomically() -> None:
    config = MinMaxConfig(weight_symmetric=False)
    source = nn.Sequential(nn.Linear(2, 2))
    quantize(source, config)
    checkpoint = {name: value.clone() for name, value in source.state_dict().items()}
    checkpoint["0.weight_zero_point"] = torch.zeros(2, dtype=torch.int8)
    restored = nn.Sequential(nn.Linear(2, 2))
    original = restored[0]

    with pytest.raises(ValueError, match="weight zero-point shape must match scale shape"):
        load_quantized_state_dict(restored, config, checkpoint)

    assert restored[0] is original
    assert quantization_state(restored) is QuantizationState.UNPREPARED
