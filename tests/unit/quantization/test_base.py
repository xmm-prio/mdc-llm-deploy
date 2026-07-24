from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import Tensor, nn

from mdc_llm_deploy.quantization import (
    MinMaxConfig,
    QuantizationState,
    TargetSelector,
    calibrate,
    convert,
    prepare,
    quantization_state,
    quantize,
)


class _RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.observed_training: bool | None = None
        self.observed_grad_enabled: bool | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        self.observed_training = self.training
        self.observed_grad_enabled = torch.is_grad_enabled()
        return self.linear(inputs)


def test_selector_uses_include_and_exclude_with_exclude_priority() -> None:
    selector = TargetSelector(include=("encoder.*",), exclude=("*.output",))

    assert selector.matches("encoder.input")
    assert not selector.matches("encoder.output")
    assert not selector.matches("decoder.input")


@pytest.mark.parametrize(
    ("include", "exclude"),
    [
        ((), ()),
        (("",), ()),
        (("*",), ("",)),
    ],
)
def test_selector_rejects_invalid_patterns(
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        TargetSelector(include=include, exclude=exclude)


def test_three_stage_api_is_in_place_and_tracks_state() -> None:
    model = _RecordingModel().train()
    inputs = torch.ones(1, 2)

    assert prepare(model, MinMaxConfig()) is model
    assert quantization_state(model) is QuantizationState.PREPARED
    assert calibrate(model, [{"inputs": inputs}]) is model
    assert quantization_state(model) is QuantizationState.CALIBRATED
    assert model.training
    assert model.observed_training is False
    assert model.observed_grad_enabled is False
    assert convert(model) is model
    assert quantization_state(model) is QuantizationState.CONVERTED


def test_calibration_progress_and_stage_logs_can_be_controlled(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = _RecordingModel()
    prepare(model, MinMaxConfig())
    capsys.readouterr()

    with caplog.at_level("INFO"):
        calibrate(model, [{"inputs": torch.ones(1, 2)}], show_progress=True)

    captured = capsys.readouterr()
    assert "Calibrating batches" in captured.out + captured.err
    assert "Quantization calibration completed" in caplog.text
    assert "batch_count=1 show_progress=True" in caplog.text
    assert "processed_batches=1 restored_training=True" in caplog.text

    second_model = _RecordingModel()
    prepare(second_model, MinMaxConfig())
    capsys.readouterr()
    calibrate(second_model, [{"inputs": torch.ones(1, 2)}], show_progress=False)

    captured = capsys.readouterr()
    assert "Calibrating batches" not in captured.out + captured.err


def test_calibration_progress_does_not_preconsume_generator() -> None:
    model = _RecordingModel()
    events: list[str] = []

    def batches() -> Iterator[dict[str, Tensor]]:
        events.append("started")
        yield {"inputs": torch.ones(1, 2)}
        events.append("finished")

    prepared_batches = batches()
    prepare(model, MinMaxConfig())
    assert events == []

    calibrate(model, prepared_batches, show_progress=False)

    assert events == ["started", "finished"]


def test_lifecycle_rejects_out_of_order_operations() -> None:
    model = nn.Sequential(nn.Linear(2, 2))

    with pytest.raises(RuntimeError, match="has not been prepared"):
        calibrate(model)

    prepare(model, MinMaxConfig())
    with pytest.raises(RuntimeError, match="calibrated"):
        convert(model)
    with pytest.raises(RuntimeError, match="active quantization lifecycle"):
        prepare(model, MinMaxConfig())


def test_prepare_failure_is_atomic() -> None:
    model = nn.Sequential(nn.Linear(2, 2))
    original = model[0]
    with torch.no_grad():
        model[0].weight[0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        prepare(model, MinMaxConfig())

    assert model[0] is original
    assert quantization_state(model) is QuantizationState.UNPREPARED


def test_one_step_failure_removes_partial_lifecycle(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _RecordingModel()
    original = model.linear

    with (
        caplog.at_level("ERROR", logger="mdc_llm_deploy.quantization.lifecycle.api"),
        pytest.raises(TypeError, match="mapping"),
    ):
        quantize(model, MinMaxConfig(), batches=[object()])  # type: ignore[list-item]

    assert model.linear is original
    assert quantization_state(model) is QuantizationState.UNPREPARED
    failures = [
        record
        for record in caplog.records
        if record.levelname == "ERROR" and record.message.startswith("Quantization workflow failed")
    ]
    assert len(failures) == 1
    assert failures[0].exc_info is None
    assert "failed_stage=calibration" in failures[0].message
    assert "state_before_cleanup=prepared" in failures[0].message
    assert "lifecycle_state_removed=True" in failures[0].message


def test_convert_rejects_structure_change_before_replacement() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    original_second = model[1]
    prepare(model, MinMaxConfig())
    calibrate(model)
    model[0] = nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="changed after prepare"):
        convert(model)

    assert model[1] is original_second
    assert quantization_state(model) is QuantizationState.CALIBRATED
