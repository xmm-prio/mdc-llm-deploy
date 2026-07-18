from __future__ import annotations

import copy
import io
import sys
from collections.abc import Iterator, Mapping
from typing import ClassVar

import pytest
import torch
from torch import Tensor, nn
from torch.fx import Graph, GraphModule

import mdc_llm_deploy.quantization.api as quantization_api
import mdc_llm_deploy.quantization.calibration as calibration_module
from mdc_llm_deploy.errors import QuantizationConfigError
from mdc_llm_deploy.graph.lifecycle import (
    GraphMetadata,
    GraphStage,
    TensorAbi,
    metadata,
    set_metadata,
)
from mdc_llm_deploy.quantization import oneshot
from mdc_llm_deploy.quantization.calibration import collect_calibration_artifacts
from mdc_llm_deploy.quantization.config import ActivationSpec
from mdc_llm_deploy.quantization.planning import (
    CalibrationPlan,
    TargetPlan,
    plan_calibration,
)


def _graph() -> GraphModule:
    root = nn.Module()
    root.add_module("linear", nn.Linear(2, 2, bias=False))
    graph = Graph()
    value = graph.placeholder("x")
    weight = graph.get_attr("linear.weight")
    value = graph.call_function(
        torch.ops.aten.linear.default,
        (value, weight, None),
    )
    graph.output(value)
    module = GraphModule(root, graph)
    set_metadata(
        module,
        GraphMetadata(
            schema_version=1,
            stage=GraphStage.FLOAT_PREFILL,
            model_kind="dense",
            input_abi=(TensorAbi("x", "float32", (1, 2)),),
            output_abi=(TensorAbi("output", "float32", (1, 2)),),
            sequence_length=2,
        ),
    )
    return module


def _config(*, activation: bool) -> dict[str, object]:
    linear: dict[str, object] = {
        "weight": {
            "bits": 8,
            "granularity": "per_channel",
            "symmetric": True,
        }
    }
    if activation:
        linear["activation"] = {
            "bits": 8,
            "granularity": "per_tensor",
            "mode": "static",
            "symmetric": True,
        }
    return {
        "modifiers": [
            {
                "type": "minmax",
                "include": ["linear"],
                "linear": linear,
            }
        ]
    }


def _calibration_plan() -> CalibrationPlan:
    target = TargetPlan(
        fqn="linear",
        target_type="linear",
        algorithm="minmax",
        modifier_index=0,
        parameter_name="linear.weight",
        weight=None,
        activation=ActivationSpec(
            bits=8,
            granularity="per_tensor",
            mode="static",
        ),
    )
    return plan_calibration((target,))


class _SinglePass:
    def __init__(self, batches: tuple[Mapping[str, Tensor], ...]) -> None:
        self._batches = batches
        self.iterations = 0

    def __iter__(self) -> Iterator[Mapping[str, Tensor]]:
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("dataloader iterated more than once")
        yield from self._batches


class _ProgressSpy:
    def __init__(self, advances: list[int]) -> None:
        self._advances = advances

    def __enter__(self) -> _ProgressSpy:
        return self

    def advance(self, amount: int = 1) -> None:
        self._advances.append(amount)

    def __exit__(self, *args: object) -> None:
        del args


class _ReporterSpy:
    totals: ClassVar[list[int | None]] = []
    advances: ClassVar[list[int]] = []

    def __init__(self, stage: str) -> None:
        assert stage == "Quantization calibration"

    def __enter__(self) -> _ReporterSpy:
        return self

    def update(self, **fields: object) -> None:
        del fields

    def progress(
        self,
        description: str,
        *,
        total: int | None = None,
    ) -> _ProgressSpy:
        assert description == "Collecting calibration batches"
        self.totals.append(total)
        return _ProgressSpy(self.advances)

    def __exit__(self, *args: object) -> None:
        del args


def test_empty_configuration_reports_skipped_without_iterating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    dataloader = _SinglePass(({"x": torch.ones(1, 2)},))

    same = oneshot(_graph(), {"modifiers": []}, dataloader)

    assert metadata(same).stage is GraphStage.FLOAT_PREFILL
    assert dataloader.iterations == 0
    output = stream.getvalue()
    assert "Quantization planning" in output
    assert "skipped" in output
    assert "Quantization calibration" not in output
    assert "Quantization materialization" not in output


def test_weight_only_flow_reports_all_stages_without_consuming_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    dataloader = _SinglePass(({"x": torch.ones(1, 2)},))

    oneshot(_graph(), _config(activation=False), dataloader)

    assert dataloader.iterations == 0
    output = stream.getvalue()
    assert "Quantization planning" in output
    assert "Quantization calibration" in output
    assert "Quantization materialization" in output
    assert "skipped" in output
    assert "linear" not in output


def test_calibration_progress_uses_declared_or_unknown_total_and_single_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ReporterSpy.totals = []
    _ReporterSpy.advances = []
    monkeypatch.setattr(calibration_module, "StageReporter", _ReporterSpy)
    batches = (
        {"x": torch.ones(1, 2)},
        {"x": torch.full((1, 2), 2.0)},
    )
    generator = _SinglePass(batches)

    collect_calibration_artifacts(_graph(), list(batches), _calibration_plan())
    collect_calibration_artifacts(_graph(), generator, _calibration_plan())

    assert _ReporterSpy.totals == [2, None]
    assert _ReporterSpy.advances == [1, 1, 1, 1]
    assert generator.iterations == 1


def test_failed_batch_is_not_counted_and_progress_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ReporterSpy.totals = []
    _ReporterSpy.advances = []
    monkeypatch.setattr(calibration_module, "StageReporter", _ReporterSpy)
    dataloader = _SinglePass(
        (
            {"x": torch.ones(1, 2)},
            {"x": torch.ones(2, 2)},
        )
    )

    with pytest.raises(QuantizationConfigError, match="Calibration shape"):
        collect_calibration_artifacts(_graph(), dataloader, _calibration_plan())

    assert _ReporterSpy.totals == [None]
    assert _ReporterSpy.advances == [1]
    assert dataloader.iterations == 1


def test_observability_does_not_change_qparams_metadata_or_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    visible = _graph()
    silent = copy.deepcopy(visible)
    batches = [{"x": torch.tensor([[1.0, 2.0]])}]

    oneshot(visible, _config(activation=True), batches)
    stream.seek(0)
    stream.truncate()
    monkeypatch.setenv("MDC_LLM_DEPLOY_LOGGING", "off")
    monkeypatch.setenv("MDC_LLM_DEPLOY_PROGRESS", "off")
    monkeypatch.setenv("MDC_LLM_DEPLOY_REPORT", "off")
    oneshot(silent, _config(activation=True), batches)

    assert stream.getvalue() == ""
    assert metadata(visible) == metadata(silent)
    for (visible_name, visible_parameter), (
        silent_name,
        silent_parameter,
    ) in zip(
        visible.named_parameters(),
        silent.named_parameters(),
        strict=True,
    ):
        assert visible_name == silent_name
        torch.testing.assert_close(
            visible_parameter,
            silent_parameter,
            rtol=0,
            atol=0,
        )


def test_materialization_failure_reports_failed_and_preserves_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    graph = _graph()
    metadata_before = metadata(graph)
    weight_before = graph.linear.weight.detach().clone()

    def fail_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("materialization failed")

    monkeypatch.setattr(
        quantization_api,
        "materialize_alias_group",
        fail_materialization,
    )

    with pytest.raises(RuntimeError, match=r"^materialization failed$"):
        oneshot(graph, _config(activation=False), ())

    output = stream.getvalue()
    assert "Quantization materialization" in output
    assert "FAILED" in output
    assert metadata(graph) == metadata_before
    torch.testing.assert_close(graph.linear.weight, weight_before, rtol=0, atol=0)
