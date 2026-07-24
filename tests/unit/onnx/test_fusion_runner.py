from __future__ import annotations

from dataclasses import dataclass

import onnx
import pytest
from onnx import TensorProto, helper

from mdc_llm_deploy.onnx import FusionReport, run_fusion_passes
from mdc_llm_deploy.onnx.fusion import FusionPassResult
from mdc_llm_deploy.onnx.fusion import runner as fusion_runner


def _identity_model() -> onnx.ModelProto:
    value = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "identity",
        [value],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


@dataclass(frozen=True)
class _RecordingPass:
    name: str
    fail: bool = False

    def apply(self, model: onnx.ModelProto) -> FusionPassResult:
        model.doc_string += self.name
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return FusionPassResult(self.name, 0)


def test_run_fusion_passes_uses_stable_public_order() -> None:
    model = _identity_model()

    report = run_fusion_passes(model)

    assert isinstance(report, FusionReport)
    assert tuple(report.counts) == (
        "rms_norm",
        "apply_rotary_pos_emb",
        "fused_infer_attention_score",
    )
    assert report.total_fused_count == 0


def test_run_fusion_passes_accepts_explicit_ordered_subset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _identity_model()
    passes = (_RecordingPass("second"), _RecordingPass("first"))

    with caplog.at_level("INFO", logger="mdc_llm_deploy.onnx.fusion.runner"):
        report = run_fusion_passes(model, passes=passes)

    assert tuple(report.counts) == ("second", "first")
    assert model.doc_string == "secondfirst"
    assert "Fusion passes started: pass_count=2 passes=('second', 'first')" in caplog.text
    assert "changed_pass_count=0 total_fused_count=0" in caplog.text


def test_run_fusion_passes_keeps_prior_results_when_later_pass_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _identity_model()
    monkeypatch.setattr(
        fusion_runner,
        "_FUSION_PASSES",
        (_RecordingPass("first"), _RecordingPass("second", fail=True)),
    )

    with (
        caplog.at_level("ERROR", logger="mdc_llm_deploy.onnx.fusion.runner"),
        pytest.raises(RuntimeError, match="second failed"),
    ):
        run_fusion_passes(model)

    assert model.doc_string == "firstsecond"
    failures = [
        record
        for record in caplog.records
        if record.levelname == "ERROR" and "Fusion pass second failed" in record.message
    ]
    assert len(failures) == 1
    assert failures[0].exc_info is None


def test_run_fusion_passes_rejects_non_model() -> None:
    with pytest.raises(TypeError, match=r"onnx\.ModelProto"):
        run_fusion_passes(object())  # type: ignore[arg-type]
