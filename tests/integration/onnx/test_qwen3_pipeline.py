from __future__ import annotations

from collections import Counter

import onnx
import pytest
import torch

from mdc_llm_deploy.onnx import process_onnx
from mdc_llm_deploy.onnx.schemas import (
    FUSED_INFER_ATTENTION_SCORE_OP,
    RMS_NORM_OP,
    ROTARY_POSITION_EMBEDDING_OP,
)

from .qwen3_export_fixtures import EXPORT_CASES, Qwen3ExportCase, export_static_generation

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(autouse=True)
def offline_export_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.mark.parametrize("case", EXPORT_CASES, ids=lambda case: case.id)
def test_qwen3_prefill_and_real_decode_complete_pipeline(case: Qwen3ExportCase) -> None:
    components = export_static_generation(case)

    for component_name in ("prefill", "decode"):
        model = components[component_name]

        returned = process_onnx(model)

        operators = Counter(node.op_type for node in model.graph.node)
        assert returned is model
        assert operators[RMS_NORM_OP] == 5
        assert operators[ROTARY_POSITION_EMBEDDING_OP] == 1
        assert operators[FUSED_INFER_ATTENTION_SCORE_OP] == (
            0 if case.dtype is torch.float32 else 1
        )
        onnx.checker.check_model(model)
