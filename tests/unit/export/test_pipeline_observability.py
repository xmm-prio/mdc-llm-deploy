from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from mdc_llm_deploy.export import convert_to_decode, export
from mdc_llm_deploy.models import AutoExportModel, ExportModelConfig
from mdc_llm_deploy.observability import get_logger
from tests.support.models.qwen3 import dense_model

_OBSERVABILITY_ENV = (
    "MDC_LLM_DEPLOY_LOGGING",
    "MDC_LLM_DEPLOY_PROGRESS",
    "MDC_LLM_DEPLOY_REPORT",
)


def _clear_package_handlers() -> None:
    logger = get_logger()
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture(autouse=True)
def _reset_package_handlers() -> Iterator[None]:
    _clear_package_handlers()
    yield
    _clear_package_handlers()


def _set_observability(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
) -> None:
    value = "1" if enabled else "0"
    for name in _OBSERVABILITY_ENV:
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize("enabled", [False, True])
def test_fx_export_observability_preserves_graph(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    enabled: bool,
) -> None:
    _set_observability(monkeypatch, enabled=enabled)
    model = dense_model(4)
    inputs = {"input_ids": torch.arange(4).reshape(1, 4)}

    graph = export(model, inputs)

    output = capsys.readouterr().err
    assert graph(**inputs)[0].shape == (1, 4, 128)
    if enabled:
        assert "FX export" in output
        assert "SUCCESS" in output
        assert "node_count" in output
        assert "input_abi_count" in output
    else:
        assert output == ""


@pytest.mark.parametrize("enabled", [False, True])
def test_decode_observability_preserves_return_identity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    enabled: bool,
) -> None:
    _set_observability(monkeypatch, enabled=enabled)
    graph = export(
        dense_model(4),
        {"input_ids": torch.arange(4).reshape(1, 4)},
    )
    capsys.readouterr()

    result = convert_to_decode(graph)

    output = capsys.readouterr().err
    assert result is graph
    if enabled:
        assert "Decode conversion" in output
        assert "SUCCESS" in output
        assert "cache_count" in output
        assert "node_count" in output
    else:
        assert output == ""


def test_observability_toggle_preserves_fx_and_decode_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = {"input_ids": torch.arange(4).reshape(1, 4)}
    model = dense_model(4)
    _set_observability(monkeypatch, enabled=False)
    disabled = export(model, inputs)

    _clear_package_handlers()
    _set_observability(monkeypatch, enabled=True)
    enabled = export(model, inputs)

    assert str(enabled.graph) == str(disabled.graph)
    _clear_package_handlers()
    _set_observability(monkeypatch, enabled=False)
    disabled_result = convert_to_decode(disabled)
    _clear_package_handlers()
    _set_observability(monkeypatch, enabled=True)
    enabled_result = convert_to_decode(enabled)
    assert disabled_result is disabled
    assert enabled_result is enabled
    assert str(enabled.graph) == str(disabled.graph)


@pytest.mark.parametrize("enabled", [False, True])
def test_model_loading_failure_is_reported_without_source_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    _set_observability(monkeypatch, enabled=enabled)
    (tmp_path / "config.json").write_text(
        '{"model_type": "unsupported"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        AutoExportModel.from_pretrained(tmp_path, ExportModelConfig(4))

    output = capsys.readouterr().err
    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == "Unsupported checkpoint model_type: 'unsupported'"
    if enabled:
        assert "Model loading" in output
        assert "FAILED" in output
    else:
        assert output == ""
    assert str(tmp_path) not in output
