from __future__ import annotations

import io
import logging
from collections.abc import Iterator, Mapping

import pytest

from mdc_llm_deploy.observability import (
    ObservabilityConfig,
    StageReporter,
    get_logger,
)
from mdc_llm_deploy.observability.logging import configure_package_logger


class _TerminalBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


class _ExplodingFields(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError("disabled reports must not inspect fields")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("disabled reports must not inspect fields")

    def __len__(self) -> int:
        raise AssertionError("disabled reports must not inspect fields")


def test_environment_defaults_on_and_controls_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    for name in (
        "MDC_LLM_DEPLOY_LOGGING",
        "MDC_LLM_DEPLOY_LOG_LEVEL",
        "MDC_LLM_DEPLOY_PROGRESS",
        "MDC_LLM_DEPLOY_REPORT",
    ):
        monkeypatch.delenv(name, raising=False)

    defaults = ObservabilityConfig.from_env(stream=stream)
    selected = ObservabilityConfig.from_env(
        environ={
            "MDC_LLM_DEPLOY_LOGGING": "off",
            "MDC_LLM_DEPLOY_LOG_LEVEL": "debug",
            "MDC_LLM_DEPLOY_PROGRESS": "0",
        },
        stream=stream,
    )

    assert defaults.logging_enabled
    assert defaults.progress_enabled
    assert defaults.report_enabled
    assert defaults.log_level == "INFO"
    assert selected == ObservabilityConfig(
        logging_enabled=False,
        log_level="DEBUG",
        progress_enabled=False,
        report_enabled=True,
        is_terminal=False,
    )


def test_package_logger_isolated_from_root_and_has_one_owned_handler() -> None:
    root = logging.getLogger()
    root_state = (root.level, tuple(root.handlers), root.propagate)
    stream = io.StringIO()
    config = ObservabilityConfig.from_env(environ={}, stream=stream)

    first = configure_package_logger(config, stream=stream)
    second = configure_package_logger(config, stream=stream)

    assert first is second is get_logger()
    assert len(first.handlers) == 1
    assert not first.propagate
    assert (root.level, tuple(root.handlers), root.propagate) == root_state


def test_log_level_controls_debug_output() -> None:
    stream = io.StringIO()
    with StageReporter(
        "debug-stage",
        stream=stream,
        environ={
            "MDC_LLM_DEPLOY_LOG_LEVEL": "DEBUG",
            "MDC_LLM_DEPLOY_REPORT": "off",
        },
    ):
        get_logger("test").debug("Operator detail")

    assert "DEBUG mdc_llm_deploy.test: Operator detail" in stream.getvalue()


def test_non_terminal_uses_static_plain_report_and_no_dynamic_progress() -> None:
    stream = io.StringIO()
    with StageReporter("export", stream=stream, environ={}) as reporter:
        with reporter.progress("nodes") as progress:
            assert not progress.enabled
            progress.advance()
        reporter.update(nodes=3)

    output = stream.getvalue()
    assert "SUCCESS" in output
    assert "nodes" in output
    assert "\x1b[" not in output
    assert "\r" not in output


def test_terminal_progress_supports_unknown_total() -> None:
    stream = _TerminalBuffer()
    with (
        StageReporter(
            "calibration",
            stream=stream,
            environ={"MDC_LLM_DEPLOY_REPORT": "off"},
        ) as reporter,
        reporter.progress("batches") as progress,
    ):
        assert progress.enabled
        progress.advance()

    assert "1 completed" in stream.getvalue()


def test_failed_stage_keeps_business_exception_and_redacts_secret_fields() -> None:
    stream = io.StringIO()

    with (
        pytest.raises(ValueError, match="business failure"),
        StageReporter(
            "quantize",
            fields={
                "api_token": "private",
                "output_path": "C:/private/model",
                "count": 2,
            },
            stream=stream,
            environ={},
        ),
    ):
        raise ValueError("business failure")

    output = stream.getvalue()
    assert "FAILED" in output
    assert "<redacted>" in output
    assert "private" not in output
    assert "C:/private/model" not in output
    assert "business failure" not in output


def test_rendering_failure_does_not_replace_business_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_render(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("renderer failure")

    monkeypatch.setattr("mdc_llm_deploy.observability.stage.render_stage_report", fail_render)

    with (
        pytest.raises(LookupError, match="business failure"),
        StageReporter("export", stream=io.StringIO(), environ={}),
    ):
        raise LookupError("business failure")


def test_disabled_features_do_not_inspect_report_fields() -> None:
    stream = io.StringIO()
    disabled = {
        "MDC_LLM_DEPLOY_LOGGING": "false",
        "MDC_LLM_DEPLOY_PROGRESS": "no",
        "MDC_LLM_DEPLOY_REPORT": "0",
    }

    with (
        StageReporter(
            "silent",
            fields=_ExplodingFields(),
            stream=stream,
            environ=disabled,
        ) as reporter,
        reporter.progress("work") as progress,
    ):
        assert not progress.enabled
        progress.advance()

    assert stream.getvalue() == ""
