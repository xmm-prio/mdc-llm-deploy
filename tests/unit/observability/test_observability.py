from __future__ import annotations

import io
import logging
import subprocess
import sys
from collections.abc import Iterator, Mapping
from typing import TextIO, cast

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


class _ExplodingTerminalBuffer(io.StringIO):
    def isatty(self) -> bool:
        raise OSError("terminal probe failed")


class _StreamWithoutIsatty:
    pass


class _ExplodingFields(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError("disabled reports must not inspect fields")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("disabled reports must not inspect fields")

    def __len__(self) -> int:
        raise AssertionError("disabled reports must not inspect fields")


@pytest.fixture(autouse=True)
def _restore_package_logger() -> Iterator[None]:
    logger = get_logger()
    saved_handlers = tuple(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate
    for handler in saved_handlers:
        logger.removeHandler(handler)

    yield

    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    for handler in saved_handlers:
        logger.addHandler(handler)
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate


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
        color_enabled=False,
    )


@pytest.mark.parametrize(
    "stream",
    [
        _ExplodingTerminalBuffer(),
        cast(TextIO, _StreamWithoutIsatty()),
    ],
)
def test_unsafe_or_missing_isatty_disables_terminal_features(stream: TextIO) -> None:
    config = ObservabilityConfig.from_env(environ={}, stream=stream)

    assert not config.is_terminal
    assert not config.color_enabled


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


def test_stage_reporter_preserves_host_logging_route() -> None:
    logger = get_logger()
    host_stream = io.StringIO()
    package_stream = io.StringIO()
    package_config = ObservabilityConfig.from_env(environ={}, stream=package_stream)
    configure_package_logger(package_config, stream=package_stream)
    assert len(logger.handlers) == 1

    host_handler = logging.StreamHandler(host_stream)
    host_formatter = logging.Formatter("HOST %(levelname)s: %(message)s")
    host_handler.setFormatter(host_formatter)
    host_handler.setLevel(logging.INFO)
    logger.addHandler(host_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    logger.info("before")
    package_stream.seek(0)
    package_stream.truncate()
    with StageReporter(
        "embedded",
        stream=package_stream,
        environ={
            "MDC_LLM_DEPLOY_LOGGING": "off",
            "MDC_LLM_DEPLOY_PROGRESS": "off",
            "MDC_LLM_DEPLOY_REPORT": "off",
        },
    ):
        pass
    logger.info("after")

    output = host_stream.getvalue()
    assert "before" in output
    assert "Stage started: embedded" in output
    assert "Stage completed: embedded" in output
    assert "after" in output
    assert package_stream.getvalue() == ""
    assert logger.handlers == [host_handler]
    assert logger.level == logging.DEBUG
    assert logger.propagate
    assert host_handler.level == logging.INFO
    assert host_handler.formatter is host_formatter


def test_root_import_preserves_preconfigured_host_logger() -> None:
    command = (
        "import logging; "
        "logger = logging.getLogger('mdc_llm_deploy'); "
        "handler = logging.StreamHandler(); "
        "formatter = logging.Formatter('HOST %(message)s'); "
        "handler.setFormatter(formatter); "
        "handler.setLevel(logging.WARNING); "
        "logger.addHandler(handler); "
        "logger.setLevel(logging.DEBUG); "
        "logger.propagate = True; "
        "import mdc_llm_deploy; "
        "assert logger.handlers == [handler]; "
        "assert logger.level == logging.DEBUG; "
        "assert logger.propagate is True; "
        "assert handler.level == logging.WARNING; "
        "assert handler.formatter is formatter"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


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


def test_terminal_logs_use_semantic_level_colors() -> None:
    stream = _TerminalBuffer()
    config = ObservabilityConfig.from_env(
        environ={"MDC_LLM_DEPLOY_LOG_LEVEL": "DEBUG"},
        stream=stream,
    )
    logger = configure_package_logger(config, stream=stream)

    logger.debug("debug detail")
    logger.info("info detail")
    logger.warning("warning detail")
    logger.error("error detail")
    logger.critical("critical detail")

    output = stream.getvalue()
    assert "\x1b[90mDEBUG" in output
    assert "\x1b[34mINFO" in output
    assert "\x1b[33mWARNING" in output
    assert "\x1b[31mERROR" in output
    assert "\x1b[1;31mCRITICAL" in output
    for message in (
        "debug detail",
        "info detail",
        "warning detail",
        "error detail",
        "critical detail",
    ):
        assert message in output


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


@pytest.mark.parametrize(
    "environ",
    [
        {"NO_COLOR": ""},
        {"NO_COLOR": "0"},
        {"TERM": "DuMb"},
    ],
)
def test_terminal_color_controls_do_not_disable_progress(
    environ: dict[str, str],
) -> None:
    stream = _TerminalBuffer()
    selected_environ = {
        **environ,
        "MDC_LLM_DEPLOY_LOG_LEVEL": "DEBUG",
        "MDC_LLM_DEPLOY_REPORT": "off",
    }

    with (
        StageReporter("no-color", stream=stream, environ=selected_environ) as reporter,
        reporter.progress("items") as progress,
    ):
        assert reporter.config.is_terminal
        assert not reporter.config.color_enabled
        assert progress.enabled
        get_logger("test").debug("plain detail")
        progress.advance()

    output = stream.getvalue()
    assert "\x1b[" not in output
    assert "DEBUG" in output
    assert "plain detail" in output
    assert "1 completed" in output
    assert "\r" in output


def test_terminal_report_and_progress_use_semantic_colors() -> None:
    stream = _TerminalBuffer()

    with (
        StageReporter(
            "export",
            stream=stream,
            environ={"MDC_LLM_DEPLOY_LOGGING": "off"},
        ) as reporter,
        reporter.progress("nodes") as progress,
    ):
        progress.advance()

    output = stream.getvalue()
    assert "\x1b[1;36mexport" in output
    assert "\x1b[32mSUCCESS" in output
    assert "\x1b[36m" in output


def test_terminal_failed_report_uses_red_status() -> None:
    stream = _TerminalBuffer()

    with (
        pytest.raises(ValueError, match="business failure"),
        StageReporter(
            "quantize",
            stream=stream,
            environ={"MDC_LLM_DEPLOY_LOGGING": "off"},
        ),
    ):
        raise ValueError("business failure")

    assert "\x1b[31mFAILED" in stream.getvalue()


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
