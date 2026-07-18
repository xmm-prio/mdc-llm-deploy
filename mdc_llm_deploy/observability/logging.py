"""Isolated package logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

from rich.logging import RichHandler

from .config import ObservabilityConfig
from .rendering import _console

PACKAGE_LOGGER_NAME = "mdc_llm_deploy"


class _PackageStderrHandler(logging.StreamHandler[TextIO]):
    """Identify the handler owned by this package."""


class _PackageRichHandler(RichHandler):
    """Identify the terminal handler owned by this package."""


_OwnedHandler = _PackageStderrHandler | _PackageRichHandler


def _new_handler(
    config: ObservabilityConfig,
    stream: TextIO,
) -> _OwnedHandler:
    """Build the package handler appropriate for the captured terminal."""
    handler: _OwnedHandler
    if config.is_terminal:
        handler = _PackageRichHandler(
            console=_console(stream, config),
            show_time=False,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
        )
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        return handler

    handler = _PackageStderrHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    return handler


def configure_package_logger(
    config: ObservabilityConfig,
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure one package-owned handler without touching the root logger."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    selected_stream = sys.stderr if stream is None else stream
    owned_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, (_PackageStderrHandler, _PackageRichHandler))
    ]
    expected_type = _PackageRichHandler if config.is_terminal else _PackageStderrHandler
    matching_handlers = [
        handler for handler in owned_handlers if isinstance(handler, expected_type)
    ]
    if matching_handlers:
        handler = matching_handlers[0]
        if isinstance(handler, _PackageStderrHandler):
            handler.setStream(selected_stream)
        else:
            handler.console = _console(selected_stream, config)
        for duplicate in owned_handlers:
            if duplicate is not handler:
                logger.removeHandler(duplicate)
                duplicate.close()
    else:
        for previous in owned_handlers:
            logger.removeHandler(previous)
            previous.close()
        handler = _new_handler(config, selected_stream)
        logger.addHandler(handler)

    level = getattr(logging, config.log_level)
    handler.setLevel(level)
    logger.setLevel(level if config.logging_enabled else logging.CRITICAL + 1)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger or one of its descendants."""
    if not name:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name == PACKAGE_LOGGER_NAME or name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")
