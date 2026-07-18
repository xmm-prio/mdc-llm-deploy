"""Isolated package logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

from .config import ObservabilityConfig

PACKAGE_LOGGER_NAME = "mdc_llm_deploy"


class _PackageStderrHandler(logging.StreamHandler[TextIO]):
    """Identify the handler owned by this package."""


def configure_package_logger(
    config: ObservabilityConfig,
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure one package-owned handler without touching the root logger."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    selected_stream = sys.stderr if stream is None else stream
    owned_handlers = [
        handler for handler in logger.handlers if isinstance(handler, _PackageStderrHandler)
    ]
    if owned_handlers:
        handler = owned_handlers[0]
        handler.setStream(selected_stream)
        for duplicate in owned_handlers[1:]:
            logger.removeHandler(duplicate)
            duplicate.close()
    else:
        handler = _PackageStderrHandler(selected_stream)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
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
