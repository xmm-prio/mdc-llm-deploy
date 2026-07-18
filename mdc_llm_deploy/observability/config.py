"""Environment-backed observability configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TextIO

_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


def _enabled(environ: Mapping[str, str], name: str) -> bool:
    value = environ.get(name)
    return value is None or value.strip().lower() not in _FALSE_VALUES


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Immutable observability settings captured at a stage boundary."""

    logging_enabled: bool
    log_level: str
    progress_enabled: bool
    report_enabled: bool
    is_terminal: bool

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        stream: TextIO,
    ) -> ObservabilityConfig:
        """Capture settings without retaining mutable environment state."""
        source = os.environ if environ is None else environ
        requested_level = source.get("MDC_LLM_DEPLOY_LOG_LEVEL", "INFO").strip().upper()
        level = requested_level if requested_level in _LOG_LEVELS else "INFO"
        return cls(
            logging_enabled=_enabled(source, "MDC_LLM_DEPLOY_LOGGING"),
            log_level=level,
            progress_enabled=_enabled(source, "MDC_LLM_DEPLOY_PROGRESS"),
            report_enabled=_enabled(source, "MDC_LLM_DEPLOY_REPORT"),
            is_terminal=bool(stream.isatty()),
        )
