"""Rich rendering primitives for local stage output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TextIO

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table

from .config import ObservabilityConfig

_SENSITIVE_FRAGMENTS = ("password", "secret", "token", "credential", "path")


def sanitize_fields(fields: Mapping[str, object]) -> dict[str, str]:
    """Convert scalar report fields while redacting secret-bearing keys."""
    sanitized: dict[str, str] = {}
    for key, value in fields.items():
        normalized_key = str(key)
        if any(fragment in normalized_key.lower() for fragment in _SENSITIVE_FRAGMENTS):
            sanitized[normalized_key] = "<redacted>"
        elif value is None or isinstance(value, (bool, int, float, str)):
            sanitized[normalized_key] = str(value)
        else:
            sanitized[normalized_key] = f"<{type(value).__name__}>"
    return sanitized


def _console(stream: TextIO, config: ObservabilityConfig) -> Console:
    return Console(
        file=stream,
        force_terminal=config.is_terminal,
        color_system="standard" if config.is_terminal else None,
        no_color=not config.is_terminal,
        highlight=False,
        soft_wrap=True,
    )


def render_stage_report(
    *,
    stream: TextIO,
    config: ObservabilityConfig,
    stage: str,
    status: str,
    elapsed_seconds: float,
    fields: Mapping[str, object],
) -> None:
    """Render one static stage table when reporting is enabled."""
    if not config.report_enabled:
        return
    table = Table(title=stage, show_header=True)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("status", status)
    table.add_row("elapsed_seconds", f"{elapsed_seconds:.3f}")
    for key, value in sanitize_fields(fields).items():
        table.add_row(key, value)
    _console(stream, config).print(table)


class StageProgress:
    """Manage a stage-local Rich progress task without consuming inputs."""

    def __init__(
        self,
        description: str,
        *,
        config: ObservabilityConfig,
        stream: TextIO,
        total: int | None = None,
    ) -> None:
        self._enabled = config.progress_enabled and config.is_terminal
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        if self._enabled:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                TextColumn("{task.completed} completed"),
                console=_console(stream, config),
                transient=False,
            )
            self._task_id = self._progress.add_task(description, total=total)

    @property
    def enabled(self) -> bool:
        """Report whether dynamic rendering is active."""
        return self._enabled

    def __enter__(self) -> StageProgress:
        if self._progress is not None:
            self._progress.start()
        return self

    def advance(self, amount: int = 1) -> None:
        """Advance only after caller work completes."""
        if self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id, amount)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._progress is not None:
            self._progress.stop()
