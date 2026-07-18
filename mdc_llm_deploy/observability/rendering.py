"""Rich rendering primitives for local stage output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TextIO

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .config import ObservabilityConfig

_SENSITIVE_FRAGMENTS = ("password", "secret", "token", "credential", "path")
_TERMINAL_THEME = Theme(
    {
        "logging.level.debug": "bright_black",
        "logging.level.info": "blue",
        "logging.level.warning": "yellow",
        "logging.level.error": "red",
        "logging.level.critical": "bold red",
        "report.title": "bold cyan",
        "report.success": "green",
        "report.failed": "red",
        "progress": "cyan",
    }
)


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
        force_terminal=config.color_enabled,
        color_system="standard" if config.color_enabled else None,
        no_color=not config.color_enabled,
        highlight=False,
        soft_wrap=True,
        theme=_TERMINAL_THEME,
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
    table = Table(title=stage, title_style="report.title", show_header=True)
    table.add_column("Field")
    table.add_column("Value")
    status_style = "report.success" if status == "SUCCESS" else "report.failed"
    table.add_row("status", Text(status, style=status_style))
    table.add_row("elapsed_seconds", f"{elapsed_seconds:.3f}")
    for key, value in sanitize_fields(fields).items():
        table.add_row(key, value)
    _console(stream, config).print(table)


class _PlainTerminalProgress:
    """Render dynamic no-color progress using carriage returns only."""

    def __init__(self, description: str, stream: TextIO) -> None:
        self._description = description
        self._stream = stream
        self._completed = 0
        self._started = False

    def start(self) -> None:
        self._started = True
        self._render()

    def advance(self, amount: int) -> None:
        self._completed += amount
        if self._started:
            self._render()

    def stop(self) -> None:
        if self._started:
            self._stream.write("\n")
            self._stream.flush()
            self._started = False

    def _render(self) -> None:
        self._stream.write(
            f"\r{self._description} {self._completed} completed"
        )
        self._stream.flush()


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
        self._progress: Progress | _PlainTerminalProgress | None = None
        self._task_id: TaskID | None = None
        if self._enabled:
            if not config.color_enabled:
                self._progress = _PlainTerminalProgress(description, stream)
                return
            self._progress = Progress(
                SpinnerColumn(style="progress"),
                TextColumn("{task.description}", style="progress"),
                TextColumn("{task.completed} completed", style="progress"),
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
        if isinstance(self._progress, _PlainTerminalProgress):
            self._progress.advance(amount)
        elif self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id, amount)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._progress is not None:
            self._progress.stop()
