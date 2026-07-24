"""Shared logging and terminal progress primitives."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter

from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

_PACKAGE_LOGGER_NAME = "mdc_llm_deploy"


def configure_logging() -> None:
    """Configure colored package logging unless the application already configured logging."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    if package_logger.handlers:
        return

    handler = RichHandler(rich_tracebacks=True)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a library logger without configuring application-wide handlers."""
    return logging.getLogger(name)


@contextmanager
def log_stage(
    logger: logging.Logger,
    stage: str,
    *,
    details: str = "",
) -> Iterator[None]:
    """Log one stage boundary, duration, and failure without hiding exceptions."""
    suffix = f" ({details})" if details else ""
    logger.info("%s started%s", stage, suffix)
    started_at = perf_counter()
    try:
        yield
    except Exception:
        logger.exception("%s failed after %.3fs%s", stage, perf_counter() - started_at, suffix)
        raise
    logger.info("%s completed in %.3fs%s", stage, perf_counter() - started_at, suffix)


@contextmanager
def progress_task(
    description: str,
    *,
    total: int | None,
    show_progress: bool,
) -> Iterator[Callable[[], None]]:
    """Create one optional Rich progress task and return its advance callback."""
    if not show_progress:
        yield lambda: None
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task_id = progress.add_task(description, total=total)

        def advance() -> None:
            progress.advance(task_id)

        yield advance


__all__ = ["configure_logging", "get_logger", "log_stage", "progress_task"]
