"""Stage-local observability lifecycle."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from time import perf_counter
from types import TracebackType
from typing import Literal, TextIO

from .config import ObservabilityConfig
from .logging import configure_package_logger, get_logger
from .rendering import StageProgress, render_stage_report


class _OverlayFields(Mapping[str, object]):
    """Overlay updated fields without eagerly scanning base fields."""

    def __init__(
        self,
        updates: Mapping[str, object],
        base: Mapping[str, object],
    ) -> None:
        self._updates = updates
        self._base = base

    def __getitem__(self, key: str) -> object:
        try:
            return self._updates[key]
        except KeyError:
            return self._base[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._updates
        yield from (key for key in self._base if key not in self._updates)

    def __len__(self) -> int:
        return len(set(self._base) | set(self._updates))


class StageReporter:
    """Log and render one stage without retaining cross-stage state."""

    def __init__(
        self,
        stage: str,
        *,
        fields: Mapping[str, object] | None = None,
        stream: TextIO | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.stage = stage
        self._base_fields = fields if fields is not None else {}
        self._updated_fields: dict[str, object] = {}
        self._stream = sys.stderr if stream is None else stream
        self._environ = environ
        self._config: ObservabilityConfig | None = None
        self._started_at: float | None = None

    @property
    def config(self) -> ObservabilityConfig:
        """Return settings captured on stage entry."""
        if self._config is None:
            raise RuntimeError("Stage reporter has not been entered")
        return self._config

    def __enter__(self) -> StageReporter:
        self._config = ObservabilityConfig.from_env(
            environ=self._environ,
            stream=self._stream,
        )
        configure_package_logger(self._config, stream=self._stream)
        self._started_at = perf_counter()
        get_logger("stage").info("Stage started: %s", self.stage)
        return self

    def update(self, **fields: object) -> None:
        """Add already-derived scalar fields to the final report."""
        self._updated_fields.update(fields)

    def progress(self, description: str, *, total: int | None = None) -> StageProgress:
        """Create known- or unknown-total progress without inspecting inputs."""
        return StageProgress(
            description,
            config=self.config,
            stream=self._stream,
            total=total,
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc_value, traceback
        started_at = self._started_at
        elapsed = 0.0 if started_at is None else perf_counter() - started_at
        status = "SUCCESS" if exc_type is None else "FAILED"
        logger = get_logger("stage")
        if exc_type is None:
            logger.info("Stage completed: %s", self.stage)
        else:
            logger.error("Stage failed: %s (%s)", self.stage, exc_type.__name__)
        try:
            render_stage_report(
                stream=self._stream,
                config=self.config,
                stage=self.stage,
                status=status,
                elapsed_seconds=elapsed,
                fields=_OverlayFields(self._updated_fields, self._base_fields),
            )
        except Exception:
            logger.warning("Stage report rendering failed: %s", self.stage)
        return False
