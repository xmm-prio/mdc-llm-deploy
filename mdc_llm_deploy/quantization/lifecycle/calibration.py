"""Algorithm-independent calibration batch handling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sized
from typing import Any, TypeAlias

import torch
from torch import nn

from ...core.observability import get_logger, progress_task

CalibrationBatch: TypeAlias = Mapping[str, Any]
_logger = get_logger(__name__)


def run_calibration(
    model: nn.Module,
    batches: Iterable[CalibrationBatch],
    *,
    show_progress: bool = True,
) -> None:
    """Run validated model invocation batches while preserving training state."""
    training = model.training
    total = len(batches) if isinstance(batches, Sized) else None
    processed = 0
    _logger.info(
        "Calibration batches started: batch_count=%s original_training=%s inference_mode=True",
        total if total is not None else "unknown",
        training,
    )
    model.eval()
    try:
        with (
            torch.inference_mode(),
            progress_task(
                "Calibrating batches",
                total=total,
                show_progress=show_progress,
            ) as advance,
        ):
            for batch in batches:
                if not isinstance(batch, Mapping):
                    raise TypeError("each calibration batch must be a mapping")
                model(**dict(batch))
                processed += 1
                advance()
    finally:
        model.train(training)
        _logger.debug(
            "Calibration model mode restored: training=%s processed_batches=%d",
            training,
            processed,
        )
    _logger.info(
        "Calibration batches completed: processed_batches=%d restored_training=%s",
        processed,
        training,
    )


__all__ = ["CalibrationBatch", "run_calibration"]
