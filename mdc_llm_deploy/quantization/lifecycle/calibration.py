"""Algorithm-independent calibration batch handling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sized
from typing import Any, TypeAlias

import torch
from torch import nn

from ..._observability import progress_task

CalibrationBatch: TypeAlias = Mapping[str, Any]


def run_calibration(
    model: nn.Module,
    batches: Iterable[CalibrationBatch],
    *,
    show_progress: bool = True,
) -> None:
    """Run validated model invocation batches while preserving training state."""
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            total = len(batches) if isinstance(batches, Sized) else None
            with progress_task(
                "Calibrating batches",
                total=total,
                show_progress=show_progress,
            ) as advance:
                for batch in batches:
                    if not isinstance(batch, Mapping):
                        raise TypeError("each calibration batch must be a mapping")
                    model(**dict(batch))
                    advance()
    finally:
        model.train(training)


__all__ = ["CalibrationBatch", "run_calibration"]
