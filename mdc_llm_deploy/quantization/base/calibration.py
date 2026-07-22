"""Algorithm-independent calibration batch handling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias

import torch
from torch import nn

CalibrationBatch: TypeAlias = Mapping[str, Any]


def run_calibration(model: nn.Module, batches: Iterable[CalibrationBatch]) -> None:
    """Run validated model invocation batches while preserving training state."""
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                if not isinstance(batch, Mapping):
                    raise TypeError("each calibration batch must be a mapping")
                model(**dict(batch))
    finally:
        model.train(training)


__all__ = ["CalibrationBatch", "run_calibration"]
