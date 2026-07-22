"""Algorithm-independent calibration batch handling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias

import torch
from torch import nn

CalibrationBatch: TypeAlias = tuple[tuple[Any, ...], Mapping[str, Any]]


def run_calibration(model: nn.Module, batches: Iterable[CalibrationBatch]) -> None:
    """Run validated model invocation batches while preserving training state."""
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                if not isinstance(batch, tuple) or len(batch) != 2:
                    raise TypeError("each calibration batch must be an (args, kwargs) tuple")
                args, kwargs = batch
                if not isinstance(args, tuple):
                    raise TypeError("calibration batch args must be a tuple")
                if not isinstance(kwargs, Mapping):
                    raise TypeError("calibration batch kwargs must be a mapping")
                model(*args, **dict(kwargs))
    finally:
        model.train(training)


__all__ = ["CalibrationBatch", "run_calibration"]
