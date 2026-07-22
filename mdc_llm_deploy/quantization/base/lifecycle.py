"""Algorithm-independent quantization lifecycle contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import StrEnum
from typing import Generic, TypeVar

from torch import nn

from .calibration import CalibrationBatch
from .config import QuantizationConfig

ConfigT = TypeVar("ConfigT", bound=QuantizationConfig)


class QuantizationState(StrEnum):
    """Stable states of an in-place quantization lifecycle."""

    UNPREPARED = "unprepared"
    PREPARED = "prepared"
    CALIBRATED = "calibrated"
    CONVERTED = "converted"


class Quantizer(ABC, Generic[ConfigT]):
    """Algorithm contract used by the public lifecycle API."""

    def __init__(self, config: ConfigT) -> None:
        self._config = config
        self._state = QuantizationState.UNPREPARED

    @property
    def config(self) -> ConfigT:
        """Return immutable algorithm configuration."""
        return self._config

    @property
    def state(self) -> QuantizationState:
        """Return current lifecycle state."""
        return self._state

    def prepare(self, model: nn.Module) -> nn.Module:
        """Validate and collect algorithm state without partial mutation."""
        self._require_state(QuantizationState.UNPREPARED, "prepare")
        self._prepare(model)
        self._state = QuantizationState.PREPARED
        return model

    def calibrate(
        self,
        model: nn.Module,
        batches: Iterable[CalibrationBatch],
        *,
        show_progress: bool = True,
    ) -> nn.Module:
        """Collect runtime statistics from model invocation batches."""
        self._require_state(QuantizationState.PREPARED, "calibrate")
        self._calibrate(model, batches, show_progress=show_progress)
        self._state = QuantizationState.CALIBRATED
        return model

    def convert(self, model: nn.Module) -> nn.Module:
        """Freeze quantization parameters and replace supported modules."""
        self._require_state(QuantizationState.CALIBRATED, "convert")
        self._convert(model)
        self._state = QuantizationState.CONVERTED
        return model

    def _require_state(self, expected: QuantizationState, operation: str) -> None:
        if self._state is not expected:
            raise RuntimeError(
                f"{operation} requires state {expected.value}, current state is {self._state.value}"
            )

    @abstractmethod
    def _prepare(self, model: nn.Module) -> None:
        """Collect algorithm-specific preparation state."""

    @abstractmethod
    def _calibrate(
        self,
        model: nn.Module,
        batches: Iterable[CalibrationBatch],
        *,
        show_progress: bool,
    ) -> None:
        """Collect algorithm-specific runtime statistics."""

    @abstractmethod
    def _convert(self, model: nn.Module) -> None:
        """Apply algorithm-specific conversion."""


__all__ = ["QuantizationState", "Quantizer"]
