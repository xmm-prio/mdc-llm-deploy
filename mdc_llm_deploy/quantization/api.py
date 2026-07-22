"""Public in-place quantization lifecycle API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import singledispatch
from typing import Any

from torch import Tensor, nn

from .base import CalibrationBatch, QuantizationConfig, QuantizationState, Quantizer
from .minmax import MinMaxConfig, MinMaxQuantizer

_SESSION_ATTRIBUTE = "_mdc_quantization_session"


@dataclass(slots=True)
class _QuantizationSession:
    quantizer: Quantizer[Any]


@singledispatch
def _create_quantizer(config: QuantizationConfig) -> Quantizer[Any]:
    raise TypeError(f"unsupported quantization config type: {type(config).__qualname__}")


@_create_quantizer.register
def _(config: MinMaxConfig) -> Quantizer[Any]:
    return MinMaxQuantizer(config)


def prepare(model: nn.Module, config: QuantizationConfig) -> nn.Module:
    """Prepare a model in place and return the same object."""
    if _session(model) is not None:
        raise RuntimeError("model already has an active quantization lifecycle")
    quantizer = _create_quantizer(config)
    quantizer.prepare(model)
    setattr(model, _SESSION_ATTRIBUTE, _QuantizationSession(quantizer))
    return model


def calibrate(
    model: nn.Module,
    batches: Iterable[CalibrationBatch] = (),
) -> nn.Module:
    """Calibrate a prepared model in place and return the same object."""
    _required_session(model).quantizer.calibrate(model, batches)
    return model


def convert(model: nn.Module) -> nn.Module:
    """Convert a calibrated model in place and return the same object."""
    _required_session(model).quantizer.convert(model)
    return model


def quantize(
    model: nn.Module,
    config: QuantizationConfig,
    batches: Iterable[CalibrationBatch] = (),
) -> nn.Module:
    """Run prepare, calibrate, and convert in place."""
    started = False
    try:
        prepare(model, config)
        started = True
        calibrate(model, batches)
        convert(model)
    except Exception:
        session = _session(model)
        if (
            started
            and session is not None
            and session.quantizer.state is not QuantizationState.CONVERTED
        ):
            delattr(model, _SESSION_ATTRIBUTE)
        raise
    return model


def load_quantized_state_dict(
    model: nn.Module,
    config: MinMaxConfig,
    state_dict: Mapping[str, Tensor],
) -> nn.Module:
    """Rebuild a converted MinMax model and strictly load frozen qparams."""
    if _session(model) is not None:
        raise RuntimeError("model already has an active quantization lifecycle")
    quantizer = MinMaxQuantizer(config)
    quantizer.restore(model, state_dict)
    setattr(model, _SESSION_ATTRIBUTE, _QuantizationSession(quantizer))
    return model


def quantization_state(model: nn.Module) -> QuantizationState:
    """Return model quantization lifecycle state."""
    session = _session(model)
    return QuantizationState.UNPREPARED if session is None else session.quantizer.state


def _session(model: nn.Module) -> _QuantizationSession | None:
    candidate = getattr(model, _SESSION_ATTRIBUTE, None)
    if candidate is None:
        return None
    if not isinstance(candidate, _QuantizationSession):
        raise RuntimeError(f"reserved model attribute {_SESSION_ATTRIBUTE!r} is already in use")
    return candidate


def _required_session(model: nn.Module) -> _QuantizationSession:
    session = _session(model)
    if session is None:
        raise RuntimeError("model has not been prepared for quantization")
    return session


__all__ = [
    "calibrate",
    "convert",
    "load_quantized_state_dict",
    "prepare",
    "quantization_state",
    "quantize",
]
