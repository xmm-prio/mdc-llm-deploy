"""Lazy registration for the internal QDQ custom operator."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from importlib import metadata
from typing import cast

import torch
from torch import Tensor

_EXPECTED_TORCH_VERSION = "2.12.0"
_OPERATOR_NAME = "mdc_llm_deploy::qdq"
_INT8_ONNX_DTYPE = 3
_REGISTRATION_LOCK = threading.Lock()
_operator: Callable[[Tensor, Tensor, Tensor | None, int | None], Tensor] | None = None
_OnnxAttribute = (
    int
    | float
    | str
    | bool
    | Sequence[int]
    | Sequence[float]
    | Sequence[str]
    | Sequence[bool]
)


def require_supported_torch_version() -> None:
    """Require the exact Torch release validated by the QDQ representation."""
    installed = metadata.version("torch")
    if installed != _EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"QDQ export requires torch=={_EXPECTED_TORCH_VERSION}; found torch=={installed}"
        )


def _reshape_parameter(parameter: Tensor, inputs: Tensor, axis: int | None) -> Tensor:
    if axis is None or parameter.ndim != 1:
        return parameter
    normalized_axis = axis if axis >= 0 else inputs.ndim + axis
    shape = [1] * inputs.ndim
    shape[normalized_axis] = parameter.shape[0]
    return parameter.reshape(shape)


def _eager_qdq(
    inputs: Tensor,
    scale: Tensor,
    zero_point: Tensor | None,
    axis: int | None,
) -> Tensor:
    typed_scale = _reshape_parameter(
        scale.to(device=inputs.device, dtype=inputs.dtype), inputs, axis
    )
    if zero_point is None:
        typed_zero_point: Tensor | int = 0
    else:
        typed_zero_point = _reshape_parameter(
            zero_point.to(device=inputs.device, dtype=inputs.dtype), inputs, axis
        )
    quantized = torch.round(inputs / typed_scale + typed_zero_point).clamp(-128, 127)
    return (quantized - typed_zero_point) * typed_scale


def _fake_qdq(
    inputs: Tensor,
    scale: Tensor,
    zero_point: Tensor | None,
    axis: int | None,
) -> Tensor:
    del scale, zero_point, axis
    return torch.empty_like(inputs)


def _decompose_qdq(
    inputs: Tensor,
    scale: Tensor,
    zero_point: Tensor | None,
    axis: int | None,
) -> Tensor:
    quantize_attributes: dict[str, _OnnxAttribute] = {}
    dequantize_attributes: dict[str, _OnnxAttribute] = {}
    if axis is not None:
        quantize_attributes["axis"] = axis
        dequantize_attributes["axis"] = axis
    if zero_point is None:
        quantize_attributes["output_dtype"] = _INT8_ONNX_DTYPE

    quantized = torch.onnx.ops.symbolic(
        "QuantizeLinear",
        (inputs, scale, zero_point),
        quantize_attributes,
        dtype=torch.int8,
        shape=inputs.shape,
        version=21,
    )
    return torch.onnx.ops.symbolic(
        "DequantizeLinear",
        (quantized, scale, zero_point),
        dequantize_attributes,
        dtype=inputs.dtype,
        shape=inputs.shape,
        version=21,
    )


def _registered_operator() -> Callable[[Tensor, Tensor, Tensor | None, int | None], Tensor] | None:
    return _operator


def register_qdq_operator() -> Callable[[Tensor, Tensor, Tensor | None, int | None], Tensor]:
    """Register and return the internal QDQ operator exactly once."""
    global _operator
    operator = _registered_operator()
    if operator is not None:
        return operator

    with _REGISTRATION_LOCK:
        operator = _registered_operator()
        if operator is not None:
            return operator

        require_supported_torch_version()
        if _OPERATOR_NAME in torch._C._dispatch_get_all_op_names():
            raise RuntimeError(f"Custom operator registration conflict: {_OPERATOR_NAME} already exists")
        custom_operator = torch.library.custom_op(
            _OPERATOR_NAME,
            _eager_qdq,
            mutates_args=(),
        )
        torch.library.register_fake(custom_operator)(_fake_qdq)
        overload = torch.ops.mdc_llm_deploy.qdq.default
        torch._decomp.register_decomposition(overload)(_decompose_qdq)
        _operator = cast(
            Callable[[Tensor, Tensor, Tensor | None, int | None], Tensor],
            overload,
        )
        return _operator


__all__ = ["register_qdq_operator", "require_supported_torch_version"]
