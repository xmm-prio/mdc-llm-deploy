"""MinMax fake-quantized linear module."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ..qdq import qdq
from .observer import MinMaxObserver, QuantizationParameters
from .qparams import _QParamBinding, _QParamPrefix


def fake_quantize_int8(
    value: Tensor,
    scale: Tensor,
    zero_point: Tensor | None,
) -> Tensor:
    """Apply ties-to-even INT8 fake quantization with broadcast qparams."""
    typed_scale = scale.to(device=value.device, dtype=value.dtype)
    typed_scale = typed_scale.clamp_min(torch.finfo(value.dtype).tiny)
    if not bool(torch.isfinite(typed_scale).all()) or not bool((typed_scale > 0).all()):
        raise ValueError("typed quantization scale must be finite and strictly positive")
    if zero_point is None:
        quantized = torch.round(value / typed_scale).clamp(-128, 127)
        return quantized * typed_scale
    typed_zero_point = zero_point.to(device=value.device, dtype=value.dtype)
    quantized = (torch.round(value / typed_scale) + typed_zero_point).clamp(-128, 127)
    return (quantized - typed_zero_point) * typed_scale


def symmetric_per_tensor_scale(weight: Tensor) -> Tensor:
    """Compute a finite positive INT8 symmetric per-tensor scale."""
    observer = MinMaxObserver()
    observer.observe(weight)
    return observer.calculate_qparams(symmetric=True).scale


def fake_quantize_symmetric_per_tensor(weight: Tensor, scale: Tensor) -> Tensor:
    """Apply ties-to-even INT8 symmetric per-tensor fake quantization."""
    return fake_quantize_int8(weight, scale, None)


class MinMaxLinear(nn.Module):
    """Linear layer with frozen MinMax weight and activation fake quantization."""

    in_features: int
    out_features: int
    weight: nn.Parameter
    bias: nn.Parameter | None
    weight_scale: Tensor | None
    weight_zero_point: Tensor | None
    activation_scale: Tensor | None
    activation_zero_point: Tensor | None
    weight_qdq_scale: Tensor | None
    weight_qdq_zero_point: Tensor | None
    activation_qdq_scale: Tensor | None
    activation_qdq_zero_point: Tensor | None

    def __init__(
        self,
        source: nn.Linear,
        *,
        weight_qparams: QuantizationParameters | None,
        activation_qparams: QuantizationParameters | None,
        weight_axis: int | None,
        activation_axis: int | None,
    ) -> None:
        super().__init__()
        weight_binding = self._bind_parameters(
            "weight",
            weight_qparams,
            source.weight,
            weight_axis,
        )
        activation_binding = self._bind_parameters(
            "activation",
            activation_qparams,
            source.weight,
            activation_axis,
        )
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight_axis = weight_axis
        self.activation_axis = activation_axis
        self._activation_binding = activation_binding
        self.register_parameter("weight", cast(nn.Parameter, source.weight))
        self.register_parameter("bias", source.bias)
        self._install_binding("weight", weight_binding)
        self._install_binding("activation", activation_binding)
        self.train(source.training)

    @staticmethod
    def _bind_parameters(
        prefix: _QParamPrefix,
        qparams: QuantizationParameters | None,
        reference: Tensor,
        axis: int | None,
    ) -> _QParamBinding | None:
        if qparams is None:
            return None
        return _QParamBinding.from_parameters(prefix, qparams, reference, axis)

    def _install_binding(
        self,
        prefix: str,
        binding: _QParamBinding | None,
    ) -> None:
        if binding is not None:
            binding.install(self)
            return
        self.register_buffer(f"{prefix}_scale", None)
        self.register_buffer(f"{prefix}_zero_point", None)
        self.register_buffer(f"{prefix}_qdq_scale", None, persistent=False)
        self.register_buffer(f"{prefix}_qdq_zero_point", None, persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Run linear projection with enabled fake quantizers."""
        quantized_inputs = inputs
        if self.activation_scale is not None:
            if self._activation_binding is None:
                raise RuntimeError("activation quantization binding is missing")
            self._activation_binding.validate_activation_input(inputs)
            if self.activation_qdq_scale is None:
                raise RuntimeError("activation QDQ scale is missing")
            quantized_inputs = qdq(
                inputs,
                self.activation_qdq_scale,
                self.activation_qdq_zero_point,
                axis=self.activation_axis,
            )
        quantized_weight: Tensor = self.weight
        if self.weight_scale is not None:
            if self.weight_qdq_scale is None:
                raise RuntimeError("weight QDQ scale is missing")
            quantized_weight = qdq(
                self.weight,
                self.weight_qdq_scale,
                self.weight_qdq_zero_point,
                axis=self.weight_axis,
            )
        return functional.linear(quantized_inputs, quantized_weight, self.bias)

    def extra_repr(self) -> str:
        """Return module representation matching torch linear fields."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, weight_axis={self.weight_axis}, "
            f"activation_axis={self.activation_axis}"
        )


__all__ = [
    "MinMaxLinear",
    "fake_quantize_int8",
    "fake_quantize_symmetric_per_tensor",
    "symmetric_per_tensor_scale",
]
