"""Frozen MinMax quantization parameter specifications and bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .config import MinMaxConfig
from .observer import MinMaxObserver, QuantizationParameters

_QParamPrefix = Literal["weight", "activation"]
_Granularity = Literal["per_tensor", "per_channel", "per_token"]


@dataclass(frozen=True, slots=True)
class _QParamBinding:
    """Own normalized frozen qparams for one quantized tensor role."""

    prefix: _QParamPrefix
    axis: int | None
    scale: Tensor
    zero_point: Tensor | None
    qdq_scale: Tensor
    qdq_zero_point: Tensor | None

    @classmethod
    def from_parameters(
        cls,
        prefix: _QParamPrefix,
        qparams: QuantizationParameters,
        reference: Tensor,
        axis: int | None,
    ) -> _QParamBinding:
        """Normalize public qparams into one frozen internal binding."""
        scale = (
            qparams.scale.detach()
            .clone()
            .to(device=reference.device, dtype=torch.float32)
        )
        zero_point = qparams.zero_point
        normalized_zero_point = (
            None
            if zero_point is None
            else zero_point.detach()
            .clone()
            .to(device=reference.device, dtype=torch.int8)
        )
        qdq_scale = scale.to(dtype=reference.dtype)
        qdq_zero_point = normalized_zero_point
        if axis is not None:
            qdq_scale = qdq_scale.reshape(-1)
            if qdq_zero_point is not None:
                qdq_zero_point = qdq_zero_point.reshape(-1)
        return cls(
            prefix=prefix,
            axis=axis,
            scale=scale,
            zero_point=normalized_zero_point,
            qdq_scale=qdq_scale,
            qdq_zero_point=qdq_zero_point,
        )

    @property
    def parameters(self) -> QuantizationParameters:
        """Expose normalized qparams for the public Linear constructor."""
        return QuantizationParameters(self.scale, self.zero_point)

    def install(self, module: nn.Module) -> None:
        """Install flat persistent and derived QDQ buffers on a module."""
        module.register_buffer(f"{self.prefix}_scale", self.scale)
        module.register_buffer(f"{self.prefix}_zero_point", self.zero_point)
        module.register_buffer(f"{self.prefix}_qdq_scale", self.qdq_scale, persistent=False)
        module.register_buffer(
            f"{self.prefix}_qdq_zero_point",
            self.qdq_zero_point,
            persistent=False,
        )

    def validate_activation_input(self, inputs: Tensor) -> None:
        """Reject runtime shapes incompatible with frozen per-token qparams."""
        if self.prefix != "activation" or self.axis is None:
            return
        expected_rank = self.scale.ndim
        if inputs.ndim != expected_rank:
            raise ValueError(
                f"per-token activation rank changed from {expected_rank} to {inputs.ndim}"
            )
        axis = self.axis % expected_rank
        expected_length = self.scale.shape[axis]
        if inputs.shape[axis] != expected_length:
            raise ValueError(
                "per-token activation length changed "
                f"from {expected_length} to {inputs.shape[axis]}"
            )


@dataclass(frozen=True, slots=True)
class _QParamSpec:
    """Interpret configuration and freeze qparams for one tensor role."""

    prefix: _QParamPrefix
    enabled: bool
    symmetric: bool
    granularity: _Granularity
    axis: int | None

    @classmethod
    def from_config(cls, config: MinMaxConfig) -> tuple[_QParamSpec, _QParamSpec]:
        """Build weight and activation specifications from one configuration."""
        weight_axis = 0 if config.weight_granularity == "per_channel" else None
        activation_axis = -2 if config.activation_granularity == "per_token" else None
        return (
            cls(
                prefix="weight",
                enabled=config.weight,
                symmetric=config.weight_symmetric,
                granularity=config.weight_granularity,
                axis=weight_axis,
            ),
            cls(
                prefix="activation",
                enabled=config.activation,
                symmetric=config.activation_symmetric,
                granularity=config.activation_granularity,
                axis=activation_axis,
            ),
        )

    def create_observer(self) -> MinMaxObserver | None:
        """Create an observer when this tensor role is enabled."""
        return MinMaxObserver(axis=self.axis) if self.enabled else None

    def freeze(
        self,
        observer: MinMaxObserver,
        module: nn.Linear,
    ) -> _QParamBinding:
        """Freeze observed qparams and bind them to a target Linear."""
        qparams = observer.calculate_qparams(symmetric=self.symmetric)
        return self._bind(qparams, module)

    def from_checkpoint(
        self,
        state_dict: Mapping[str, Tensor],
        names: Sequence[str],
        module: nn.Linear,
    ) -> _QParamBinding | None:
        """Validate alias checkpoint qparams and bind them to a target Linear."""
        if not self.enabled:
            return None
        scales = [state_dict[f"{name}.{self.prefix}_scale"] for name in names]
        zero_points = (
            None
            if self.symmetric
            else [state_dict[f"{name}.{self.prefix}_zero_point"] for name in names]
        )
        if any(not torch.equal(scales[0], scale) for scale in scales[1:]):
            raise ValueError(
                f"shared Linear aliases have inconsistent {self.prefix} scales"
            )
        if zero_points is not None and any(
            not torch.equal(zero_points[0], zero_point) for zero_point in zero_points[1:]
        ):
            raise ValueError(
                f"shared Linear aliases have inconsistent {self.prefix} zero-points"
            )
        scale = scales[0]
        if not scale.is_floating_point():
            raise TypeError(f"{self.prefix} scale must use a floating-point dtype")
        zero_point = None if zero_points is None else zero_points[0]
        if zero_point is not None:
            if zero_point.dtype is not torch.int8:
                raise TypeError(f"{self.prefix} zero-point must use torch.int8")
            if zero_point.shape != scale.shape:
                raise ValueError(
                    f"{self.prefix} zero-point shape must match scale shape"
                )
        return self._bind(QuantizationParameters(scale, zero_point), module)

    def expected_checkpoint_keys(self, name: str) -> tuple[str, ...]:
        """Return persistent qparam keys expected for one module alias."""
        if not self.enabled:
            return ()
        scale_key = f"{name}.{self.prefix}_scale"
        if self.symmetric:
            return (scale_key,)
        return (scale_key, f"{name}.{self.prefix}_zero_point")

    def _bind(
        self,
        qparams: QuantizationParameters,
        module: nn.Linear,
    ) -> _QParamBinding:
        scale = qparams.scale.detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise ValueError(
                f"{self.prefix} scale must be finite and strictly positive"
            )
        self._validate_shape(scale, module)
        return _QParamBinding.from_parameters(
            self.prefix,
            qparams,
            module.weight,
            self.axis,
        )

    def _validate_shape(self, scale: Tensor, module: nn.Linear) -> None:
        shape = scale.shape
        if self.prefix == "weight":
            expected = (
                (module.out_features, 1) if self.granularity == "per_channel" else ()
            )
            if shape != expected:
                raise ValueError(
                    f"weight scale shape {tuple(shape)} does not match {expected}"
                )
            return
        if self.granularity == "per_tensor":
            if shape:
                raise ValueError("per-tensor activation scale must be scalar")
            return
        if len(shape) < 2 or shape[-2] < 1:
            raise ValueError(
                "per-token activation scale must encode rank and token length"
            )
        if any(size != 1 for index, size in enumerate(shape) if index != len(shape) - 2):
            raise ValueError(
                "per-token activation scale must vary only along token axis"
            )
