"""Quantization plan materialization for one graph target."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.fx import GraphModule

from ..config import ActivationSpec, WeightSpec
from ..errors import QuantizationConfigError
from ..graph_types import QuantizedTarget
from .gptq import GptqFallbackError, gptq_weight_quantize
from .math import (
    calculate_qparams,
    quantize,
)
from .planner import TargetPlan
from .types import QuantizedTensor


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Materialized target plus auxiliary deployment metadata."""

    target: QuantizedTarget
    activation_qparams: dict[str, Any] | None
    integer_sha256: str | None


def _parameter(
    candidate: GraphModule,
    target: TargetPlan,
) -> Tensor | None:
    if target.parameter_name is None:
        return None
    parameters = dict(candidate.named_parameters())
    try:
        return parameters[target.parameter_name]
    except KeyError as error:
        raise QuantizationConfigError(
            f"Target parameter disappeared: {target.parameter_name}"
        ) from error


def _required_sample(
    calibration: Mapping[str, Tensor],
    fqn: str,
) -> Tensor:
    try:
        return calibration[fqn]
    except KeyError as error:
        raise QuantizationConfigError(
            f"No activation calibration captured for {fqn!r}"
        ) from error


def _activation_parameters(
    sample: Tensor,
    spec: ActivationSpec | WeightSpec,
) -> tuple[Tensor, Tensor]:
    axis = 0 if spec.granularity == "per_token" else None
    return calculate_qparams(
        sample,
        bits=spec.bits,
        symmetric=spec.symmetric,
        axis=axis,
    )


def _materialize_weight(
    parameter: Tensor,
    target: TargetPlan,
    calibration: Mapping[str, Tensor],
) -> tuple[QuantizedTensor, str | None]:
    if target.weight is None:
        raise QuantizationConfigError(
            f"Target {target.fqn!r} has no weight spec"
        )
    axis = (
        0
        if target.weight.granularity == "per_channel"
        else None
    )
    fallback_reason: str | None = None
    if target.algorithm == "gptq":
        samples = _required_sample(calibration, target.fqn)
        try:
            result = gptq_weight_quantize(
                parameter,
                samples.to(parameter.device),
                bits=target.weight.bits,
                percdamp=target.percdamp,
                actorder=target.actorder,
                block_size=target.block_size,
                per_channel=(
                    target.weight.granularity == "per_channel"
                ),
            )
        except GptqFallbackError as error:
            result = quantize(
                parameter,
                bits=target.weight.bits,
                symmetric=True,
                axis=axis,
            )
            fallback_reason = error.reason
    else:
        result = quantize(
            parameter,
            bits=target.weight.bits,
            symmetric=target.weight.symmetric,
            axis=axis,
        )
    with torch.no_grad():
        parameter.copy_(result.dequantized)
    return result, fallback_reason


def _activation_metadata(
    target: TargetPlan,
    calibration: Mapping[str, Tensor],
) -> dict[str, Any] | None:
    if target.activation is None:
        return None
    scale, zero_point = _activation_parameters(
        _required_sample(calibration, target.fqn),
        target.activation,
    )
    return {
        "bits": target.activation.bits,
        "granularity": target.activation.granularity,
        "mode": target.activation.mode,
        "symmetric": target.activation.symmetric,
        "scale": [
            float(item) for item in scale.reshape(-1).cpu()
        ],
        "zero_point": [
            int(item) for item in zero_point.reshape(-1).cpu()
        ],
    }


def _integer_sha256(
    result: QuantizedTensor | None,
) -> str | None:
    if result is None:
        return None
    payload = (
        result.values.detach().cpu().contiguous().numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def materialize_target(
    candidate: GraphModule,
    target: TargetPlan,
    calibration: Mapping[str, Tensor],
) -> MaterializationResult:
    """Apply one target plan and return its immutable contract data."""
    parameter = _parameter(candidate, target)
    spec = target.weight or target.activation
    if spec is None:
        raise QuantizationConfigError(
            f"Target {target.fqn!r} has no tensor spec"
        )
    fallback_reason: str | None = None
    result: QuantizedTensor | None = None
    if parameter is not None and target.weight is not None:
        result, fallback_reason = _materialize_weight(
            parameter,
            target,
            calibration,
        )
        scale = result.scale
        zero_point = result.zero_point
    else:
        scale, zero_point = _activation_parameters(
            _required_sample(calibration, target.fqn),
            spec,
        )
    activation_qparams = _activation_metadata(
        target,
        calibration,
    )
    integer_hash = _integer_sha256(result)
    materialized = QuantizedTarget(
        fqn=target.fqn,
        target_type=target.target_type,
        algorithm=target.algorithm,
        bits=spec.bits,
        granularity=spec.granularity,
        symmetric=spec.symmetric,
        scale=tuple(float(item) for item in scale.reshape(-1).cpu()),
        zero_point=tuple(
            int(item) for item in zero_point.reshape(-1).cpu()
        ),
        fallback_reason=fallback_reason,
    )
    return MaterializationResult(
        target=materialized,
        activation_qparams=activation_qparams,
        integer_sha256=integer_hash,
    )
