"""Quantization plan materialization for one graph target."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor
from torch.fx import GraphModule

from ..errors import QuantizationConfigError
from ..graph.metadata import QuantizedTarget
from .algorithms.gptq import GptqFallbackError, gptq_weight_quantize
from .algorithms.math import quantize
from .calibration import CalibrationArtifacts
from .config import ActivationSpec
from .planning import QuantizedTensor, TargetPlan


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Materialized target plus auxiliary deployment metadata."""

    target: QuantizedTarget
    activation_qparams: dict[str, Any] | None
    integer_sha256: str | None


@dataclass(frozen=True, slots=True)
class MaterializationContext:
    """Candidate-local inputs captured before materialization writes.

    The parameter mapping is an initial snapshot used once per planned alias
    group. It does not expose parameters registered during materialization.
    """

    candidate: GraphModule
    parameters: Mapping[str, Tensor]

    @classmethod
    def capture(cls, candidate: GraphModule) -> MaterializationContext:
        """Capture the candidate parameter registry before the first write."""
        parameters = dict(
            candidate.named_parameters(remove_duplicate=False)
        )
        return cls(candidate, MappingProxyType(parameters))

    def parameter(self, target: TargetPlan) -> Tensor | None:
        """Resolve one initial plan target without refreshing the snapshot."""
        if target.parameter_name is None:
            return None
        try:
            return self.parameters[target.parameter_name]
        except KeyError as error:
            raise QuantizationConfigError(
                f"Target parameter disappeared: {target.parameter_name}"
            ) from error


def _required_sample(
    calibration: CalibrationArtifacts,
    fqn: str,
) -> Tensor:
    try:
        return calibration.samples(fqn)
    except KeyError as error:
        raise QuantizationConfigError(
            f"No activation calibration captured for {fqn!r}"
        ) from error


def _require_same_device(
    parameter: Tensor,
    sample: Tensor,
    *,
    algorithm: str,
    fqn: str,
) -> None:
    if sample.device != parameter.device:
        raise QuantizationConfigError(
            f"{algorithm} device mismatch for {fqn!r}: parameter is on "
            f"{parameter.device}, calibration is on {sample.device}"
        )


def _required_qparams(
    calibration: CalibrationArtifacts,
    target: TargetPlan,
    spec: ActivationSpec,
) -> tuple[Tensor, Tensor]:
    try:
        return calibration.qparams(target.fqn, spec)
    except KeyError as error:
        raise QuantizationConfigError(
            f"No activation calibration captured for {target.fqn!r}"
        ) from error


def _materialize_weight(
    context: MaterializationContext,
    parameter: Tensor,
    target: TargetPlan,
    calibration: CalibrationArtifacts,
    sample: Tensor | None = None,
) -> tuple[QuantizedTensor, str | None, tuple[float, ...] | None]:
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
        if (
            target.target_type == "moe"
            and target.parameter_name is not None
            and target.parameter_name.endswith(".expert_weights")
        ):
            raise QuantizationConfigError(
                "GPTQ does not support packed MoeExpert weights"
            )
        samples = (
            _required_sample(calibration, target.fqn)
            if sample is None
            else sample
        )
        _require_same_device(
            parameter,
            samples,
            algorithm="GPTQ",
            fqn=target.fqn,
        )
        try:
            result = gptq_weight_quantize(
                parameter,
                samples,
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
    intermediate_scales: tuple[float, ...] | None = None
    if (
        target.target_type == "moe"
        and target.parameter_name is not None
        and target.parameter_name.endswith(".expert_weights")
    ):
        module_name = target.parameter_name.rsplit(".", 1)[0]
        module = context.candidate.get_submodule(module_name)
        expert_count = int(parameter.shape[0])
        samples = (
            _required_sample(calibration, target.fqn)
            if sample is None
            else sample
        )
        _require_same_device(
            parameter,
            samples,
            algorithm="MoeExpert",
            fqn=target.fqn,
        )
        samples = samples.float()
        hidden_size = int(samples.shape[-1])
        intermediate_size = int(
            parameter.shape[1] // (3 * hidden_size)
        )
        projection_size = hidden_size * intermediate_size
        calculated_scales: list[float] = []
        for expert_id in range(expert_count):
            packed = parameter[expert_id].float()
            gate = packed[:projection_size].reshape(
                intermediate_size,
                hidden_size,
            )
            up = packed[
                projection_size : 2 * projection_size
            ].reshape(intermediate_size, hidden_size)
            intermediate = torch.nn.functional.silu(
                samples @ gate.t()
            ) * (samples @ up.t())
            maximum = float(intermediate.abs().amax().cpu())
            calculated_scales.append(max(maximum / 127.0, 1e-12))
        intermediate_scales = tuple(calculated_scales)
        scales = result.scale.reshape(-1)
        if scales.numel() == 1:
            scales = scales.repeat(expert_count * 3)
        if scales.numel() != expert_count * 3:
            raise QuantizationConfigError(
                "Packed MoeExpert requires one scale per projection"
            )
        module.register_parameter(
            "expert_weights",
            torch.nn.Parameter(
                result.values.detach().clone(),
                requires_grad=False,
            ),
        )
        module.register_parameter(
            "quant_scales",
            torch.nn.Parameter(
                scales.reshape(expert_count, 3).float(),
                requires_grad=False,
            ),
        )
        scale_name = f"{module_name}.quant_scales"
        for node in context.candidate.graph.nodes:
            if (
                node.op != "call_function"
                or node.target
                != torch.ops.mdc_llm_deploy.moe_expert.default
                or len(node.args) < 4
            ):
                continue
            packed = node.args[3]
            if (
                getattr(packed, "op", None) != "get_attr"
                or getattr(packed, "target", None)
                != target.parameter_name
            ):
                continue
            with context.candidate.graph.inserting_before(node):
                scale_node = context.candidate.graph.get_attr(scale_name)
            arguments = list(node.args)
            arguments.extend([None] * (6 - len(arguments)))
            arguments[4] = scale_node
            node.args = tuple(arguments)
            break
        else:
            raise QuantizationConfigError(
                "Packed MoeExpert node disappeared during materialization"
            )
    else:
        with torch.no_grad():
            parameter.copy_(result.dequantized)
    return result, fallback_reason, intermediate_scales


def _activation_metadata(
    target: TargetPlan,
    calibration: CalibrationArtifacts,
) -> dict[str, Any] | None:
    if target.activation is None:
        return None
    scale, zero_point = _required_qparams(
        calibration,
        target,
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
    context: MaterializationContext,
    target: TargetPlan,
    calibration: CalibrationArtifacts,
    *,
    weight_sample: Tensor | None = None,
) -> MaterializationResult:
    """Apply one target plan and return its immutable contract data."""
    parameter = context.parameter(target)
    spec = target.weight or target.activation
    if spec is None:
        raise QuantizationConfigError(
            f"Target {target.fqn!r} has no tensor spec"
        )
    fallback_reason: str | None = None
    result: QuantizedTensor | None = None
    intermediate_scales: tuple[float, ...] | None = None
    if parameter is not None and target.weight is not None:
        result, fallback_reason, intermediate_scales = _materialize_weight(
            context,
            parameter,
            target,
            calibration,
            weight_sample,
        )
        scale = result.scale
        zero_point = result.zero_point
    else:
        if target.activation is None:
            raise QuantizationConfigError(
                f"Target {target.fqn!r} has no activation spec"
            )
        scale, zero_point = _required_qparams(
            calibration,
            target,
            target.activation,
        )
    activation_qparams = _activation_metadata(
        target,
        calibration,
    )
    if intermediate_scales is not None:
        if activation_qparams is None:
            raise QuantizationConfigError(
                "Packed MoeExpert requires activation quantization"
            )
        activation_qparams["intermediate_scale"] = list(
            intermediate_scales
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


def materialize_alias_group(
    context: MaterializationContext,
    targets: tuple[TargetPlan, ...],
    calibration: CalibrationArtifacts,
) -> tuple[MaterializationResult, ...]:
    """Materialize one initial alias group without refreshing its parameter."""
    if not targets:
        return ()
    representative = targets[0]
    weight_sample: Tensor | None = None
    if (
        len(targets) > 1
        and representative.parameter_name is not None
        and (
            representative.algorithm == "gptq"
            or (
                representative.target_type == "moe"
                and representative.parameter_name.endswith(
                    ".expert_weights"
                )
            )
        )
    ):
        unique_samples: dict[object, Tensor] = {}
        for target in targets:
            try:
                items = calibration.sample_items(target.fqn)
            except KeyError as error:
                raise QuantizationConfigError(
                    f"No activation calibration captured for {target.fqn!r}"
                ) from error
            for boundary, sample in items:
                unique_samples.setdefault(boundary, sample)
        samples = tuple(unique_samples.values())
        devices = {sample.device for sample in samples}
        if len(devices) != 1:
            raise QuantizationConfigError(
                "Aliased calibration targets span devices: "
                f"{sorted(str(device) for device in devices)}"
            )
        weight_sample = samples[0] if len(samples) == 1 else torch.cat(samples)
    first = materialize_target(
        context,
        representative,
        calibration,
        weight_sample=weight_sample,
    )
    results = [first]
    for target in targets[1:]:
        activation_qparams = _activation_metadata(
            target,
            calibration,
        )
        if (
            activation_qparams is not None
            and first.activation_qparams is not None
            and "intermediate_scale" in first.activation_qparams
        ):
            activation_qparams["intermediate_scale"] = (
                first.activation_qparams["intermediate_scale"]
            )
        results.append(
            MaterializationResult(
                target=replace(first.target, fqn=target.fqn),
                activation_qparams=activation_qparams,
                integer_sha256=first.integer_sha256,
            )
        )
    return tuple(results)
