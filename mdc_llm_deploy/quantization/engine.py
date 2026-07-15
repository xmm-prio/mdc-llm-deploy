"""Transactional calibration and quantization engine."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor
from torch.fx import GraphModule, Interpreter, Node

from ..config import QuantizationConfig
from ..errors import GraphStateError, QuantizationConfigError
from ..graph import GraphStage, QuantizedTarget, metadata, set_metadata, transactional_update
from .math import QuantizedTensor, calculate_qparams, gptq_weight_quantize, quantize
from .planner import TargetPlan, plan_quantization


class _CalibrationInterpreter(Interpreter):
    """Capture actual linear inputs and attention edges from one FX execution."""

    def __init__(self, graph: GraphModule, attention_fqns: tuple[str, ...]) -> None:
        super().__init__(graph, garbage_collect_values=False)
        self.attention_fqns = attention_fqns
        self.samples: dict[str, list[Tensor]] = {}

    def _record(self, name: str, value: Any) -> None:
        if isinstance(value, Tensor) and value.is_floating_point():
            self.samples.setdefault(name, []).append(value.detach())

    def _attention_fqn(self, node: Node) -> str | None:
        stack = node.meta.get("nn_module_stack")
        if not isinstance(stack, Mapping):
            return None
        module_fqns = tuple(
            value[0]
            for value in stack.values()
            if isinstance(value, tuple) and value and isinstance(value[0], str)
        )
        return next(
            (
                fqn
                for fqn in self.attention_fqns
                if fqn in module_fqns
                or any(module.startswith(f"{fqn}.") for module in module_fqns)
            ),
            None,
        )

    def run_node(self, node: Node) -> Any:
        """Execute one node and retain only quantization boundary tensors."""
        args, _ = self.fetch_args_kwargs_from_env(node)
        result = super().run_node(node)
        if node.op != "call_function":
            return result
        if node.target == torch.ops.aten.linear.default and len(node.args) >= 2:
            weight_node = node.args[1]
            if isinstance(weight_node, Node) and weight_node.op == "get_attr":
                fqn = str(weight_node.target).removesuffix(".weight")
                self._record(fqn, args[0])
                edge = {
                    "q_proj": "query",
                    "k_proj": "key",
                    "v_proj": "value",
                }.get(fqn.rsplit(".", 1)[-1])
                attention_fqn = self._attention_fqn(node)
                if edge is not None and attention_fqn is not None:
                    self._record(f"{attention_fqn}.{edge}", result)
        attention_fqn = self._attention_fqn(node)
        if (
            attention_fqn is not None
            and node.target == torch.ops.aten.mul.Tensor
            and isinstance(result, Tensor)
            and result.ndim == 4
            and result.shape[-2] == result.shape[-1]
            and any(isinstance(argument, (float, int)) for argument in args)
        ):
            self._record(f"{attention_fqn}.score", result)
        return result


def _calibration_samples(
    graph: GraphModule,
    dataloader: Iterable[Mapping[str, Tensor]],
) -> dict[str, Tensor]:
    expected = tuple(item.name for item in metadata(graph).input_abi)
    attention_fqns = tuple(
        boundary.fqn
        for boundary in metadata(graph).boundaries
        if boundary.kind == "attention"
    )
    captured: dict[str, list[Tensor]] = {}
    observed = 0
    for batch in dataloader:
        if not isinstance(batch, Mapping):
            raise TypeError("calibration batches must be mappings")
        if tuple(batch) != expected:
            raise QuantizationConfigError(
                f"Calibration keys must be {expected}, got {tuple(batch)}"
            )
        for name, tensor in batch.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Calibration value {name!r} must be a tensor")
            expected_shape = next(item.shape for item in metadata(graph).input_abi if item.name == name)
            if tuple(tensor.shape) != expected_shape:
                raise QuantizationConfigError(
                    f"Calibration shape for {name!r} must be {expected_shape}"
                )
            source = tensor.detach().float()
            if not torch.isfinite(source).all():
                raise QuantizationConfigError(
                    f"Calibration value {name!r} contains NaN or Inf"
                )
        recorder = _CalibrationInterpreter(graph, attention_fqns)
        try:
            recorder.run(*(batch[name] for name in expected))
        except Exception as error:
            raise QuantizationConfigError(
                f"Calibration graph execution failed: {error}"
            ) from error
        for fqn, values in recorder.samples.items():
            captured.setdefault(fqn, []).extend(values)
        observed += 1
    if observed == 0:
        raise QuantizationConfigError("Calibration dataloader must yield at least one batch")
    return {
        fqn: torch.cat(tuple(value.reshape(-1, value.shape[-1]) for value in values))
        for fqn, values in captured.items()
    }


def _parameter(candidate: GraphModule, target: TargetPlan) -> Tensor | None:
    if target.parameter_name is None:
        return None
    parameters = dict(candidate.named_parameters())
    try:
        return parameters[target.parameter_name]
    except KeyError as error:
        raise QuantizationConfigError(
            f"Target parameter disappeared: {target.parameter_name}"
        ) from error


def _materialize_target(
    candidate: GraphModule,
    target: TargetPlan,
    calibration: Mapping[str, Tensor],
) -> tuple[QuantizedTarget, dict[str, Any] | None, str | None]:
    parameter = _parameter(candidate, target)
    spec = target.weight or target.activation
    if spec is None:
        raise QuantizationConfigError(f"Target {target.fqn!r} has no tensor spec")
    fallback_reason: str | None = None
    result: QuantizedTensor | None = None
    if parameter is not None and target.weight is not None:
        axis = 0 if target.weight.granularity == "per_channel" else None
        if target.algorithm == "gptq":
            samples = calibration.get(target.fqn)
            if samples is None:
                raise QuantizationConfigError(
                    f"No activation calibration captured for {target.fqn!r}"
                )
            try:
                result = gptq_weight_quantize(
                    parameter,
                    samples.to(parameter.device),
                    bits=target.weight.bits,
                    percdamp=target.percdamp,
                    actorder=target.actorder,
                    block_size=target.block_size,
                    per_channel=target.weight.granularity == "per_channel",
                )
            except (RuntimeError, ValueError) as error:
                result = quantize(
                    parameter,
                    bits=target.weight.bits,
                    symmetric=True,
                    axis=axis,
                )
                fallback_reason = f"cholesky_failed:{type(error).__name__}"
        else:
            result = quantize(
                parameter,
                bits=target.weight.bits,
                symmetric=target.weight.symmetric,
                axis=axis,
            )
        with torch.no_grad():
            parameter.copy_(result.dequantized)
        scale = result.scale
        zero_point = result.zero_point
    else:
        activation_sample = calibration.get(target.fqn)
        if activation_sample is None:
            raise QuantizationConfigError(
                f"No activation calibration captured for {target.fqn!r}"
            )
        axis = 0 if spec.granularity == "per_token" else None
        scale, zero_point = calculate_qparams(
            activation_sample,
            bits=spec.bits,
            symmetric=spec.symmetric,
            axis=axis,
        )
    activation_qparams: dict[str, Any] | None = None
    if target.activation is not None:
        activation_sample = calibration.get(target.fqn)
        if activation_sample is None:
            raise QuantizationConfigError(
                f"No activation calibration captured for {target.fqn!r}"
            )
        activation_axis = 0 if target.activation.granularity == "per_token" else None
        activation_scale, activation_zero_point = calculate_qparams(
            activation_sample,
            bits=target.activation.bits,
            symmetric=target.activation.symmetric,
            axis=activation_axis,
        )
        activation_qparams = {
            "bits": target.activation.bits,
            "granularity": target.activation.granularity,
            "mode": target.activation.mode,
            "symmetric": target.activation.symmetric,
            "scale": [
                float(item) for item in activation_scale.reshape(-1).cpu()
            ],
            "zero_point": [
                int(item) for item in activation_zero_point.reshape(-1).cpu()
            ],
        }
    integer_hash = (
        hashlib.sha256(result.values.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if result is not None
        else None
    )
    materialized = QuantizedTarget(
        fqn=target.fqn,
        target_type=target.target_type,
        algorithm=target.algorithm,
        bits=spec.bits,
        granularity=spec.granularity,
        symmetric=spec.symmetric,
        scale=tuple(float(item) for item in scale.reshape(-1).cpu()),
        zero_point=tuple(int(item) for item in zero_point.reshape(-1).cpu()),
        fallback_reason=fallback_reason,
    )
    return materialized, activation_qparams, integer_hash


def oneshot(
    graph: GraphModule,
    config: QuantizationConfig | Mapping[str, object] | str,
    calibration_dataloader: Iterable[Mapping[str, Tensor]],
) -> GraphModule:
    """Calibrate and fake-quantize a prefill graph atomically."""
    value = metadata(graph)
    if value.stage != GraphStage.FLOAT_PREFILL:
        raise GraphStateError("oneshot requires a FLOAT_PREFILL graph")
    parsed = QuantizationConfig.load(config)
    if not parsed.modifiers:
        return graph
    plan = plan_quantization(graph, parsed)
    if not plan:
        raise QuantizationConfigError("Quantization selectors matched no targets")
    calibration = _calibration_samples(graph, calibration_dataloader)

    def mutate(candidate: GraphModule) -> None:
        current = metadata(candidate)
        materialized = tuple(
            _materialize_target(candidate, item, calibration) for item in plan
        )
        targets = tuple(item[0] for item in materialized)
        properties = dict(current.properties)
        properties["algorithms"] = sorted({item.algorithm for item in targets})
        properties["gptq"] = any(item.algorithm == "gptq" for item in targets)
        properties["fake_quant"] = True
        properties["activation_qparams"] = {
            plan_item.fqn: result[1]
            for plan_item, result in zip(plan, materialized, strict=True)
            if result[1] is not None
        }
        properties["quantized_integer_sha256"] = {
            plan_item.fqn: result[2]
            for plan_item, result in zip(plan, materialized, strict=True)
            if result[2] is not None
        }
        properties["gptq_fallbacks"] = {
            item.fqn: item.fallback_reason
            for item in targets
            if item.fallback_reason is not None
        }
        moe_targets = [item for item in targets if item.target_type == "moe"]
        if moe_targets:
            properties["moe_quant_parameter_order"] = (
                "input",
                *tuple(
                    f"expert.{expert}.{projection}"
                    for expert in range(5)
                    for projection in ("gate", "up", "intermediate", "down")
                ),
            )
        set_metadata(
            candidate,
            replace(
                current,
                stage=GraphStage.QUANTIZED_PREFILL,
                quantized_targets=targets,
                config_fingerprint=parsed.fingerprint,
                properties=properties,
            ),
        )

    return transactional_update(graph, mutate)
