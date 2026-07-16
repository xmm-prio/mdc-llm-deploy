"""Transactional calibration and quantization engine."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

from torch import Tensor
from torch.fx import GraphModule

from ..config import QuantizationConfig
from ..errors import GraphStateError, QuantizationConfigError
from ..graph import metadata, set_metadata, transactional_update
from ..graph_types import GraphStage
from ..placement import capture_placement
from .calibration import collect_calibration_samples
from .materialization import MaterializationResult, materialize_alias_group
from .placement import (
    group_alias_targets,
    restore_parameter_aliases,
    validate_quantized_placement,
)
from .planner import TargetPlan, plan_quantization


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
    placement = capture_placement(graph)
    groups = group_alias_targets(graph, plan)
    calibration = collect_calibration_samples(graph, calibration_dataloader)

    def mutate(candidate: GraphModule) -> None:
        current = metadata(candidate)
        paired: list[tuple[TargetPlan, MaterializationResult]] = []
        for group in groups:
            results = materialize_alias_group(
                candidate,
                group.targets,
                calibration,
            )
            restore_parameter_aliases(candidate, group)
            paired.extend(zip(group.targets, results, strict=True))
        materialized = tuple(result for _, result in paired)
        targets = tuple(item.target for item in materialized)
        properties = dict(current.properties)
        properties["fake_quant"] = True
        properties["activation_qparams"] = {
            plan_item.fqn: result.activation_qparams
            for plan_item, result in paired
            if result.activation_qparams is not None
        }
        properties["quantized_integer_sha256"] = {
            plan_item.fqn: result.integer_sha256
            for plan_item, result in paired
            if result.integer_sha256 is not None
        }
        properties["gptq_fallbacks"] = {
            item.fqn: item.fallback_reason
            for item in targets
            if item.fallback_reason is not None
        }
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
        validate_quantized_placement(placement, candidate)

    return transactional_update(graph, mutate)
