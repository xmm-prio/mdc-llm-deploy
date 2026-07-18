"""Transactional calibration and quantization engine."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace

from torch import Tensor
from torch.fx import GraphModule

from ..errors import GraphStateError, QuantizationConfigError
from ..graph.lifecycle import metadata, set_metadata, transactional_update
from ..graph.metadata import GraphStage
from ..observability import StageReporter, get_logger
from ..placement import capture_placement
from .calibration import collect_calibration_artifacts
from .config import QuantizationConfig
from .materialization import (
    MaterializationContext,
    MaterializationResult,
    materialize_alias_group,
)
from .placement import (
    group_alias_targets,
    restore_parameter_aliases,
    validate_quantized_placement,
)
from .planning import TargetPlan, plan_calibration, plan_quantization


def oneshot(
    graph: GraphModule,
    config: QuantizationConfig | Mapping[str, object] | str,
    calibration_dataloader: Iterable[Mapping[str, Tensor]],
) -> GraphModule:
    """Calibrate and fake-quantize a prefill graph atomically."""
    logger = get_logger("quantization")
    with StageReporter("Quantization planning") as planning_reporter:
        value = metadata(graph)
        if value.stage != GraphStage.FLOAT_PREFILL:
            raise GraphStateError("oneshot requires a FLOAT_PREFILL graph")
        parsed = QuantizationConfig.load(config)
        fingerprint = parsed.fingerprint
        planning_reporter.update(
            config_fingerprint=fingerprint[:12],
            modifier_count=len(parsed.modifiers),
        )
        if not parsed.modifiers:
            logger.info("Quantization planning skipped: no modifiers")
            planning_reporter.update(
                outcome="skipped",
                target_count=0,
                calibration_boundary_count=0,
                alias_group_count=0,
                algorithm_counts="none",
            )
            return graph
        plan = plan_quantization(graph, parsed)
        if not plan:
            raise QuantizationConfigError("Quantization selectors matched no targets")
        calibration_plan = plan_calibration(plan)
        placement = capture_placement(graph)
        groups = group_alias_targets(graph, plan)
        algorithm_counts = Counter(item.algorithm for item in plan)
        planning_reporter.update(
            outcome="completed",
            target_count=len(plan),
            calibration_boundary_count=len(calibration_plan.requirements),
            alias_group_count=len(groups),
            algorithm_counts=", ".join(
                f"{name}:{count}" for name, count in sorted(algorithm_counts.items())
            ),
        )
        for item in plan:
            logger.debug(
                "Planned quantization target: fqn=%s type=%s algorithm=%s",
                item.fqn,
                item.target_type,
                item.algorithm,
            )
    calibration = collect_calibration_artifacts(
        graph,
        calibration_dataloader,
        calibration_plan,
    )

    def mutate(candidate: GraphModule) -> None:
        current = metadata(candidate)
        context = MaterializationContext.capture(candidate)
        paired: list[tuple[TargetPlan, MaterializationResult]] = []
        with materialization_reporter.progress(
            "Materializing quantization targets",
            total=len(groups),
        ) as progress:
            for group in groups:
                results = materialize_alias_group(
                    context,
                    group.targets,
                    calibration,
                )
                restore_parameter_aliases(candidate, group)
                paired.extend(zip(group.targets, results, strict=True))
                progress.advance()
        materialized = tuple(result for _, result in paired)
        targets = tuple(item.target for item in materialized)
        fallback_count = sum(
            item.fallback_reason is not None for item in targets
        )
        materialization_reporter.update(
            target_count=len(targets),
            alias_group_count=len(groups),
            fallback_count=fallback_count,
        )
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
                config_fingerprint=fingerprint,
                properties=properties,
            ),
        )
        validate_quantized_placement(placement, candidate)

    with StageReporter("Quantization materialization") as materialization_reporter:
        return transactional_update(graph, mutate)
