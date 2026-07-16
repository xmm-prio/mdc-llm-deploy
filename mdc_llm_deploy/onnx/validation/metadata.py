"""Model-level metadata contract validation for MDC ONNX."""

from __future__ import annotations

from dataclasses import dataclass

import onnx

from ...capabilities import Algorithm, ModelKind, Target
from ...errors import OnnxExportError
from ...graph.metadata import GRAPH_SCHEMA_VERSION, GraphStage


@dataclass(frozen=True, slots=True)
class ValidatedMetadata:
    """Validated metadata values used by downstream checks."""

    properties: dict[str, str]
    stage: str
    mask_mode: str
    algorithms: frozenset[str]
    targets: frozenset[str]


def _metadata_values(
    value: str,
    label: str,
) -> tuple[str, ...]:
    items = tuple(value.split(","))
    if (
        not items
        or any(not item for item in items)
        or len(items) != len(set(items))
    ):
        raise OnnxExportError(
            f"Invalid MDC {label} metadata"
        )
    return items


def validate_metadata(
    model: onnx.ModelProto,
) -> ValidatedMetadata:
    """Validate required MDC markers and cross-property semantics."""
    properties = {
        item.key: item.value for item in model.metadata_props
    }
    required = {
        "mdc.graph_schema_version",
        "mdc.stage",
        "mdc.mask_mode",
        "mdc.mask_semantics",
        "mdc.model_kind",
        "mdc.algorithm",
        "mdc.target",
        "mdc.dialect",
        "mdc.numeric_spine",
        "mdc.lowering_source",
    }
    if required - properties.keys():
        raise OnnxExportError(
            "MDC metadata properties are incomplete"
        )
    if properties["mdc.graph_schema_version"] != str(
        GRAPH_SCHEMA_VERSION
    ):
        raise OnnxExportError(
            "Unsupported MDC graph schema version"
        )
    if properties["mdc.dialect"] != "MDC ONNX":
        raise OnnxExportError("Invalid MDC dialect marker")
    if (
        properties["mdc.numeric_spine"]
        != "validated-standard-aten"
    ):
        raise OnnxExportError(
            "MDC model lacks a validated numerical spine"
        )
    mask_mode = properties["mdc.mask_mode"]
    if mask_mode not in {"masked", "maskless"}:
        raise OnnxExportError("Invalid MDC mask mode")
    expected_semantics = (
        "explicit-causal"
        if mask_mode == "masked"
        else "all-visible-non-causal"
    )
    if properties["mdc.mask_semantics"] != expected_semantics:
        raise OnnxExportError(
            "Mask semantics metadata is inconsistent"
        )
    try:
        stage_value = GraphStage(properties["mdc.stage"])
        ModelKind(properties["mdc.model_kind"])
    except (TypeError, ValueError) as error:
        raise OnnxExportError(
            "Invalid MDC graph stage or model kind"
        ) from error
    algorithm_names = _metadata_values(
        properties["mdc.algorithm"],
        "algorithm",
    )
    target_names = _metadata_values(
        properties["mdc.target"],
        "target",
    )
    try:
        algorithms = frozenset(
            Algorithm(name).value
            for name in algorithm_names
        )
    except (TypeError, ValueError) as error:
        raise OnnxExportError(
            "Invalid MDC algorithm metadata"
        ) from error
    if Algorithm.GPTQ.value in algorithms:
        raise OnnxExportError(
            "MDC ONNX does not support GPTQ metadata"
        )
    if target_names == (Algorithm.FP16.value,):
        targets = frozenset({Algorithm.FP16.value})
    else:
        try:
            targets = frozenset(
                Target(name).value for name in target_names
            )
        except (TypeError, ValueError) as error:
            raise OnnxExportError(
                "Invalid MDC target metadata"
            ) from error
    if stage_value.is_quantized:
        if (
            Algorithm.FP16.value in algorithms
            or Algorithm.FP16.value in targets
        ):
            raise OnnxExportError(
                "Quantized stage metadata cannot declare fp16"
            )
    elif (
        algorithms != {Algorithm.FP16.value}
        or targets != {Algorithm.FP16.value}
    ):
        raise OnnxExportError(
            "Float stage metadata must declare fp16"
        )
    return ValidatedMetadata(
        properties=properties,
        stage=stage_value.value,
        mask_mode=mask_mode,
        algorithms=algorithms,
        targets=targets,
    )
