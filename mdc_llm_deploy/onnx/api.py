"""Atomic lowering from an ATen FX graph to the MDC ONNX dialect."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import onnx
from onnx import helper
from torch.fx import GraphModule

from ..errors import GraphStateError, OnnxExportError
from ..graph.fx.ownership import is_fqn_descendant
from ..graph.lifecycle import metadata
from ..graph.metadata import (
    SAVE_KV_CACHE_PROPERTY,
    GraphMetadata,
    order_attention_boundaries,
    resolve_save_kv_cache,
)
from ..observability import StageReporter
from ..operators.contracts.onnx import MDC_ONNX_OPSET
from .export.artifacts import commit_mdc_onnx, commit_standard_onnx
from .export.normalization import validate_normalized_onnx
from .export.standard import build_standard_onnx
from .transform.attention import (
    MaskMode as MaskMode,
)
from .transform.attention import (
    attention_cache_dtype_overrides,
    lower_maskless_attention,
    lower_rms_norms,
    lower_rope_attention,
    validate_lowered_attention_cache,
)
from .transform.cleanup import (
    prune_unreachable,
    remove_dynamic_value_info,
    topologically_sort,
)
from .transform.linear import append_quantized_linears
from .transform.moe import adapt_quantized_moe
from .transform.output import finalize_artifact_outputs
from .transform.support import OnnxLoweringContext
from .validation.compatibility import validate_mdc_onnx_compatibility
from .validation.io import validate_io_abi
from .validation.model import validate_mdc_model


def _lower(
    standard: onnx.ModelProto,
    value: GraphMetadata,
    mask_mode: MaskMode,
) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(standard)
    model.producer_name = "mdc_llm_deploy"
    model.producer_version = "0.1.0"
    del model.opset_import[:]
    model.opset_import.append(
        helper.make_opsetid("", MDC_ONNX_OPSET)
    )
    if mask_mode == "maskless":
        lower_maskless_attention(model)
    if any(boundary.kind == "rms_norm" for boundary in value.boundaries):
        lower_rms_norms(model, value)
    lowering_context = OnnxLoweringContext.from_model(model)
    try:
        attention_boundaries = order_attention_boundaries(value.boundaries)
    except GraphStateError as error:
        raise OnnxExportError(
            f"Invalid attention layer order: {error}"
        ) from error
    for layer_id, attention in enumerate(attention_boundaries):
        ropes = tuple(
            boundary
            for boundary in value.boundaries
            if boundary.kind == "rope"
            and is_fqn_descendant(boundary.fqn, attention.fqn)
        )
        if len(ropes) != 1:
            raise OnnxExportError(
                f"Attention {attention.fqn!r} requires one owned RoPE boundary"
            )
        lower_rope_attention(
            model,
            replace(value, boundaries=(attention, ropes[0])),
            mask_mode,
            layer_id=layer_id,
            context=lowering_context,
        )
    append_quantized_linears(model, value, lowering_context)
    adapt_quantized_moe(model, value, lowering_context)
    validate_lowered_attention_cache(model, value)
    finalize_artifact_outputs(model, value)
    prune_unreachable(model)
    topologically_sort(model)
    algorithms = sorted({item.algorithm for item in value.quantized_targets}) or ["fp16"]
    targets = sorted({item.target_type for item in value.quantized_targets}) or ["fp16"]
    properties = {
        "mdc.graph_schema_version": str(value.schema_version),
        "mdc.stage": value.stage.value,
        "mdc.mask_mode": mask_mode,
        "mdc.mask_semantics": (
            "explicit-causal" if mask_mode == "masked" else "all-visible-non-causal"
        ),
        "mdc.model_kind": value.model_kind,
        "mdc.algorithm": ",".join(algorithms),
        "mdc.target": ",".join(targets),
        "mdc.config_fingerprint": value.config_fingerprint or "",
        "mdc.dialect": "MDC ONNX",
        "mdc.numeric_spine": "validated-standard-aten",
        "mdc.lowering_source": "fx-boundaries-and-graph-metadata",
        SAVE_KV_CACHE_PROPERTY: str(
            resolve_save_kv_cache(value.properties)
        ).lower(),
    }
    linear_target_count = sum(item.target_type == "linear" for item in value.quantized_targets)
    if linear_target_count:
        properties["mdc.linear.target_count"] = str(linear_target_count)
    remove_dynamic_value_info(model)
    helper.set_model_props(
        model,
        properties,
    )
    return model


def _prepare_output_path(output_path: str | Path) -> Path:
    target = Path(output_path)
    if target.suffix.lower() != ".onnx":
        raise OnnxExportError("output_path must use .onnx suffix")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def onnx_export(
    graph: GraphModule,
    output_path: str | Path,
    *,
    external_data: bool = True,
) -> onnx.ModelProto:
    """Lower an FX graph and atomically replace requested ONNX artifacts."""
    with StageReporter(
        "MDC ONNX export",
        fields={"external_data": external_data},
    ) as reporter:
        value = metadata(graph)
        reporter.update(
            model_kind=value.model_kind,
            graph_stage=value.stage.value,
        )
        configured_mask = value.properties.get("mask_mode", "causal")
        if configured_mask not in {"causal", "none"}:
            raise OnnxExportError("Graph mask_mode metadata is invalid")
        mask_mode: MaskMode = (
            "masked" if configured_mask == "causal" else "maskless"
        )
        validate_mdc_onnx_compatibility(value, mask_mode)
        target = _prepare_output_path(output_path)
        standard = build_standard_onnx(graph, value, target.parent)
        model = _lower(standard, value, mask_mode)
        validate_mdc_model(
            model,
            value,
            output_dtype_overrides=attention_cache_dtype_overrides(value),
        )
        published = commit_mdc_onnx(
            model,
            target,
            external_data=external_data,
        )
        reporter.update(
            node_count=len(published.graph.node),
            initializer_count=len(published.graph.initializer),
        )
        return published


def standard_onnx_export(
    graph: GraphModule,
    output_path: str | Path,
    *,
    external_data: bool = True,
) -> onnx.ModelProto:
    """Export and atomically publish a standard ONNX model."""
    with StageReporter(
        "Standard ONNX export",
        fields={"external_data": external_data},
    ) as reporter:
        value = metadata(graph)
        reporter.update(
            model_kind=value.model_kind,
            graph_stage=value.stage.value,
        )
        target = _prepare_output_path(output_path)
        model = build_standard_onnx(graph, value, target.parent)
        finalize_artifact_outputs(model, value)
        validate_normalized_onnx(model)
        validate_io_abi(model, value)
        published = commit_standard_onnx(
            model,
            target,
            external_data=external_data,
        )
        reporter.update(
            node_count=len(published.graph.node),
            initializer_count=len(published.graph.initializer),
        )
        return published
