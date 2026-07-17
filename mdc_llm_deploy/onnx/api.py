"""Atomic lowering from an ATen FX graph to the MDC ONNX dialect."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import onnx
from onnx import helper
from torch.fx import GraphModule

from ..errors import OnnxExportError
from ..graph.fx.ownership import is_fqn_descendant
from ..graph.lifecycle import metadata
from ..graph.metadata import GraphMetadata
from ..operators.contracts.onnx import MDC_ONNX_OPSET
from .export.artifacts import commit_validated_onnx
from .export.standard import export_standard_onnx
from .transform.attention import (
    MaskMode as MaskMode,
)
from .transform.attention import (
    lower_maskless_attention,
    lower_rms_norms,
    lower_rope_attention,
)
from .transform.cleanup import (
    prune_unreachable,
    remove_dynamic_value_info,
    topologically_sort,
)
from .transform.linear import append_quantized_linears
from .transform.moe import adapt_quantized_moe
from .transform.output import retain_logits_output
from .validation.compatibility import validate_onnx_compatibility
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
    attention_boundaries = sorted(
        (
            boundary
            for boundary in value.boundaries
            if boundary.kind == "attention"
        ),
        key=lambda boundary: boundary.fqn,
    )
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
        )
    append_quantized_linears(model, value)
    adapt_quantized_moe(model, value)
    if value.output_abi and value.output_abi[0].name == "logits":
        retain_logits_output(model)
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


def onnx_export(
    graph: GraphModule,
    output_path: str | Path,
    *,
    external_data: bool = True,
) -> onnx.ModelProto:
    """Lower an FX graph and atomically replace requested ONNX artifacts."""
    value = metadata(graph)
    configured_mask = value.properties.get("mask_mode", "causal")
    if configured_mask not in {"causal", "none"}:
        raise OnnxExportError("Graph mask_mode metadata is invalid")
    mask_mode: MaskMode = (
        "masked" if configured_mask == "causal" else "maskless"
    )
    validate_onnx_compatibility(value, mask_mode)
    target = Path(output_path)
    if target.suffix.lower() != ".onnx":
        raise OnnxExportError("output_path must use .onnx suffix")
    target.parent.mkdir(parents=True, exist_ok=True)
    standard = export_standard_onnx(graph, value, target.parent)
    model = _lower(standard, value, mask_mode)
    validate_mdc_model(model)
    return commit_validated_onnx(
        model,
        target,
        external_data=external_data,
    )
