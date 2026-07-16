"""Atomic lowering from an ATen FX graph to the MDC ONNX dialect."""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import helper
from torch.fx import GraphModule

from ..errors import OnnxExportError
from ..graph import metadata
from ..graph_types import GraphMetadata
from ..onnx_protocol import MDC_ONNX_OPSET
from .artifact_io import commit_validated_onnx
from .attention_lowering import (
    MaskMode as MaskMode,
)
from .attention_lowering import (
    lower_maskless_attention,
    lower_rms_norms,
    lower_rope_attention,
)
from .compatibility import validate_onnx_compatibility
from .graph_cleanup import (
    prune_unreachable,
    remove_dynamic_value_info,
    topologically_sort,
)
from .linear_lowering import append_quantized_linears
from .moe_lowering import append_moe, moe_metadata_properties
from .standard_export import export_standard_onnx
from .validator import validate_mdc_model


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
    lower_rms_norms(model, value)
    lower_rope_attention(model, value, mask_mode)
    append_quantized_linears(model, value)
    append_moe(model, value)
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
    properties.update(moe_metadata_properties(value))
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
    mask_mode: MaskMode,
    overwrite: bool = False,
) -> onnx.ModelProto:
    """Lower an FX graph and atomically replace the requested ONNX file."""
    value = metadata(graph)
    validate_onnx_compatibility(value, mask_mode)
    target = Path(output_path)
    if target.suffix.lower() != ".onnx":
        raise OnnxExportError("output_path must use .onnx suffix")
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    standard = export_standard_onnx(graph, value, target.parent)
    model = _lower(standard, value, mask_mode)
    validate_mdc_model(model)
    return commit_validated_onnx(model, target, overwrite=overwrite)
