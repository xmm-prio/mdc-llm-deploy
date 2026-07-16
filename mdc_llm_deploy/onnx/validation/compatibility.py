"""Metadata-driven compatibility checks for MDC ONNX lowering."""

from __future__ import annotations

from ...errors import OnnxExportError, UnsupportedPatternError
from ...graph import validate_capability_request
from ...graph_types import GraphMetadata


def validate_onnx_compatibility(
    value: GraphMetadata,
    mask_mode: str,
) -> None:
    """Validate graph metadata against MDC ONNX constraints."""
    if mask_mode not in {"masked", "maskless"}:
        raise ValueError(
            "mask_mode must be 'masked' or 'maskless'"
        )
    if value.stage.is_prefill and not value.quantized_targets:
        return
    try:
        validate_capability_request(
            value,
            mask_mode=mask_mode,
            artifact="onnx",
        )
    except UnsupportedPatternError as error:
        raise OnnxExportError(str(error)) from error
