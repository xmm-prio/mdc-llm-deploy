"""Model-independent structural validation for MDC ONNX."""

from __future__ import annotations

from dataclasses import dataclass

import onnx

from ...errors import OnnxExportError
from ...operators.contracts.attention import ATTENTION_INPUT_COUNT
from ...operators.contracts.onnx import MDC_ONNX_DOMAIN, MDC_ONNX_OPSET
from ...operators.contracts.schema import OPERATOR_SCHEMAS
from .metadata import ValidatedMetadata, validate_metadata
from .topology import (
    QuantizationTopologyEvidence,
    validate_quantization_topology,
)

_CUSTOM_SCHEMAS = {
    schema.onnx_name: schema for schema in OPERATOR_SCHEMAS.values()
}
_STANDARD_DOMAINS = {"", MDC_ONNX_DOMAIN}
_REQUIRED_INPUTS = {
    "NPURmsNorm": 2,
    "ApplyRotaryPosEmb": 4,
    "FusedInferAttentionScore": 3,
    "NPUAscendQuantV2": 2,
    "AscendDequant": 2,
    "MoeExpert": 4,
}


@dataclass(frozen=True, slots=True)
class ValidatedMdcArtifact:
    """Validated serialized MDC ONNX model and metadata."""

    model: onnx.ModelProto
    metadata: ValidatedMetadata
    topology: QuantizationTopologyEvidence


def _validate_names(model: onnx.ModelProto) -> None:
    input_names = [item.name for item in model.graph.input]
    output_names = [item.name for item in model.graph.output]
    if (
        len(input_names) != len(set(input_names))
        or len(output_names) != len(set(output_names))
    ):
        raise OnnxExportError("ONNX graph input/output names must be unique")
    if set(input_names) & set(output_names):
        raise OnnxExportError("ONNX graph inputs and outputs must be distinct")


def _validate_custom_nodes(model: onnx.ModelProto) -> None:
    for node in model.graph.node:
        if node.op_type not in _CUSTOM_SCHEMAS:
            continue
        schema = _CUSTOM_SCHEMAS[node.op_type]
        if len(node.output) != len(schema.outputs):
            raise OnnxExportError(
                f"{node.op_type} output count does not match schema"
            )
        required_inputs = _REQUIRED_INPUTS[node.op_type]
        maximum_inputs = (
            ATTENTION_INPUT_COUNT
            if node.op_type == "FusedInferAttentionScore"
            else len(schema.inputs)
        )
        if (
            len(node.input) < required_inputs
            or not all(node.input[index] for index in range(required_inputs))
            or len(node.input) > maximum_inputs
        ):
            raise OnnxExportError(
                f"{node.op_type} input count does not match schema"
            )


def validate_standard_model(model: onnx.ModelProto) -> None:
    """Validate a standard ONNX model without MDC dialect assumptions."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be onnx.ModelProto")
    if model.ir_version <= 0:
        raise OnnxExportError("ONNX IR version must be positive")
    _validate_names(model)
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as error:
        raise OnnxExportError(f"ONNX checker failed: {error}") from error


def validate_mdc_model_structure(model: onnx.ModelProto) -> None:
    """Validate protobuf structure, I/O, opset, and custom schemas."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be onnx.ModelProto")
    if model.ir_version <= 0:
        raise OnnxExportError("ONNX IR version must be positive")
    opsets = {item.domain: item.version for item in model.opset_import}
    if opsets.get("", opsets.get(MDC_ONNX_DOMAIN)) != MDC_ONNX_OPSET:
        raise OnnxExportError(f"MDC ONNX must use opset {MDC_ONNX_OPSET}")
    if set(opsets) - _STANDARD_DOMAINS:
        raise OnnxExportError("ONNX imports an unsupported operator domain")
    _validate_names(model)
    _validate_custom_nodes(model)
    if not any(node.op_type in _CUSTOM_SCHEMAS for node in model.graph.node):
        validate_standard_model(model)


def validate_mdc_model(model: onnx.ModelProto) -> ValidatedMetadata:
    """Validate a complete in-memory MDC ONNX artifact."""
    validate_mdc_model_structure(model)
    metadata = validate_metadata(model)
    validate_quantization_topology(model, metadata)
    return metadata


def _load_serialized_model(path: str) -> onnx.ModelProto:
    try:
        return onnx.load(path, load_external_data=True)
    except Exception as error:
        raise OnnxExportError(f"Cannot read ONNX protobuf: {error}") from error


def validate_serialized_standard_model(path: str) -> onnx.ModelProto:
    """Load external data and validate a serialized standard ONNX artifact."""
    model = _load_serialized_model(path)
    validate_standard_model(model)
    return model


def load_validated_mdc_artifact(path: str) -> ValidatedMdcArtifact:
    """Load external data and validate a complete MDC ONNX artifact."""
    model = _load_serialized_model(path)
    validate_mdc_model_structure(model)
    metadata = validate_metadata(model)
    topology = validate_quantization_topology(model, metadata)
    return ValidatedMdcArtifact(
        model=model,
        metadata=metadata,
        topology=topology,
    )


__all__ = [
    "ValidatedMdcArtifact",
    "load_validated_mdc_artifact",
    "validate_mdc_model",
    "validate_mdc_model_structure",
    "validate_serialized_standard_model",
    "validate_standard_model",
]
