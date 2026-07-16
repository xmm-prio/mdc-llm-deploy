"""Independent structural validator for the non-standard MDC ONNX dialect."""

from __future__ import annotations

import onnx

from ..errors import OnnxExportError
from ..onnx_protocol import MDC_ONNX_DOMAIN, MDC_ONNX_OPSET
from .attention_validation import (
    validate_attention_contract,
)
from .io_validation import validate_io_abi
from .metadata_validation import validate_metadata
from .model_inspection import static_shape as _shape
from .moe_validation import validate_moe_contract
from .operator_validation import (
    validate_dequant_initializers,
)
from .topology_validation import (
    STANDARD_DOMAINS,
    quantized_target_families,
    validate_custom_node_reachability,
    validate_graph_topology,
)


def validate_mdc_model(model: onnx.ModelProto) -> None:
    """Validate protobuf, SSA, topology, ABI, metadata, and MDC constraints."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be onnx.ModelProto")
    if model.ir_version <= 0:
        raise OnnxExportError("ONNX IR version must be positive")
    opsets = {item.domain: item.version for item in model.opset_import}
    if (
        opsets.get("", opsets.get(MDC_ONNX_DOMAIN))
        != MDC_ONNX_OPSET
    ):
        raise OnnxExportError(
            f"MDC ONNX must use opset {MDC_ONNX_OPSET}"
        )
    if set(opsets) - STANDARD_DOMAINS:
        raise OnnxExportError("MDC ONNX imports a forbidden operator domain")
    validated_metadata = validate_metadata(model)
    properties = validated_metadata.properties
    mask_mode = validated_metadata.mask_mode
    targets = validated_metadata.targets

    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        _shape(value)
    stage = validated_metadata.stage
    validate_io_abi(model, stage)
    counts = validate_graph_topology(model, mask_mode)
    actual_targets = quantized_target_families(model)
    declared_targets = targets - {"fp16"}
    if actual_targets != declared_targets:
        raise OnnxExportError(
            "MDC target metadata does not match quantized topology"
        )
    required = {"NPURmsNorm", "ApplyRotaryPosEmb", "FusedInferAttentionScore"}
    absent = required - counts.keys()
    if absent:
        raise OnnxExportError(f"Missing MDC operators: {sorted(absent)}")
    if counts["FusedInferAttentionScore"] != 1:
        raise OnnxExportError(
            "Release lowering requires exactly one FusedInferAttentionScore"
        )
    if "linear" in targets:
        required_linear = {"NPUAscendQuantV2", "MatMul", "AscendDequant"}
        if required_linear - counts.keys():
            raise OnnxExportError("Linear quantization lowering is incomplete")
    if (
        "moe" in targets
        and counts["MoeExpert"] != 1
    ):
        raise OnnxExportError(
            "MoE quantization lowering requires exactly one MoeExpert"
        )
    attention = next(
        node for node in model.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    validate_attention_contract(
        model,
        attention,
        mask_mode=mask_mode,
        stage=stage,
    )
    if "moe" in targets:
        validate_moe_contract(model, properties)
    validate_dequant_initializers(model)
    validate_custom_node_reachability(model, properties)


def validate_serialized_model(path: str) -> onnx.ModelProto:
    """Load and validate a serialized MDC protobuf without standard schema checks."""
    try:
        model = onnx.load(path, load_external_data=False)
    except Exception as error:
        raise OnnxExportError(f"Cannot read ONNX protobuf: {error}") from error
    validate_mdc_model(model)
    return model
