"""Independent structural validator for the non-standard MDC ONNX dialect."""

from __future__ import annotations

import math
from collections import Counter

import onnx
from onnx import TensorProto, numpy_helper

from ..errors import OnnxExportError

CUSTOM_OPS = {
    "NPURmsNorm",
    "ApplyRotaryPosEmb",
    "FusedInferAttentionScore",
    "NPUAscendQuantV2",
    "AscendDequant",
    "MoeExpert",
}
_STANDARD_DOMAINS = {"", "ai.onnx"}


def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise OnnxExportError(f"Value {value.name!r} has no shape")
    result: list[int] = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise OnnxExportError(f"Value {value.name!r} has a dynamic shape")
        result.append(dimension.dim_value)
    return tuple(result)


def _properties(model: onnx.ModelProto) -> dict[str, str]:
    return {item.key: item.value for item in model.metadata_props}


def _attributes(node: onnx.NodeProto) -> dict[str, onnx.AttributeProto]:
    return {item.name: item for item in node.attribute}


def _require_attributes(node: onnx.NodeProto, required: set[str]) -> None:
    missing = required - _attributes(node).keys()
    if missing:
        raise OnnxExportError(
            f"{node.op_type} attributes are incomplete: {sorted(missing)}"
        )


def _validate_operator(node: onnx.NodeProto, mask_mode: str) -> None:
    if node.op_type == "NPURmsNorm":
        if len(node.input) != 2 or len(node.output) != 2:
            raise OnnxExportError("NPURmsNorm must use 2 inputs and 2 outputs")
        _require_attributes(node, {"epsilon"})
        epsilon = onnx.helper.get_attribute_value(_attributes(node)["epsilon"])
        if not math.isclose(float(epsilon), 1e-6, rel_tol=0.0, abs_tol=1e-12):
            raise OnnxExportError("NPURmsNorm epsilon must equal 1e-6")
    elif node.op_type == "ApplyRotaryPosEmb":
        if len(node.input) != 4 or len(node.output) != 2:
            raise OnnxExportError("ApplyRotaryPosEmb must use 4 inputs and 2 outputs")
        _require_attributes(node, {"layout", "rotary_mode"})
        layout = onnx.helper.get_attribute_value(_attributes(node)["layout"])
        rotary_mode = onnx.helper.get_attribute_value(_attributes(node)["rotary_mode"])
        if layout != 1 or rotary_mode != b"half":
            raise OnnxExportError("ApplyRotaryPosEmb must use BSND half rotation")
    elif node.op_type == "FusedInferAttentionScore":
        if len(node.input) != 29 or len(node.output) != 2:
            raise OnnxExportError(
                "FusedInferAttentionScore must use the complete 29-slot ABI"
            )
        _require_attributes(
            node,
            {
                "num_heads",
                "num_key_value_heads",
                "scale",
                "input_layout",
                "sparse_mode",
                "pre_tokens",
                "next_tokens",
                "softmax_lse_flag",
            },
        )
        attributes = _attributes(node)
        if onnx.helper.get_attribute_value(attributes["input_layout"]) != b"BNSD":
            raise OnnxExportError("FusedInferAttentionScore layout must be BNSD")
        if onnx.helper.get_attribute_value(attributes["sparse_mode"]) != 0:
            raise OnnxExportError("FusedInferAttentionScore sparse_mode must be 0")
        if mask_mode == "masked" and not node.input[4]:
            raise OnnxExportError("Masked attention requires atten_mask")
        if mask_mode == "maskless" and node.input[4]:
            raise OnnxExportError("Maskless attention must omit atten_mask")
    elif node.op_type == "NPUAscendQuantV2":
        if len(node.input) not in {2, 3} or len(node.output) != 1:
            raise OnnxExportError("NPUAscendQuantV2 ABI is invalid")
        _require_attributes(node, {"axis", "dtype"})
        if onnx.helper.get_attribute_value(_attributes(node)["dtype"]) != 2:
            raise OnnxExportError("Release quantization must use INT8 dtype=2")
    elif node.op_type == "AscendDequant":
        if len(node.input) != 2 or len(node.output) != 1:
            raise OnnxExportError("AscendDequant ABI is invalid")
        _require_attributes(node, {"sqrt_mode", "relu_flag", "dtype"})
        attributes = _attributes(node)
        if onnx.helper.get_attribute_value(attributes["sqrt_mode"]) != 0:
            raise OnnxExportError("AscendDequant sqrt_mode must be false")
        if onnx.helper.get_attribute_value(attributes["relu_flag"]) != 0:
            raise OnnxExportError("AscendDequant relu_flag must be false")
        if onnx.helper.get_attribute_value(attributes["dtype"]) not in {0, 1}:
            raise OnnxExportError("AscendDequant dtype must be 0 or 1")
    elif node.op_type == "MoeExpert":
        if len(node.input) not in {5, 6} or len(node.output) != 1:
            raise OnnxExportError("MoeExpert ABI is invalid")


def _validate_dequant_initializers(model: onnx.ModelProto) -> None:
    initializers = {item.name: item for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "AscendDequant":
            continue
        scale = initializers.get(node.input[1])
        if scale is None or scale.data_type != TensorProto.UINT64:
            raise OnnxExportError("AscendDequant scale must be a UINT64 initializer")
        values = numpy_helper.to_array(scale).astype("uint64", copy=False)
        if ((values >> 32) != 0).any():
            raise OnnxExportError("AscendDequant scale high 32 bits must be zero")
        decoded = (values & 0xFFFFFFFF).astype("uint32").view("float32")
        if not ((decoded > 0) & (decoded < float("inf"))).all():
            raise OnnxExportError("AscendDequant scale must decode to finite positives")


def _validate_custom_node_reachability(
    model: onnx.ModelProto,
    properties: dict[str, str],
) -> None:
    """Require every MDC custom node to contribute to graph outputs."""
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
    }
    pending = [item.name for item in model.graph.output]
    visited_values: set[str] = set()
    while pending:
        value = pending.pop()
        if value in visited_values:
            continue
        visited_values.add(value)
        producer = producers.get(value)
        if producer is None:
            continue
        pending.extend(name for name in producer.input if name)

    reachable_ops = {
        "NPURmsNorm",
        "ApplyRotaryPosEmb",
        "FusedInferAttentionScore",
        "NPUAscendQuantV2",
        "AscendDequant",
    }
    custom_nodes = [
        node
        for node in model.graph.node
        if node.op_type in reachable_ops
        and not node.name.startswith("mdc.moe.")
    ]
    isolated = [
        node.name or node.op_type
        for node in custom_nodes
        if not any(output in visited_values for output in node.output)
    ]
    if isolated:
        raise OnnxExportError(
            f"MDC custom nodes do not reach graph outputs: {isolated}"
        )

    quantized_nodes = [
        node
        for node in custom_nodes
        if node.op_type in {"NPUAscendQuantV2", "AscendDequant"}
    ]
    targets = set(properties["mdc.target"].split(","))
    if targets == {"linear"}:
        raw_count = properties.get("mdc.linear.target_count")
        try:
            target_count = int(raw_count or "")
        except ValueError as error:
            raise OnnxExportError("Linear target count metadata is invalid") from error
        counts = Counter(node.op_type for node in quantized_nodes)
        if (
            target_count <= 0
            or counts["NPUAscendQuantV2"] != target_count
            or counts["AscendDequant"] != target_count
        ):
            raise OnnxExportError("Linear quantization target coverage is incomplete")


def _validate_io_abi(model: onnx.ModelProto, stage: str) -> None:
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    expected_inputs = (
        ["input_ids"]
        if stage.endswith("PREFILL")
        else [
            "input_ids",
            "past_key_values.0.key",
            "past_key_values.0.value",
        ]
    )
    if [item.name for item in inputs] != expected_inputs:
        raise OnnxExportError("MDC runtime input ABI is invalid")
    if [item.name for item in outputs] != [
        "logits",
        "present.0.key",
        "present.0.value",
    ]:
        raise OnnxExportError("MDC runtime output ABI is invalid")
    input_ids = inputs[0]
    if input_ids.type.tensor_type.elem_type != TensorProto.INT64:
        raise OnnxExportError("input_ids must use INT64")
    logits_shape = _shape(outputs[0])
    key_shape = _shape(outputs[1])
    value_shape = _shape(outputs[2])
    if len(logits_shape) != 3 or logits_shape[0] != 1:
        raise OnnxExportError("logits ABI shape is invalid")
    if len(key_shape) != 4 or key_shape != value_shape or key_shape[0] != 1:
        raise OnnxExportError("KV output ABI shape is invalid")
    if stage.endswith("PREFILL"):
        if _shape(input_ids) != (1, logits_shape[1]) or key_shape[2] != logits_shape[1]:
            raise OnnxExportError("Prefill sequence ABI is inconsistent")
    else:
        if _shape(input_ids) != (1, 1) or logits_shape[1] != 1:
            raise OnnxExportError("Decode query ABI must use one token")
        if _shape(inputs[1]) != (key_shape[0], key_shape[1], key_shape[2] - 1, key_shape[3]):
            raise OnnxExportError("Decode key cache must omit the current token")
        if _shape(inputs[2]) != (value_shape[0], value_shape[1], value_shape[2] - 1, value_shape[3]):
            raise OnnxExportError("Decode value cache must omit the current token")


def _validate_mask_initializer(
    model: onnx.ModelProto,
    attention: onnx.NodeProto,
    stage: str,
) -> None:
    initializers = {item.name: item for item in model.graph.initializer}
    mask = initializers.get(attention.input[4])
    if mask is None or mask.data_type != TensorProto.BOOL:
        raise OnnxExportError("Attention mask must be a BOOL initializer")
    cache_shape = _shape(model.graph.output[1])
    query_length = _shape(model.graph.output[0])[1]
    expected = (1, 1, query_length, cache_shape[2])
    if tuple(mask.dims) != expected:
        raise OnnxExportError(
            f"{stage} attention mask shape must be {expected}"
        )


def _validate_moe_contract(model: onnx.ModelProto, properties: dict[str, str]) -> None:
    node = next(item for item in model.graph.node if item.op_type == "MoeExpert")
    initializers = {item.name: item for item in model.graph.initializer}
    scales = initializers.get(node.input[4])
    offsets = initializers.get(node.input[5]) if len(node.input) == 6 else None
    if scales is None or scales.data_type != TensorProto.FLOAT or tuple(scales.dims) != (21,):
        raise OnnxExportError("MoeExpert quant_scales must be FLOAT32[21]")
    if (
        offsets is None
        or offsets.data_type != TensorProto.INT32
        or tuple(offsets.dims) != (21,)
    ):
        raise OnnxExportError("MoeExpert quant_offsets must be INT32[21]")
    required = {
        "mdc.moe.expert_order",
        "mdc.moe.weight_projection_order",
        "mdc.moe.weight_offsets",
        "mdc.moe.quant_parameter_count",
    }
    if required - properties.keys() or properties["mdc.moe.quant_parameter_count"] != "21":
        raise OnnxExportError("MoE packing metadata is incomplete")


def validate_mdc_model(model: onnx.ModelProto) -> None:
    """Validate protobuf, SSA, topology, ABI, metadata, and MDC constraints."""
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be onnx.ModelProto")
    if model.ir_version <= 0:
        raise OnnxExportError("ONNX IR version must be positive")
    opsets = {item.domain: item.version for item in model.opset_import}
    if opsets.get("", opsets.get("ai.onnx")) != 18:
        raise OnnxExportError("MDC ONNX must use opset 18")
    if set(opsets) - _STANDARD_DOMAINS:
        raise OnnxExportError("MDC ONNX imports a forbidden operator domain")
    properties = _properties(model)
    required_properties = {
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
    if required_properties - properties.keys():
        raise OnnxExportError("MDC metadata properties are incomplete")
    if properties["mdc.dialect"] != "MDC ONNX":
        raise OnnxExportError("Invalid MDC dialect marker")
    if properties["mdc.numeric_spine"] != "validated-standard-aten":
        raise OnnxExportError("MDC model lacks a validated numerical spine")
    mask_mode = properties["mdc.mask_mode"]
    if mask_mode not in {"masked", "maskless"}:
        raise OnnxExportError("Invalid MDC mask mode")
    expected_semantics = (
        "explicit-causal" if mask_mode == "masked" else "all-visible-non-causal"
    )
    if properties["mdc.mask_semantics"] != expected_semantics:
        raise OnnxExportError("Mask semantics metadata is inconsistent")

    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        _shape(value)
    stage = properties["mdc.stage"]
    if stage not in {
        "FLOAT_PREFILL",
        "QUANTIZED_PREFILL",
        "FLOAT_DECODE",
        "QUANTIZED_DECODE",
    }:
        raise OnnxExportError("Invalid MDC graph stage")
    _validate_io_abi(model, stage)
    input_names = [item.name for item in model.graph.input]
    output_names = [item.name for item in model.graph.output]
    if len(input_names) != len(set(input_names)) or len(output_names) != len(set(output_names)):
        raise OnnxExportError("ONNX graph I/O names must be unique")
    initializer_names = {item.name for item in model.graph.initializer}
    if initializer_names.intersection(output_names):
        raise OnnxExportError("Graph outputs must not be initializer placeholders")

    known = set(input_names) | initializer_names
    produced: set[str] = set()
    output_producers: dict[str, str] = {}
    for node in model.graph.node:
        if node.domain not in _STANDARD_DOMAINS:
            raise OnnxExportError(f"Node {node.name!r} uses forbidden domain")
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            raise OnnxExportError("MDC ONNX must not contain QDQ nodes")
        missing = [name for name in node.input if name and name not in known]
        if missing:
            raise OnnxExportError(
                f"Node {node.name!r} is not topologically sorted: {missing}"
            )
        for name in node.output:
            if not name or name in known or name in produced:
                raise OnnxExportError(f"ONNX SSA violation at output {name!r}")
            produced.add(name)
            known.add(name)
            output_producers[name] = node.op_type
        if node.op_type in CUSTOM_OPS:
            _validate_operator(node, mask_mode)
    missing_outputs = [name for name in output_names if name not in produced]
    if missing_outputs:
        raise OnnxExportError(
            f"Graph outputs lack numerical producers: {missing_outputs}"
        )
    if any(output_producers[name] in {"Constant", "ConstantOfShape"} for name in output_names):
        raise OnnxExportError("Graph outputs must not be constant placeholders")

    counts = Counter(node.op_type for node in model.graph.node)
    required = {"NPURmsNorm", "ApplyRotaryPosEmb", "FusedInferAttentionScore"}
    absent = required - counts.keys()
    if absent:
        raise OnnxExportError(f"Missing MDC operators: {sorted(absent)}")
    if "linear" in properties["mdc.target"]:
        required_linear = {"NPUAscendQuantV2", "MatMul", "AscendDequant"}
        if required_linear - counts.keys():
            raise OnnxExportError("Linear quantization lowering is incomplete")
    if "moe" in properties["mdc.target"] and counts["MoeExpert"] == 0:
        raise OnnxExportError("MoE quantization lowering is incomplete")
    attention = next(
        node for node in model.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    if mask_mode == "masked":
        _validate_mask_initializer(model, attention, stage)
    if "moe" in properties["mdc.target"]:
        _validate_moe_contract(model, properties)
    _validate_dequant_initializers(model)
    _validate_custom_node_reachability(model, properties)


def validate_serialized_model(path: str) -> onnx.ModelProto:
    """Load and validate a serialized MDC protobuf without standard schema checks."""
    try:
        model = onnx.load(path, load_external_data=False)
    except Exception as error:
        raise OnnxExportError(f"Cannot read ONNX protobuf: {error}") from error
    validate_mdc_model(model)
    return model
