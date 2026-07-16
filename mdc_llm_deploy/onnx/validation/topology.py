"""Topology and target-coverage checks for MDC custom ONNX nodes."""

from __future__ import annotations

from collections import Counter

import onnx

from ...attention_layout import AttentionInput
from ...errors import OnnxExportError
from ...onnx_protocol import MDC_ONNX_DOMAIN
from ...operator_schema import OPERATOR_SCHEMAS
from .operator import validate_operator

STANDARD_DOMAINS = frozenset({"", MDC_ONNX_DOMAIN})
CUSTOM_OPS = frozenset(
    schema.onnx_name for schema in OPERATOR_SCHEMAS.values()
)


def quantized_target_families(
    model: onnx.ModelProto,
) -> frozenset[str]:
    """Infer quantized target families from validated graph topology."""
    initializers = {
        item.name: item for item in model.graph.initializer
    }
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    result: set[str] = set()
    moe_nodes = [
        node
        for node in model.graph.node
        if node.op_type == "MoeExpert"
    ]
    quantized_moe_nodes = [
        node
        for node in moe_nodes
        if (
            len(node.input) > 4
            and bool(node.input[4])
            and (weight := initializers.get(node.input[3])) is not None
            and weight.data_type == onnx.TensorProto.INT8
        )
    ]
    if quantized_moe_nodes and len(quantized_moe_nodes) != len(moe_nodes):
        raise OnnxExportError(
            "MoeExpert quantization coverage is inconsistent"
        )
    if quantized_moe_nodes:
        result.add("moe")
    attention_quantization_inputs = (
        AttentionInput.DEQUANT_SCALE1,
        AttentionInput.QUANT_SCALE1,
        AttentionInput.DEQUANT_SCALE2,
        AttentionInput.QUANT_SCALE2,
        AttentionInput.QUANT_OFFSET2,
        AttentionInput.ANTIQUANT_SCALE,
        AttentionInput.ANTIQUANT_OFFSET,
        AttentionInput.KEY_ANTIQUANT_SCALE,
        AttentionInput.KEY_ANTIQUANT_OFFSET,
        AttentionInput.VALUE_ANTIQUANT_SCALE,
        AttentionInput.VALUE_ANTIQUANT_OFFSET,
        AttentionInput.KEY_ROPE_ANTIQUANT_SCALE,
        AttentionInput.DEQUANT_SCALE_QUERY,
    )
    attention_nodes = [
        node
        for node in model.graph.node
        if node.op_type == "FusedInferAttentionScore"
    ]
    quantized_attention_nodes = [
        node
        for node in attention_nodes
        if any(
            index < len(node.input) and bool(node.input[index])
            for index in attention_quantization_inputs
        )
    ]
    if (
        quantized_attention_nodes
        and len(quantized_attention_nodes) != len(attention_nodes)
    ):
        raise OnnxExportError(
            "Attention quantization coverage is inconsistent"
        )
    if quantized_attention_nodes:
        result.add("attention")
    for node in model.graph.node:
        if node.op_type != "AscendDequant" or not node.input:
            continue
        accumulator = producers.get(node.input[0])
        if (
            accumulator is None
            or accumulator.op_type != "MatMul"
            or not accumulator.input
        ):
            continue
        quantizer = producers.get(accumulator.input[0])
        if (
            quantizer is not None
            and quantizer.op_type == "NPUAscendQuantV2"
        ):
            result.add("linear")
    return frozenset(result)


def validate_graph_topology(
    model: onnx.ModelProto,
    mask_mode: str,
) -> Counter[str]:
    """Validate domains, ordering, SSA, outputs, and custom nodes."""
    input_names = [item.name for item in model.graph.input]
    output_names = [item.name for item in model.graph.output]
    if (
        len(input_names) != len(set(input_names))
        or len(output_names) != len(set(output_names))
    ):
        raise OnnxExportError(
            "ONNX graph I/O names must be unique"
        )
    initializer_names = {
        item.name for item in model.graph.initializer
    }
    if initializer_names.intersection(output_names):
        raise OnnxExportError(
            "Graph outputs must not be initializer placeholders"
        )

    known = set(input_names) | initializer_names
    produced: set[str] = set()
    output_producers: dict[str, str] = {}
    for node in model.graph.node:
        if node.domain not in STANDARD_DOMAINS:
            raise OnnxExportError(
                f"Node {node.name!r} uses forbidden domain"
            )
        if node.op_type in {
            "QuantizeLinear",
            "DequantizeLinear",
        }:
            raise OnnxExportError(
                "MDC ONNX must not contain QDQ nodes"
            )
        missing = [
            name
            for name in node.input
            if name and name not in known
        ]
        if missing:
            raise OnnxExportError(
                f"Node {node.name!r} is not topologically "
                f"sorted: {missing}"
            )
        for name in node.output:
            if not name or name in known or name in produced:
                raise OnnxExportError(
                    f"ONNX SSA violation at output {name!r}"
                )
            produced.add(name)
            known.add(name)
            output_producers[name] = node.op_type
        if node.op_type in CUSTOM_OPS:
            validate_operator(node, mask_mode)
    missing_outputs = [
        name for name in output_names if name not in produced
    ]
    if missing_outputs:
        raise OnnxExportError(
            "Graph outputs lack numerical producers: "
            f"{missing_outputs}"
        )
    if any(
        output_producers[name]
        in {"Constant", "ConstantOfShape"}
        for name in output_names
    ):
        raise OnnxExportError(
            "Graph outputs must not be constant placeholders"
        )
    return Counter(node.op_type for node in model.graph.node)


def validate_custom_node_reachability(
    model: onnx.ModelProto,
    properties: dict[str, str],
) -> None:
    """Require custom nodes to reach outputs and cover linear targets."""
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

    custom_nodes = [
        node
        for node in model.graph.node
        if node.op_type in CUSTOM_OPS
    ]
    isolated = [
        node.name or node.op_type
        for node in custom_nodes
        if not any(
            output in visited_values for output in node.output
        )
    ]
    if isolated:
        raise OnnxExportError(
            "MDC custom nodes do not reach graph outputs: "
            f"{isolated}"
        )

    quantized_nodes = [
        node
        for node in custom_nodes
        if node.op_type in {
            "NPUAscendQuantV2",
            "AscendDequant",
        }
    ]
    targets = set(properties["mdc.target"].split(","))
    if targets != {"linear"}:
        return
    raw_count = properties.get("mdc.linear.target_count")
    try:
        target_count = int(raw_count or "")
    except ValueError as error:
        raise OnnxExportError(
            "Linear target count metadata is invalid"
        ) from error
    counts = Counter(node.op_type for node in quantized_nodes)
    if (
        target_count <= 0
        or counts["NPUAscendQuantV2"] != target_count
        or counts["AscendDequant"] != target_count
    ):
        raise OnnxExportError(
            "Linear quantization target coverage is incomplete"
        )
