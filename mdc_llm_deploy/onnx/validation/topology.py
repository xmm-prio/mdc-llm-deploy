"""Topology and target-coverage checks for MDC custom ONNX nodes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import onnx
from onnx import TensorProto, numpy_helper

from ...errors import OnnxExportError
from ...operators.contracts.attention import AttentionInput
from ...operators.contracts.onnx import MDC_ONNX_DOMAIN
from ...operators.contracts.schema import OPERATOR_SCHEMAS
from .metadata import ValidatedMetadata
from .operator import validate_dequant_initializers, validate_operator

STANDARD_DOMAINS = frozenset({"", MDC_ONNX_DOMAIN})
CUSTOM_OPS = frozenset(
    schema.onnx_name for schema in OPERATOR_SCHEMAS.values()
)


@dataclass(frozen=True, slots=True)
class _QuantizedTargetScan:
    initializers: Mapping[str, onnx.TensorProto]
    producers: Mapping[str, onnx.NodeProto]
    moe_nodes: tuple[onnx.NodeProto, ...]
    attention_nodes: tuple[onnx.NodeProto, ...]
    dequant_nodes: tuple[onnx.NodeProto, ...]

    @classmethod
    def collect(cls, model: onnx.ModelProto) -> _QuantizedTargetScan:
        """Collect immutable inputs for quantized target classification."""
        initializers: dict[str, onnx.TensorProto] = {}
        for initializer in model.graph.initializer:
            initializers[initializer.name] = initializer

        producers: dict[str, onnx.NodeProto] = {}
        moe_nodes: list[onnx.NodeProto] = []
        attention_nodes: list[onnx.NodeProto] = []
        dequant_nodes: list[onnx.NodeProto] = []
        candidates = {
            "MoeExpert": moe_nodes,
            "FusedInferAttentionScore": attention_nodes,
            "AscendDequant": dequant_nodes,
        }
        for node in model.graph.node:
            for output in node.output:
                if output:
                    producers[output] = node
            candidate_nodes = candidates.get(node.op_type)
            if candidate_nodes is not None:
                candidate_nodes.append(node)

        return cls(
            initializers=MappingProxyType(initializers),
            producers=MappingProxyType(producers),
            moe_nodes=tuple(moe_nodes),
            attention_nodes=tuple(attention_nodes),
            dequant_nodes=tuple(dequant_nodes),
        )


@dataclass(frozen=True, slots=True)
class QuantizationTopologyEvidence:
    """Immutable results from centralized MDC topology validation."""

    operator_counts: tuple[tuple[str, int], ...]
    observed_quantized_targets: frozenset[str]


def _attribute_int(node: onnx.NodeProto, name: str) -> int:
    attributes = {item.name: item for item in node.attribute}
    return int(onnx.helper.get_attribute_value(attributes[name]))


def _initializer_key(
    initializer: onnx.TensorProto,
) -> tuple[int, tuple[int, ...], bytes]:
    try:
        content = numpy_helper.to_array(initializer).tobytes()
    except Exception as error:
        raise OnnxExportError(
            f"Cannot read quantization initializer {initializer.name!r}"
        ) from error
    return initializer.data_type, tuple(initializer.dims), content


def _validate_unique_quantizers(
    model: onnx.ModelProto,
    scan: _QuantizedTargetScan,
) -> None:
    signatures: set[
        tuple[
            str,
            tuple[int, tuple[int, ...], bytes],
            tuple[int, tuple[int, ...], bytes] | None,
            int,
            int,
        ]
    ] = set()
    for node in model.graph.node:
        if node.op_type != "NPUAscendQuantV2":
            continue
        scale = scan.initializers.get(node.input[1])
        offset = (
            scan.initializers.get(node.input[2])
            if len(node.input) == 3 and node.input[2]
            else None
        )
        if scale is None or (
            len(node.input) == 3 and node.input[2] and offset is None
        ):
            raise OnnxExportError(
                "NPUAscendQuantV2 parameters must be initializers"
            )
        signature = (
            node.input[0],
            _initializer_key(scale),
            _initializer_key(offset) if offset is not None else None,
            _attribute_int(node, "axis"),
            _attribute_int(node, "dtype"),
        )
        if signature in signatures:
            raise OnnxExportError(
                "Equivalent NPUAscendQuantV2 nodes must be shared"
            )
        signatures.add(signature)


def _validate_quantized_consumers(
    model: onnx.ModelProto,
    scan: _QuantizedTargetScan,
) -> None:
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name:
                consumers.setdefault(input_name, []).append((node, index))
    graph_outputs = {item.name for item in model.graph.output}
    legal_custom_slots = {
        "MoeExpert": {0},
        "FusedInferAttentionScore": {
            int(AttentionInput.QUERY),
            int(AttentionInput.KEY),
            int(AttentionInput.VALUE),
        },
    }
    for quantizer in (
        node
        for node in model.graph.node
        if node.op_type == "NPUAscendQuantV2"
    ):
        output = quantizer.output[0]
        uses = consumers.get(output, [])
        legal_uses = [
            (consumer.op_type == "MatMul" and index == 0)
            or index in legal_custom_slots.get(consumer.op_type, set())
            for consumer, index in uses
        ]
        if (
            (not uses and output not in graph_outputs)
            or not all(legal_uses)
        ):
            raise OnnxExportError(
                f"Quantizer {quantizer.name or output!r} has no legal "
                "quantized consumer"
            )

def _validate_linear_topology(
    scan: _QuantizedTargetScan,
    metadata: ValidatedMetadata,
) -> None:
    for node in scan.dequant_nodes:
        accumulator = (
            scan.producers.get(node.input[0]) if node.input else None
        )
        quantizer = (
            scan.producers.get(accumulator.input[0])
            if accumulator is not None
            and accumulator.op_type == "MatMul"
            and accumulator.input
            else None
        )
        if quantizer is None or quantizer.op_type != "NPUAscendQuantV2":
            raise OnnxExportError(
                "AscendDequant input 0 must come from MatMul input 0 "
                "fed by NPUAscendQuantV2"
            )
    if "linear" not in metadata.targets:
        return
    raw_count = metadata.properties.get("mdc.linear.target_count")
    try:
        target_count = int(raw_count or "")
    except ValueError as error:
        raise OnnxExportError(
            "Linear target count metadata is invalid"
        ) from error
    if target_count <= 0 or len(scan.dequant_nodes) != target_count:
        raise OnnxExportError(
            "Linear quantization target coverage is incomplete"
        )


def _validate_attention_quantized_ports(
    model: onnx.ModelProto,
    attention: onnx.NodeProto,
    scan: _QuantizedTargetScan,
) -> None:
    specs = {
        item.name: item.type.tensor_type.elem_type
        for item in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        )
    }
    input_slots = (
        AttentionInput.QUERY,
        AttentionInput.KEY,
        AttentionInput.VALUE,
    )
    int8_inputs = {
        slot: (
            specs.get(attention.input[slot]) == TensorProto.INT8
            or (
                (producer := scan.producers.get(attention.input[slot]))
                is not None
                and producer.op_type == "NPUAscendQuantV2"
            )
        )
        for slot in input_slots
    }
    supported_quantization_slots = {
        AttentionInput.DEQUANT_SCALE1,
        AttentionInput.QUANT_SCALE1,
        AttentionInput.DEQUANT_SCALE2,
        AttentionInput.KEY_ANTIQUANT_SCALE,
        AttentionInput.KEY_ANTIQUANT_OFFSET,
        AttentionInput.VALUE_ANTIQUANT_SCALE,
        AttentionInput.VALUE_ANTIQUANT_OFFSET,
        AttentionInput.DEQUANT_SCALE_QUERY,
    }
    quantization_slots = {
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
    }
    if any(
        attention.input[slot]
        for slot in quantization_slots - supported_quantization_slots
    ):
        raise OnnxExportError(
            "Attention uses unsupported quantization ports"
        )

    def require_scale(slot: AttentionInput, required: bool) -> None:
        input_name = attention.input[slot]
        if not input_name:
            if required:
                raise OnnxExportError(
                    f"Attention quantization requires {slot.name.lower()}"
                )
            return
        initializer = scan.initializers.get(input_name)
        if (
            initializer is None
            or initializer.data_type != TensorProto.FLOAT
            or tuple(initializer.dims) not in {(), (1,)}
        ):
            raise OnnxExportError(
                f"Attention {slot.name.lower()} must be a one-element "
                "FLOAT32 initializer"
            )
        value = numpy_helper.to_array(initializer).reshape(-1)
        if value.size != 1 or not (0 < float(value[0]) < float("inf")):
            raise OnnxExportError(
                f"Attention {slot.name.lower()} must be finite and positive"
            )

    for input_slot, scale_slot in (
        (AttentionInput.QUERY, AttentionInput.DEQUANT_SCALE_QUERY),
        (AttentionInput.KEY, AttentionInput.KEY_ANTIQUANT_SCALE),
        (AttentionInput.VALUE, AttentionInput.VALUE_ANTIQUANT_SCALE),
    ):
        require_scale(scale_slot, int8_inputs[input_slot])
    require_scale(
        AttentionInput.DEQUANT_SCALE1,
        int8_inputs[AttentionInput.QUERY]
        and int8_inputs[AttentionInput.KEY],
    )
    for slot in (
        AttentionInput.QUANT_SCALE1,
        AttentionInput.DEQUANT_SCALE2,
    ):
        require_scale(slot, False)
    for offset_slot, scale_slot in (
        (
            AttentionInput.KEY_ANTIQUANT_OFFSET,
            AttentionInput.KEY_ANTIQUANT_SCALE,
        ),
        (
            AttentionInput.VALUE_ANTIQUANT_OFFSET,
            AttentionInput.VALUE_ANTIQUANT_SCALE,
        ),
    ):
        offset_name = attention.input[offset_slot]
        if not offset_name:
            continue
        offset = scan.initializers.get(offset_name)
        if (
            not attention.input[scale_slot]
            or offset is None
            or offset.data_type != TensorProto.FLOAT
            or tuple(offset.dims) not in {(), (1,)}
        ):
            raise OnnxExportError(
                "Attention antiquant offset requires a matching scale and "
                "one-element INT32 initializer"
            )


def quantized_target_families(
    model: onnx.ModelProto,
) -> frozenset[str]:
    """Infer quantized target families from validated graph topology."""
    scan = _QuantizedTargetScan.collect(model)
    result: set[str] = set()
    quantized_moe_nodes = [
        node
        for node in scan.moe_nodes
        if (
            len(node.input) > 4
            and bool(node.input[4])
            and (weight := scan.initializers.get(node.input[3])) is not None
            and weight.data_type == onnx.TensorProto.INT8
        )
    ]
    if (
        quantized_moe_nodes
        and len(quantized_moe_nodes) != len(scan.moe_nodes)
    ):
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
    quantized_attention_nodes = [
        node
        for node in scan.attention_nodes
        if any(
            index < len(node.input) and bool(node.input[index])
            for index in attention_quantization_inputs
        )
    ]
    if (
        quantized_attention_nodes
        and len(quantized_attention_nodes) != len(scan.attention_nodes)
    ):
        raise OnnxExportError(
            "Attention quantization coverage is inconsistent"
        )
    if quantized_attention_nodes:
        result.add("attention")
    for node in scan.dequant_nodes:
        if not node.input:
            continue
        accumulator = scan.producers.get(node.input[0])
        if (
            accumulator is None
            or accumulator.op_type != "MatMul"
            or not accumulator.input
        ):
            continue
        quantizer = scan.producers.get(accumulator.input[0])
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

    del properties


def validate_quantization_topology(
    model: onnx.ModelProto,
    metadata: ValidatedMetadata,
) -> QuantizationTopologyEvidence:
    """Validate all topology and quantized-family rules in one pass."""
    counts = validate_graph_topology(model, metadata.mask_mode)
    validate_custom_node_reachability(model, metadata.properties)
    validate_dequant_initializers(model)
    scan = _QuantizedTargetScan.collect(model)
    _validate_unique_quantizers(model, scan)
    _validate_quantized_consumers(model, scan)
    _validate_linear_topology(scan, metadata)
    for attention in scan.attention_nodes:
        _validate_attention_quantized_ports(model, attention, scan)
    declared_targets = metadata.targets
    expected = (
        frozenset()
        if declared_targets == {"fp16"}
        else declared_targets
    )
    observed = (
        frozenset()
        if declared_targets == {"fp16"}
        else quantized_target_families(model)
    )
    if observed != expected:
        raise OnnxExportError(
            "Observed quantized targets do not match MDC metadata"
        )
    return QuantizationTopologyEvidence(
        operator_counts=tuple(sorted(counts.items())),
        observed_quantized_targets=observed,
    )
