"""MoE lowering into the fixed MDC ONNX release ABI."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..errors import OnnxExportError
from ..graph_types import GraphMetadata, QuantizedTarget
from ..model_properties import MoeDimensions
from ..moe_layout import DEFAULT_MOE_LAYOUT
from ..quantization_properties import (
    ActivationQuantizationParameters,
)
from .graph_cleanup import producer_map
from .lowering_support import (
    FLOAT_ONNX_DTYPES,
    activation_target,
    append_quant,
    append_value,
    initializer,
    model_types,
    unique_name,
)


def _moe_targets(value: GraphMetadata) -> tuple[QuantizedTarget, ...]:
    return tuple(item for item in value.quantized_targets if item.target_type == "moe")


def _moe_requested(value: GraphMetadata) -> bool:
    return value.model_kind == "moe" and bool(_moe_targets(value))


def _required_activation_target(
    value: GraphMetadata,
    target: QuantizedTarget,
) -> QuantizedTarget:
    try:
        qparams = ActivationQuantizationParameters.for_target(
            value.properties,
            target.fqn,
        )
    except ValueError as error:
        raise OnnxExportError(str(error)) from error
    if qparams is None:
        raise OnnxExportError(f"MoE target {target.fqn!r} lacks activation qparams")
    if (
        qparams.bits != 8
        or qparams.granularity != "per_tensor"
        or qparams.mode != "static"
        or len(qparams.scale) != 1
        or len(qparams.zero_point) != 1
    ):
        raise OnnxExportError(
            f"MoE target {target.fqn!r} requires scalar static INT8 activation qparams"
        )
    return activation_target(value, target)


def _layout_dimensions(value: GraphMetadata) -> tuple[int, int]:
    try:
        dimensions = MoeDimensions.from_properties(
            value.properties
        )
    except ValueError as error:
        raise OnnxExportError(str(error)) from error
    expected = {
        "num_experts": (
            dimensions.routed_expert_count,
            DEFAULT_MOE_LAYOUT.routed_expert_count,
        ),
        "num_shared_experts": (
            dimensions.shared_expert_count,
            DEFAULT_MOE_LAYOUT.shared_expert_count,
        ),
        "num_experts_per_tok": (
            dimensions.routed_top_k,
            DEFAULT_MOE_LAYOUT.routed_top_k,
        ),
    }
    for name, (actual, expected_value) in expected.items():
        if actual != expected_value:
            raise OnnxExportError(f"MoE model property {name!r} must equal {expected_value}")
    return dimensions.hidden_size, dimensions.intermediate_size


def append_moe(model: onnx.ModelProto, value: GraphMetadata) -> bool:
    """Append fixed-ABI MoE lowering when requested by graph metadata."""
    targets = _moe_targets(value)
    if not _moe_requested(value):
        return False
    hidden_size, intermediate_size = _layout_dimensions(value)
    post_attention_norm = next(
        (node for node in model.graph.node if node.name == "mdc.rms_norm.post_attention_norm"),
        None,
    )
    final_norm = next(
        (node for node in model.graph.node if node.name == "mdc.rms_norm.final_norm"),
        None,
    )
    if post_attention_norm is None or final_norm is None:
        raise OnnxExportError("Cannot locate MoE normalization boundaries")
    source = post_attention_norm.output[0]
    source_type = model_types(model).get(source)
    if source_type is None:
        raise OnnxExportError("MoE source lacks static type metadata")
    _, source_shape = source_type
    if len(source_shape) != 3 or source_shape[-1] != hidden_size:
        raise OnnxExportError("MoE source shape is invalid")
    final_residual = producer_map(model).get(final_norm.input[0])
    residual_input = post_attention_norm.input[0]
    if (
        final_residual is None
        or final_residual.op_type != "Add"
        or residual_input not in final_residual.input
    ):
        raise OnnxExportError("Cannot locate MoE residual merge")
    body_indices = [
        index
        for index, input_name in enumerate(final_residual.input)
        if input_name != residual_input
    ]
    if len(body_indices) != 1:
        raise OnnxExportError("MoE residual merge has an invalid ABI")
    token_count = int(np.prod(source_shape[:-1]))
    input_activations = tuple(
        _required_activation_target(value, target)
        for target in targets
        if DEFAULT_MOE_LAYOUT.projection_for_fqn(target.fqn)
        in DEFAULT_MOE_LAYOUT.input_activation_projections
    )
    if not input_activations:
        raise OnnxExportError("MoE input activation qparams are missing")
    activation = input_activations[0]
    activation_contract = (
        activation.bits,
        activation.granularity,
        activation.symmetric,
        activation.scale,
        activation.zero_point,
    )
    if any(
        (
            item.bits,
            item.granularity,
            item.symmetric,
            item.scale,
            item.zero_point,
        )
        != activation_contract
        for item in input_activations[1:]
    ):
        raise OnnxExportError("MoE gate/up input activation qparams must match")
    quantized = append_quant(model, source, source_shape, activation, "mdc.moe.quant")
    flattened = unique_name(model, "mdc.moe.input")
    reshape_name = unique_name(model, "mdc.moe.input_shape")
    model.graph.initializer.append(
        initializer(
            reshape_name,
            np.asarray([token_count, hidden_size], dtype=np.int64),
        )
    )
    model.graph.node.append(
        helper.make_node(
            "Reshape",
            [quantized, reshape_name],
            [flattened],
            name="mdc.moe.reshape",
        )
    )
    append_value(
        model,
        flattened,
        TensorProto.INT8,
        (token_count, hidden_size),
    )
    topk = next(
        (node for node in model.graph.node if node.op_type == "TopK"),
        None,
    )
    if topk is None or len(topk.output) != 2:
        raise OnnxExportError("Cannot locate MoE router TopK outputs")
    routed_values, routed_ids = topk.output
    routed_ids_i16 = unique_name(model, "mdc.moe.routed_ids")
    routed_values_fp32 = unique_name(model, "mdc.moe.routed_weights_fp32")
    model.graph.node.extend(
        [
            helper.make_node(
                "Cast",
                [routed_ids],
                [routed_ids_i16],
                name="mdc.moe.ids_cast",
                to=TensorProto.INT16,
            ),
            helper.make_node(
                "Cast",
                [routed_values],
                [routed_values_fp32],
                name="mdc.moe.weights_cast",
                to=TensorProto.FLOAT,
            ),
        ]
    )
    routed_shape_name = unique_name(model, "mdc.moe.routed_shape")
    axes_name = unique_name(model, "mdc.moe.reduce_axes")
    shared_ids_name = unique_name(model, "mdc.moe.shared_ids")
    shared_weights_name = unique_name(model, "mdc.moe.shared_weights")
    model.graph.initializer.extend(
        [
            initializer(
                routed_shape_name,
                np.asarray(
                    [token_count, DEFAULT_MOE_LAYOUT.routed_top_k],
                    dtype=np.int64,
                ),
            ),
            initializer(axes_name, np.asarray([-1], dtype=np.int64)),
            initializer(
                shared_ids_name,
                np.full(
                    (token_count, 1),
                    DEFAULT_MOE_LAYOUT.shared_expert_id,
                    dtype=np.int16,
                ),
            ),
            initializer(
                shared_weights_name,
                np.ones((token_count, 1), dtype=np.float16),
            ),
        ]
    )
    ids_2d = unique_name(model, "mdc.moe.routed_ids_2d")
    weights_2d = unique_name(model, "mdc.moe.routed_weights_2d")
    weight_sum = unique_name(model, "mdc.moe.routed_weight_sum")
    normalized = unique_name(model, "mdc.moe.normalized_weights")
    normalized_fp16 = unique_name(model, "mdc.moe.normalized_weights_fp16")
    ids_name = unique_name(model, "mdc.moe.topk_ids")
    weights_name = unique_name(model, "mdc.moe.topk_weight")
    model.graph.node.extend(
        [
            helper.make_node("Reshape", [routed_ids_i16, routed_shape_name], [ids_2d]),
            helper.make_node(
                "Reshape",
                [routed_values_fp32, routed_shape_name],
                [weights_2d],
            ),
            helper.make_node(
                "ReduceSum",
                [weights_2d, axes_name],
                [weight_sum],
                keepdims=1,
            ),
            helper.make_node("Div", [weights_2d, weight_sum], [normalized]),
            helper.make_node(
                "Cast",
                [normalized],
                [normalized_fp16],
                to=TensorProto.FLOAT16,
            ),
            helper.make_node(
                "Concat",
                [ids_2d, shared_ids_name],
                [ids_name],
                axis=1,
            ),
            helper.make_node(
                "Concat",
                [normalized_fp16, shared_weights_name],
                [weights_name],
                axis=1,
            ),
        ]
    )
    route_shape = DEFAULT_MOE_LAYOUT.routing_shape(token_count)
    append_value(model, ids_name, TensorProto.INT16, route_shape)
    append_value(model, weights_name, TensorProto.FLOAT16, route_shape)

    initializers = {
        item.name: item
        for item in model.graph.initializer
        if len(item.dims) == 2 and item.data_type in FLOAT_ONNX_DTYPES
    }
    target_by_fqn = {item.fqn: item for item in targets}
    packed_parts: list[np.ndarray] = []
    scales = np.ones(
        DEFAULT_MOE_LAYOUT.quant_parameter_count,
        dtype=np.float32,
    )
    offsets = np.zeros(
        DEFAULT_MOE_LAYOUT.quant_parameter_count,
        dtype=np.int32,
    )
    scales[0] = float(activation.scale[0])
    offsets[0] = int(activation.zero_point[0])
    for segment in DEFAULT_MOE_LAYOUT.weight_segments(
        hidden_size,
        intermediate_size,
    ):
        expert_name = (
            f"experts.{segment.expert_id}"
            if segment.expert_id < DEFAULT_MOE_LAYOUT.routed_expert_count
            else "shared_expert"
        )
        weight = next(
            (
                item
                for name, item in initializers.items()
                if expert_name in name and segment.projection in name
            ),
            None,
        )
        if weight is None:
            raise OnnxExportError(f"Cannot locate MoE weight {expert_name}.{segment.projection}")
        target = next(
            (
                item
                for fqn, item in target_by_fqn.items()
                if expert_name in fqn and segment.projection in fqn
            ),
            None,
        )
        if target is None:
            raise OnnxExportError(
                f"Cannot locate MoE quantization target {expert_name}.{segment.projection}"
            )
        scale_index = DEFAULT_MOE_LAYOUT.scale_index(
            segment.expert_id,
            DEFAULT_MOE_LAYOUT.quant_slot_for_projection(
                segment.projection
            ),
        )
        scale = float(target.scale[0])
        zero_point = int(target.zero_point[0])
        array = numpy_helper.to_array(weight).astype(np.float32).T
        if array.shape != (segment.rows, segment.columns):
            raise OnnxExportError(f"MoE weight {expert_name}.{segment.projection} shape is invalid")
        packed_part = (
            np.clip(
                np.rint(array / scale) + zero_point,
                -128,
                127,
            )
            .astype(np.int8)
            .reshape(-1)
        )
        if packed_part.size != segment.length:
            raise OnnxExportError(
                f"MoE weight {expert_name}.{segment.projection} length is invalid"
            )
        packed_parts.append(packed_part)
        scales[scale_index] = scale
        offsets[scale_index] = zero_point
        if segment.projection == DEFAULT_MOE_LAYOUT.output_projection:
            intermediate = _required_activation_target(value, target)
            activation_index = DEFAULT_MOE_LAYOUT.scale_index(
                segment.expert_id,
                "intermediate",
            )
            scales[activation_index] = float(intermediate.scale[0])
            offsets[activation_index] = int(intermediate.zero_point[0])
    packed = np.concatenate(packed_parts)
    if packed.size != DEFAULT_MOE_LAYOUT.packed_weight_length(
        hidden_size,
        intermediate_size,
    ):
        raise OnnxExportError("Packed MoE weight length does not match the fixed ABI")
    packed_name = unique_name(model, "mdc.moe.expert_weights")
    scales_name = unique_name(model, "mdc.moe.quant_scales")
    offsets_name = unique_name(model, "mdc.moe.quant_offsets")
    model.graph.initializer.extend(
        [
            initializer(packed_name, packed),
            initializer(scales_name, scales),
            initializer(offsets_name, offsets),
        ]
    )
    output = unique_name(model, "mdc.moe.output")
    model.graph.node.append(
        helper.make_node(
            "MoeExpert",
            [
                flattened,
                ids_name,
                weights_name,
                packed_name,
                scales_name,
                offsets_name,
            ],
            [output],
            name="mdc.moe",
        )
    )
    append_value(
        model,
        output,
        TensorProto.FLOAT16,
        (token_count, hidden_size),
    )
    output_shape_name = unique_name(model, "mdc.moe.output_shape")
    output_3d = unique_name(model, "mdc.moe.output_3d")
    model.graph.initializer.append(
        initializer(
            output_shape_name,
            np.asarray(source_shape, dtype=np.int64),
        )
    )
    model.graph.node.append(
        helper.make_node(
            "Reshape",
            [output, output_shape_name],
            [output_3d],
            name="mdc.moe.output_reshape",
        )
    )
    append_value(model, output_3d, TensorProto.FLOAT16, source_shape)
    final_residual.input[body_indices[0]] = output_3d
    return True


def moe_metadata_properties(value: GraphMetadata) -> dict[str, str]:
    """Build fixed-ABI MoE metadata when MoE lowering is requested."""
    if not _moe_requested(value):
        return {}
    hidden_size, intermediate_size = _layout_dimensions(value)
    segments = DEFAULT_MOE_LAYOUT.weight_segments(
        hidden_size,
        intermediate_size,
    )
    return {
        "mdc.moe.expert_order": ",".join(DEFAULT_MOE_LAYOUT.expert_order()),
        "mdc.moe.weight_projection_order": ",".join(DEFAULT_MOE_LAYOUT.projections),
        "mdc.moe.weight_offsets": ",".join(str(item.offset) for item in segments),
        "mdc.moe.weight_lengths": ",".join(str(item.length) for item in segments),
        "mdc.moe.hidden_size": str(hidden_size),
        "mdc.moe.intermediate_size": str(intermediate_size),
        "mdc.moe.quant_parameter_count": str(DEFAULT_MOE_LAYOUT.quant_parameter_count),
        "mdc.moe.quant_parameter_order": ",".join(DEFAULT_MOE_LAYOUT.quant_parameter_order()),
    }
