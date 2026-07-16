"""Independent validator for the fixed MDC MoE ONNX contract."""

from __future__ import annotations

import onnx
from onnx import TensorProto, numpy_helper

from ..errors import OnnxExportError
from ..moe_layout import DEFAULT_MOE_LAYOUT
from .model_inspection import static_shape as _shape


def validate_moe_contract(
    model: onnx.ModelProto,
    properties: dict[str, str],
) -> None:
    """Validate MoeExpert tensors, packing, and model metadata."""
    node = next(
        item for item in model.graph.node if item.op_type == "MoeExpert"
    )
    initializers = {
        item.name: item for item in model.graph.initializer
    }
    specs = {
        item.name: (item.type.tensor_type.elem_type, _shape(item))
        for item in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        )
    }
    specs.update(
        (item.name, (item.data_type, tuple(item.dims)))
        for item in model.graph.initializer
    )
    expected_types = (
        TensorProto.INT8,
        TensorProto.INT16,
        TensorProto.FLOAT16,
        TensorProto.INT8,
        TensorProto.FLOAT,
        TensorProto.INT32,
    )
    if len(node.input) != 6 or any(
        specs.get(name, (None, ()))[0] != expected
        for name, expected in zip(node.input, expected_types, strict=True)
    ):
        raise OnnxExportError(
            "MoeExpert input dtypes must match the six-input ATC ABI"
        )
    x_shape = specs[node.input[0]][1]
    ids_shape = specs[node.input[1]][1]
    weights_shape = specs[node.input[2]][1]
    output_shape = specs.get(node.output[0])
    if len(x_shape) != 2 or x_shape[1] % 256 != 0:
        raise OnnxExportError(
            "MoeExpert hidden_size must use the "
            "ATC-verified 256-element alignment"
        )
    token_count, hidden_size = x_shape
    if (
        ids_shape != DEFAULT_MOE_LAYOUT.routing_shape(token_count)
        or weights_shape != ids_shape
    ):
        raise OnnxExportError(
            "MoeExpert routing inputs must use matching "
            f"[tokenNum, {DEFAULT_MOE_LAYOUT.route_width}] shapes"
        )
    if output_shape != (TensorProto.FLOAT16, x_shape):
        raise OnnxExportError(
            "MoeExpert output must use FLOAT16[tokenNum, hiddenSize]"
        )

    packed = initializers.get(node.input[3])
    scales = initializers.get(node.input[4])
    offsets = initializers.get(node.input[5])
    if packed is None or len(packed.dims) != 1:
        raise OnnxExportError(
            "MoeExpert expert_weights must be a one-dimensional initializer"
        )
    denominator = DEFAULT_MOE_LAYOUT.packed_projection_count * hidden_size
    if packed.dims[0] % denominator:
        raise OnnxExportError(
            "MoeExpert expert_weights packed length is invalid"
        )
    intermediate_size = packed.dims[0] // denominator
    if intermediate_size <= 0 or intermediate_size % 128 != 0:
        raise OnnxExportError(
            "MoeExpert intermediate_size must use the "
            "ATC-verified 128-element alignment"
        )
    if (
        scales is None
        or scales.data_type != TensorProto.FLOAT
        or tuple(scales.dims)
        != (DEFAULT_MOE_LAYOUT.quant_parameter_count,)
    ):
        raise OnnxExportError(
            "MoeExpert quant_scales must be "
            f"FLOAT32[{DEFAULT_MOE_LAYOUT.quant_parameter_count}]"
        )
    if (
        offsets is None
        or offsets.data_type != TensorProto.INT32
        or tuple(offsets.dims)
        != (DEFAULT_MOE_LAYOUT.quant_parameter_count,)
    ):
        raise OnnxExportError(
            "MoeExpert quant_offsets must be "
            f"INT32[{DEFAULT_MOE_LAYOUT.quant_parameter_count}]"
        )
    scale_values = numpy_helper.to_array(scales).astype(
        "float32",
        copy=False,
    )
    if not ((scale_values > 0) & (scale_values < float("inf"))).all():
        raise OnnxExportError(
            "MoeExpert quant_scales must contain finite positive values"
        )
    offset_values = numpy_helper.to_array(offsets).astype(
        "int64",
        copy=False,
    )
    if ((offset_values < -128) | (offset_values > 127)).any():
        raise OnnxExportError(
            "MoeExpert quant_offsets must fit signed INT8"
        )
    required = {
        "mdc.moe.expert_order",
        "mdc.moe.weight_projection_order",
        "mdc.moe.weight_offsets",
        "mdc.moe.weight_lengths",
        "mdc.moe.hidden_size",
        "mdc.moe.intermediate_size",
        "mdc.moe.quant_parameter_count",
        "mdc.moe.quant_parameter_order",
    }
    if (
        required - properties.keys()
        or properties["mdc.moe.quant_parameter_count"]
        != str(DEFAULT_MOE_LAYOUT.quant_parameter_count)
    ):
        raise OnnxExportError("MoE packing metadata is incomplete")
    expected_expert_order = ",".join(DEFAULT_MOE_LAYOUT.expert_order())
    if (
        properties["mdc.moe.expert_order"] != expected_expert_order
        or properties["mdc.moe.weight_projection_order"]
        != ",".join(DEFAULT_MOE_LAYOUT.projections)
        or properties["mdc.moe.quant_parameter_order"]
        != ",".join(DEFAULT_MOE_LAYOUT.quant_parameter_order())
    ):
        raise OnnxExportError("MoE packing metadata order is invalid")
    try:
        metadata_offsets = tuple(
            int(item)
            for item in properties["mdc.moe.weight_offsets"].split(",")
        )
        metadata_lengths = tuple(
            int(item)
            for item in properties["mdc.moe.weight_lengths"].split(",")
        )
        metadata_hidden = int(properties["mdc.moe.hidden_size"])
        metadata_intermediate = int(
            properties["mdc.moe.intermediate_size"]
        )
    except ValueError as error:
        raise OnnxExportError(
            "MoE packing metadata is invalid"
        ) from error
    segments = DEFAULT_MOE_LAYOUT.weight_segments(
        hidden_size,
        intermediate_size,
    )
    if (
        metadata_offsets != tuple(item.offset for item in segments)
        or metadata_lengths != tuple(item.length for item in segments)
        or metadata_hidden != hidden_size
        or metadata_intermediate != intermediate_size
    ):
        raise OnnxExportError(
            "MoE packing metadata does not match tensor shapes"
        )
