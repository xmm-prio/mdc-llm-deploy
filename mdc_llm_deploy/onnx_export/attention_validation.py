"""Independent validator for the MDC attention ONNX contract."""

from __future__ import annotations

import math

import onnx
from onnx import TensorProto, numpy_helper

from ..attention_layout import (
    ATTENTION_INPUT_COUNT,
    ATTENTION_OUTPUT_COUNT,
    RELEASE_ATTENTION_ATTRIBUTES,
    AttentionInput,
)
from ..errors import OnnxExportError
from .model_inspection import require_attributes
from .model_inspection import static_shape as _shape


def validate_attention_operator(
    node: onnx.NodeProto,
    mask_mode: str,
) -> None:
    """Validate the fixed node ABI and release attributes."""
    if (
        len(node.input) != ATTENTION_INPUT_COUNT
        or len(node.output) != ATTENTION_OUTPUT_COUNT
    ):
        raise OnnxExportError(
            "FusedInferAttentionScore must use the complete 29-slot ABI"
        )
    expected_attribute_types = {
        "num_heads": onnx.AttributeProto.INT,
        "num_key_value_heads": onnx.AttributeProto.INT,
        "scale": onnx.AttributeProto.FLOAT,
        **{
            name: (
                onnx.AttributeProto.STRING
                if isinstance(value, str)
                else onnx.AttributeProto.INT
            )
            for name, value in RELEASE_ATTENTION_ATTRIBUTES.items()
        },
    }
    attributes = require_attributes(
        node,
        expected_attribute_types,
    )
    for name, expected in RELEASE_ATTENTION_ATTRIBUTES.items():
        actual = onnx.helper.get_attribute_value(attributes[name])
        normalized = expected.encode() if isinstance(expected, str) else expected
        if actual != normalized:
            raise OnnxExportError(
                f"FusedInferAttentionScore {name} must equal {expected!r}"
            )
    num_heads = int(onnx.helper.get_attribute_value(attributes["num_heads"]))
    kv_heads = int(
        onnx.helper.get_attribute_value(attributes["num_key_value_heads"])
    )
    scale = float(onnx.helper.get_attribute_value(attributes["scale"]))
    if num_heads <= 0 or kv_heads <= 0 or num_heads % kv_heads:
        raise OnnxExportError(
            "FusedInferAttentionScore head counts are invalid"
        )
    if not math.isfinite(scale) or scale <= 0:
        raise OnnxExportError(
            "FusedInferAttentionScore scale must be finite and positive"
        )
    if any(
        not node.input[index]
        for index in (
            AttentionInput.QUERY,
            AttentionInput.KEY,
            AttentionInput.VALUE,
        )
    ):
        raise OnnxExportError(
            "FusedInferAttentionScore requires query, key, and value"
        )
    if mask_mode == "masked" and not node.input[AttentionInput.ATTEN_MASK]:
        raise OnnxExportError("Masked attention requires atten_mask")
    if mask_mode == "maskless" and node.input[AttentionInput.ATTEN_MASK]:
        raise OnnxExportError("Maskless attention must omit atten_mask")


def _validate_mask_initializer(
    model: onnx.ModelProto,
    attention: onnx.NodeProto,
    stage: str,
) -> None:
    initializers = {item.name: item for item in model.graph.initializer}
    mask = initializers.get(attention.input[AttentionInput.ATTEN_MASK])
    if mask is None or mask.data_type != TensorProto.BOOL:
        raise OnnxExportError("Attention mask must be a BOOL initializer")
    cache_shape = _shape(model.graph.output[1])
    query_length = _shape(model.graph.output[0])[1]
    expected = (1, 1, query_length, cache_shape[2])
    if tuple(mask.dims) != expected:
        raise OnnxExportError(f"{stage} attention mask shape must be {expected}")


def _validate_quantization_contract(
    model: onnx.ModelProto,
    attention: onnx.NodeProto,
) -> None:
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
    initializers = {item.name: item for item in model.graph.initializer}
    input_types = {
        slot: specs[attention.input[slot]][0]
        for slot in (
            AttentionInput.QUERY,
            AttentionInput.KEY,
            AttentionInput.VALUE,
        )
    }
    for slot in input_types:
        if len(specs[attention.input[slot]][1]) != 4:
            raise OnnxExportError("Attention Q/K/V inputs must use rank 4")
    if input_types[AttentionInput.KEY] != input_types[AttentionInput.VALUE]:
        raise OnnxExportError("Attention K/V inputs must use one dtype")
    lse_spec = specs.get(attention.output[1])
    if lse_spec != (TensorProto.FLOAT, (1,)):
        raise OnnxExportError(
            "Disabled attention LSE output must use FLOAT32[1]"
        )

    def validate_scale(
        slot: AttentionInput,
        name: str,
        required: bool,
    ) -> None:
        input_name = attention.input[slot]
        if not input_name and not required:
            return
        initializer = initializers.get(input_name)
        if not input_name or initializer is None:
            raise OnnxExportError(f"Attention quantization requires {name}")
        if (
            initializer.data_type != TensorProto.FLOAT
            or tuple(initializer.dims) not in {(), (1,)}
        ):
            raise OnnxExportError(
                f"Attention {name} must be a one-element FLOAT32 initializer"
            )
        value = float(numpy_helper.to_array(initializer).reshape(-1)[0])
        if not math.isfinite(value) or value <= 0:
            raise OnnxExportError(
                f"Attention {name} must be finite and positive"
            )

    def validate_offset(
        slot: AttentionInput,
        scale_slot: AttentionInput,
        name: str,
    ) -> None:
        input_name = attention.input[slot]
        if not input_name:
            return
        initializer = initializers.get(input_name)
        if not attention.input[scale_slot] or initializer is None:
            raise OnnxExportError(f"Attention {name} requires its scale")
        if (
            initializer.data_type != TensorProto.INT32
            or tuple(initializer.dims) not in {(), (1,)}
        ):
            raise OnnxExportError(
                f"Attention {name} must be a one-element INT32 initializer"
            )

    query_int8 = input_types[AttentionInput.QUERY] == TensorProto.INT8
    key_int8 = input_types[AttentionInput.KEY] == TensorProto.INT8
    value_int8 = input_types[AttentionInput.VALUE] == TensorProto.INT8
    for is_int8, scale_slot, offset_slot, name in (
        (
            query_int8,
            AttentionInput.DEQUANT_SCALE_QUERY,
            None,
            "dequant_scale_query",
        ),
        (
            key_int8,
            AttentionInput.KEY_ANTIQUANT_SCALE,
            AttentionInput.KEY_ANTIQUANT_OFFSET,
            "key_antiquant_scale",
        ),
        (
            value_int8,
            AttentionInput.VALUE_ANTIQUANT_SCALE,
            AttentionInput.VALUE_ANTIQUANT_OFFSET,
            "value_antiquant_scale",
        ),
    ):
        if not is_int8 and attention.input[scale_slot]:
            raise OnnxExportError(
                f"Floating attention input must not provide {name}"
            )
        validate_scale(scale_slot, name, required=is_int8)
        if offset_slot is not None:
            if not is_int8 and attention.input[offset_slot]:
                raise OnnxExportError(
                    "Floating attention input must not provide antiquant offset"
                )
            validate_offset(offset_slot, scale_slot, f"{name}_offset")

    all_int8 = query_int8 and key_int8 and value_int8
    for slot, name in (
        (AttentionInput.DEQUANT_SCALE1, "dequant_scale1"),
        (AttentionInput.DEQUANT_SCALE2, "dequant_scale2"),
    ):
        if not all_int8 and attention.input[slot]:
            raise OnnxExportError(
                f"Non-INT8 attention must not provide {name}"
            )
        validate_scale(slot, name, required=all_int8)
    validate_scale(
        AttentionInput.QUANT_SCALE1,
        "quant_scale1",
        required=all_int8,
    )


def validate_attention_contract(
    model: onnx.ModelProto,
    attention: onnx.NodeProto,
    *,
    mask_mode: str,
    stage: str,
) -> None:
    """Validate attention tensor, quantization, and mask contracts."""
    _validate_quantization_contract(model, attention)
    if mask_mode == "masked":
        _validate_mask_initializer(model, attention, stage)
