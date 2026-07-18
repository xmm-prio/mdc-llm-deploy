"""Node-level validation for MDC custom ONNX operators."""

from __future__ import annotations

import math

import onnx
from onnx import TensorProto, numpy_helper

from ...errors import OnnxExportError
from ...operators.contracts.schema import schema_for_onnx_name
from ..inspection import (
    require_attributes as _require_attributes,
)
from .attention import validate_attention_operator

_ROPE_DTYPES = {
    int(TensorProto.FLOAT16),
    int(TensorProto.FLOAT),
}


def validate_operator(
    node: onnx.NodeProto,
    mask_mode: str,
) -> None:
    """Validate one MDC custom node ABI."""
    if node.op_type == "NPURmsNorm":
        if len(node.input) != 2 or len(node.output) != 2:
            raise OnnxExportError(
                "NPURmsNorm must use 2 inputs and 2 outputs"
            )
        attributes = _require_attributes(
            node,
            {"epsilon": onnx.AttributeProto.FLOAT},
        )
        epsilon = onnx.helper.get_attribute_value(
            attributes["epsilon"]
        )
        expected_epsilon = schema_for_onnx_name(
            node.op_type
        ).attribute_defaults["epsilon"]
        if not math.isclose(
            float(epsilon),
            float(expected_epsilon),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise OnnxExportError(
                "NPURmsNorm epsilon must equal 1e-6"
            )
    elif node.op_type == "ApplyRotaryPosEmb":
        if len(node.input) != 4 or len(node.output) != 2:
            raise OnnxExportError(
                "ApplyRotaryPosEmb must use 4 inputs and 2 outputs"
            )
        attributes = _require_attributes(
            node,
            {
                "layout": onnx.AttributeProto.INT,
                "rotary_mode": onnx.AttributeProto.STRING,
            },
        )
        layout = onnx.helper.get_attribute_value(attributes["layout"])
        rotary_mode = onnx.helper.get_attribute_value(
            attributes["rotary_mode"]
        )
        expected = schema_for_onnx_name(
            node.op_type
        ).attribute_defaults
        if (
            layout != expected["layout"]
            or rotary_mode != str(
                expected["rotary_mode"]
            ).encode()
        ):
            raise OnnxExportError(
                "ApplyRotaryPosEmb must use BSND half rotation"
            )
    elif node.op_type == "FusedInferAttentionScore":
        validate_attention_operator(node, mask_mode)
    elif node.op_type == "NPUAscendQuantV2":
        if len(node.input) not in {2, 3} or len(node.output) != 1:
            raise OnnxExportError(
                "NPUAscendQuantV2 ABI is invalid"
            )
        attributes = _require_attributes(
            node,
            {
                "axis": onnx.AttributeProto.INT,
                "dtype": onnx.AttributeProto.INT,
            },
        )
        if (
            onnx.helper.get_attribute_value(
                attributes["dtype"]
            )
            != schema_for_onnx_name(
                node.op_type
            ).attribute_defaults["dtype"]
        ):
            raise OnnxExportError(
                "Release quantization must use INT8 dtype=2"
            )
    elif node.op_type == "AscendDequant":
        if len(node.input) != 2 or len(node.output) != 1:
            raise OnnxExportError("AscendDequant ABI is invalid")
        attributes = _require_attributes(
            node,
            {
                "sqrt_mode": onnx.AttributeProto.INT,
                "relu_flag": onnx.AttributeProto.INT,
                "dtype": onnx.AttributeProto.INT,
            },
        )
        expected = schema_for_onnx_name(
            node.op_type
        ).attribute_defaults
        if (
            onnx.helper.get_attribute_value(
                attributes["sqrt_mode"]
            )
            != expected["sqrt_mode"]
        ):
            raise OnnxExportError(
                "AscendDequant sqrt_mode must be false"
            )
        if (
            onnx.helper.get_attribute_value(
                attributes["relu_flag"]
            )
            != expected["relu_flag"]
        ):
            raise OnnxExportError(
                "AscendDequant relu_flag must be false"
            )
        if onnx.helper.get_attribute_value(attributes["dtype"]) not in {
            0,
            1,
        }:
            raise OnnxExportError(
                "AscendDequant dtype must be 0 or 1"
            )
    elif node.op_type == "MoeExpert":
        if len(node.input) != 6 or len(node.output) != 1:
            raise OnnxExportError("MoeExpert ABI is invalid")
    else:
        raise OnnxExportError(
            f"No MDC ONNX validator for {node.op_type!r}"
        )


def validate_rope_initializers(model: onnx.ModelProto) -> None:
    """Validate final ApplyRotaryPosEmb initializer bindings and dtypes."""
    initializers = {
        item.name: item for item in model.graph.initializer
    }
    value_dtypes = {
        item.name: int(item.type.tensor_type.elem_type)
        for item in (
            *model.graph.input,
            *model.graph.value_info,
            *model.graph.output,
        )
        if item.type.tensor_type.elem_type
    }
    value_dtypes.update(
        {
            name: int(initializer.data_type)
            for name, initializer in initializers.items()
        }
    )

    for node in model.graph.node:
        if node.op_type != "ApplyRotaryPosEmb":
            continue
        node_name = node.name or "<unnamed>"
        input_dtypes: dict[str, int] = {}
        for index, role in enumerate(("query", "key")):
            dtype = value_dtypes.get(node.input[index])
            if dtype is None:
                raise OnnxExportError(
                    f"ApplyRotaryPosEmb node {node_name!r} {role} "
                    "input dtype is unavailable"
                )
            if dtype not in _ROPE_DTYPES:
                raise OnnxExportError(
                    f"ApplyRotaryPosEmb node {node_name!r} {role} "
                    "input dtype must be FLOAT16 or FLOAT"
                )
            input_dtypes[role] = dtype
        if input_dtypes["key"] != input_dtypes["query"]:
            raise OnnxExportError(
                f"ApplyRotaryPosEmb node {node_name!r} key input "
                "dtype must match query input"
            )

        for index, role in ((2, "cos"), (3, "sin")):
            initializer = initializers.get(node.input[index])
            if initializer is None:
                raise OnnxExportError(
                    f"ApplyRotaryPosEmb node {node_name!r} {role} "
                    "input must directly reference an initializer"
                )
            dtype = int(initializer.data_type)
            if dtype not in _ROPE_DTYPES:
                raise OnnxExportError(
                    f"ApplyRotaryPosEmb node {node_name!r} {role} "
                    "initializer dtype must be FLOAT16 or FLOAT"
                )
            if dtype != input_dtypes["query"]:
                raise OnnxExportError(
                    f"ApplyRotaryPosEmb node {node_name!r} {role} "
                    "initializer dtype must match query input"
                )


def validate_dequant_initializers(model: onnx.ModelProto) -> None:
    """Validate encoded AscendDequant scales."""
    initializers = {
        item.name: item for item in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type != "AscendDequant":
            continue
        scale = initializers.get(node.input[1])
        if scale is None or scale.data_type != TensorProto.UINT64:
            raise OnnxExportError(
                "AscendDequant scale must be a UINT64 initializer"
            )
        values = numpy_helper.to_array(scale).astype(
            "uint64",
            copy=False,
        )
        if ((values >> 32) != 0).any():
            raise OnnxExportError(
                "AscendDequant scale high 32 bits must be zero"
            )
        decoded = (
            (values & 0xFFFFFFFF)
            .astype("uint32")
            .view("float32")
        )
        if not (
            (decoded > 0) & (decoded < float("inf"))
        ).all():
            raise OnnxExportError(
                "AscendDequant scale must decode to finite positives"
            )
