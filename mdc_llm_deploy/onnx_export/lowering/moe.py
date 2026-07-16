"""Adapt generic quantized MoeExpert nodes to the deployed ATC ABI."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ...errors import OnnxExportError
from ...graph_types import GraphMetadata
from .support import (
    activation_target,
    append_quant,
    append_value,
    initializer,
    model_types,
    unique_name,
)


def _intermediate_scales(
    value: GraphMetadata,
    target_fqn: str,
    expert_count: int,
) -> np.ndarray:
    activation = value.properties.get("activation_qparams")
    if not isinstance(activation, Mapping):
        raise OnnxExportError("MoeExpert activation qparams are missing")
    parameters = activation.get(target_fqn)
    if not isinstance(parameters, Mapping):
        raise OnnxExportError(
            f"MoeExpert activation qparams for {target_fqn!r} are missing"
        )
    raw = parameters.get("intermediate_scale")
    if not isinstance(raw, (tuple, list)) or len(raw) != expert_count:
        raise OnnxExportError(
            f"MoeExpert intermediate scales for {target_fqn!r} are invalid"
        )
    result = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise OnnxExportError("MoeExpert intermediate scales must be positive")
    return result


def adapt_quantized_moe(
    model: onnx.ModelProto,
    value: GraphMetadata,
) -> None:
    """Rewrite quantized generic nodes to the ATC INT8 activation layout."""
    targets = sorted(
        (
            item
            for item in value.quantized_targets
            if item.target_type == "moe"
        ),
        key=lambda item: item.fqn,
    )
    nodes = [
        node for node in model.graph.node if node.op_type == "MoeExpert"
    ]
    if not targets:
        return
    if len(nodes) != len(targets):
        raise OnnxExportError(
            "Quantized MoeExpert targets do not match ONNX nodes"
        )
    initializers = {
        item.name: item for item in model.graph.initializer
    }
    types = model_types(model)
    for node, target in zip(nodes, targets, strict=True):
        if len(node.input) < 5:
            raise OnnxExportError(
                "Quantized MoeExpert node lacks scale input"
            )
        weight = initializers.get(node.input[3])
        scales = initializers.get(node.input[4])
        if weight is None or scales is None:
            raise OnnxExportError(
                "Quantized MoeExpert parameters must be initializers"
            )
        if weight.data_type != TensorProto.INT8 or len(weight.dims) != 2:
            raise OnnxExportError(
                "ATC MoeExpert requires expert-major INT8 weights"
            )
        expert_count = int(weight.dims[0])
        projection_scales = numpy_helper.to_array(scales).reshape(
            expert_count,
            3,
        )
        intermediate = _intermediate_scales(
            value,
            target.fqn,
            expert_count,
        )
        activation = activation_target(value, target)
        if len(activation.scale) != 1:
            raise OnnxExportError(
                "ATC MoeExpert requires per-tensor activation scale"
            )
        combined = np.empty((1 + expert_count * 4,), dtype=np.float32)
        combined[0] = activation.scale[0]
        combined[1:] = np.column_stack(
            (
                projection_scales[:, 0],
                projection_scales[:, 1],
                intermediate,
                projection_scales[:, 2],
            )
        ).reshape(-1)
        scale_name = unique_name(
            model,
            f"{node.name}.atc_scales",
        )
        model.graph.initializer.append(
            initializer(scale_name, combined)
        )
        source = node.input[0]
        token_count = value.sequence_length if value.stage.is_prefill else 1
        hidden_size = value.properties.get("hidden_size")
        top_k = value.properties.get("num_experts_per_tok")
        if (
            type(hidden_size) is not int
            or hidden_size <= 0
            or type(top_k) is not int
            or top_k <= 0
        ):
            raise OnnxExportError(
                "MoeExpert ATC adaptation requires static model dimensions"
            )
        source_shape = types.get(
            source,
            (TensorProto.FLOAT16, (token_count, hidden_size)),
        )[1]
        ids_dtype, ids_shape = types.get(
            node.input[1],
            (TensorProto.INT64, (token_count, top_k)),
        )
        if source not in types:
            source_dtype = types.get(
                node.input[2],
                (TensorProto.FLOAT16, ids_shape),
            )[0]
            append_value(
                model,
                source,
                source_dtype,
                source_shape,
            )
        node.input[0] = append_quant(
            model,
            source,
            source_shape,
            activation,
            f"{node.name}.input_quant",
        )
        if ids_dtype != TensorProto.INT16:
            cast_output = unique_name(
                model,
                f"{node.name}.topk_ids_int16",
            )
            model.graph.node.append(
                helper.make_node(
                    "Cast",
                    [node.input[1]],
                    [cast_output],
                    name=f"{node.name}.topk_ids_cast",
                    to=TensorProto.INT16,
                )
            )
            append_value(
                model,
                cast_output,
                TensorProto.INT16,
                ids_shape,
            )
            node.input[1] = cast_output
        node.input[4] = scale_name
        weight.dims[:] = [numpy_helper.to_array(weight).size]
