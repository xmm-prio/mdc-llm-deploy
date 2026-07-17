"""Centralized quantization-topology validation."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.onnx.validation.model import validate_mdc_model
from mdc_llm_deploy.operators.contracts.attention import (
    ATTENTION_INPUT_COUNT,
    RELEASE_ATTENTION_ATTRIBUTES,
    AttentionInput,
)


def _set_metadata(
    model: onnx.ModelProto,
    *,
    target: str,
    stage: str = "QUANTIZED_PREFILL",
    mask_mode: str = "maskless",
    target_count: int | None = None,
) -> None:
    properties = {
        "mdc.graph_schema_version": "1",
        "mdc.stage": stage,
        "mdc.mask_mode": mask_mode,
        "mdc.mask_semantics": (
            "explicit-causal"
            if mask_mode == "masked"
            else "all-visible-non-causal"
        ),
        "mdc.model_kind": "dense",
        "mdc.algorithm": "minmax",
        "mdc.target": target,
        "mdc.dialect": "MDC ONNX",
        "mdc.numeric_spine": "validated-standard-aten",
        "mdc.lowering_source": "test",
    }
    if target_count is not None:
        properties["mdc.linear.target_count"] = str(target_count)
    helper.set_model_props(model, properties)


def _quant_initializers(prefix: str = "shared") -> list[onnx.TensorProto]:
    return [
        numpy_helper.from_array(
            np.asarray(2.0, dtype=np.float32),
            name=f"{prefix}.scale",
        ),
        numpy_helper.from_array(
            np.asarray(0.0, dtype=np.float32),
            name=f"{prefix}.offset",
        ),
    ]


def _linear_model(target_count: int = 2) -> onnx.ModelProto:
    nodes = [
        helper.make_node(
            "NPUAscendQuantV2",
            ["input", "shared.scale", "shared.offset"],
            ["shared.quantized"],
            name="shared.quant",
            axis=-1,
            dtype=2,
        )
    ]
    initializers = _quant_initializers()
    outputs: list[str] = []
    for index in range(2):
        weight_name = f"weight.{index}"
        dequant_name = f"dequant_scale.{index}"
        accumulator = f"accumulator.{index}"
        output = f"output.{index}"
        initializers.extend(
            [
                numpy_helper.from_array(
                    np.ones((2, 2), dtype=np.int8),
                    name=weight_name,
                ),
                numpy_helper.from_array(
                    np.asarray(
                        np.float32(0.5).view(np.uint32),
                        dtype=np.uint64,
                    ),
                    name=dequant_name,
                ),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    ["shared.quantized", weight_name],
                    [accumulator],
                ),
                helper.make_node(
                    "AscendDequant",
                    [accumulator, dequant_name],
                    [output],
                    sqrt_mode=0,
                    relu_flag=0,
                    dtype=0,
                ),
            ]
        )
        outputs.append(output)
    nodes.append(helper.make_node("Add", outputs, ["logits"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "shared-linear",
            [
                helper.make_tensor_value_info(
                    "input",
                    TensorProto.FLOAT,
                    [1, 2],
                )
            ],
            [
                helper.make_tensor_value_info(
                    "logits",
                    TensorProto.FLOAT,
                    [1, 2],
                )
            ],
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    _set_metadata(model, target="linear", target_count=target_count)
    return model


def test_shared_quantizer_covers_multiple_linear_targets() -> None:
    model = _linear_model()

    metadata = validate_mdc_model(model)

    assert metadata.targets == {"linear"}


def test_equivalent_duplicate_quantizers_are_rejected() -> None:
    model = _linear_model(target_count=3)
    duplicate = helper.make_node(
        "NPUAscendQuantV2",
        ["input", "shared.scale", "shared.offset"],
        ["duplicate.quantized"],
        name="duplicate.quant",
        axis=-1,
        dtype=2,
    )
    weight = numpy_helper.from_array(
        np.ones((2, 2), dtype=np.int8),
        name="weight.duplicate",
    )
    dequant_scale = numpy_helper.from_array(
        np.asarray(np.float32(0.5).view(np.uint32), dtype=np.uint64),
        name="dequant_scale.duplicate",
    )
    model.graph.initializer.extend([weight, dequant_scale])
    del model.graph.node[-1]
    model.graph.node.extend(
        [
            duplicate,
            helper.make_node(
                "MatMul",
                ["duplicate.quantized", "weight.duplicate"],
                ["accumulator.duplicate"],
            ),
            helper.make_node(
                "AscendDequant",
                ["accumulator.duplicate", "dequant_scale.duplicate"],
                ["output.duplicate"],
                sqrt_mode=0,
                relu_flag=0,
                dtype=0,
            ),
            helper.make_node(
                "Sum",
                ["output.0", "output.1", "output.duplicate"],
                ["logits"],
            ),
        ]
    )

    with pytest.raises(OnnxExportError, match="must be shared"):
        validate_mdc_model(model)


def test_linear_target_count_requires_dequant_coverage() -> None:
    model = _linear_model(target_count=3)

    with pytest.raises(OnnxExportError, match="coverage is incomplete"):
        validate_mdc_model(model)


def test_orphan_quantizer_is_rejected() -> None:
    model = _linear_model()
    model.graph.initializer.extend(_quant_initializers("orphan"))
    model.graph.node.extend(
        [
            helper.make_node(
                "NPUAscendQuantV2",
                ["input", "orphan.scale", "orphan.offset"],
                ["orphan.output"],
                name="orphan.quant",
                axis=0,
                dtype=2,
            )
        ]
    )

    with pytest.raises(OnnxExportError, match="do not reach graph outputs"):
        validate_mdc_model(model)


def _decode_attention_model() -> onnx.ModelProto:
    inputs = [""] * ATTENTION_INPUT_COUNT
    inputs[AttentionInput.QUERY] = "query"
    inputs[AttentionInput.KEY] = "key"
    inputs[AttentionInput.VALUE] = "value"
    scale_slots = {
        AttentionInput.DEQUANT_SCALE1: "dequant_scale1",
        AttentionInput.QUANT_SCALE1: "quant_scale1",
        AttentionInput.DEQUANT_SCALE2: "dequant_scale2",
        AttentionInput.KEY_ANTIQUANT_SCALE: "key_scale",
        AttentionInput.VALUE_ANTIQUANT_SCALE: "value_scale",
        AttentionInput.DEQUANT_SCALE_QUERY: "query_scale",
    }
    for slot, name in scale_slots.items():
        inputs[slot] = name
    initializers = [
        numpy_helper.from_array(
            np.asarray(0.5, dtype=np.float32),
            name=name,
        )
        for name in scale_slots.values()
    ]
    attention = helper.make_node(
        "FusedInferAttentionScore",
        inputs,
        ["logits", "lse"],
        name="decode.attention",
        num_heads=1,
        num_key_value_heads=1,
        scale=1.0,
        **dict(RELEASE_ATTENTION_ATTRIBUTES),
    )
    model = helper.make_model(
        helper.make_graph(
            [attention],
            "decode-attention",
            [
                helper.make_tensor_value_info(
                    name,
                    TensorProto.INT8,
                    [1, 1, 1, 2],
                )
                for name in ("query", "key", "value")
            ],
            [
                helper.make_tensor_value_info(
                    "logits",
                    TensorProto.FLOAT16,
                    [1, 1, 1, 2],
                )
            ],
            initializer=initializers,
            value_info=[
                helper.make_tensor_value_info(
                    "lse",
                    TensorProto.FLOAT,
                    [1],
                )
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    _set_metadata(
        model,
        target="attention",
        stage="QUANTIZED_DECODE",
    )
    return model


def test_decode_int8_graph_inputs_need_no_in_graph_quantizer() -> None:
    model = _decode_attention_model()

    metadata = validate_mdc_model(model)

    assert metadata.targets == {"attention"}
