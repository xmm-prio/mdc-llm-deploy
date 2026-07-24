from __future__ import annotations

import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.onnx import AdapterConfig, OnnxAdapter

from .minmax_export_fixtures import (
    LOWERING_SUPPORTED_CASES,
    LOWERING_UNSUPPORTED_CASES,
    MINMAX_EXPORT_CASES,
    MinMaxExportCase,
    export_quantized_linear,
    quantized_onnx_export_config,
)

_QDQ_OPS = frozenset({"QuantizeLinear", "DequantizeLinear"})


def _attribute(node: onnx.NodeProto, name: str) -> int | None:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(helper.get_attribute_value(attribute))
    return None


def _initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    return next(initializer for initializer in model.graph.initializer if initializer.name == name)


def _qdq_pairs(model: onnx.ModelProto) -> list[tuple[onnx.NodeProto, onnx.NodeProto]]:
    producers = {output: node for node in model.graph.node for output in node.output}
    pairs: list[tuple[onnx.NodeProto, onnx.NodeProto]] = []
    for dequantize in model.graph.node:
        if dequantize.op_type != "DequantizeLinear":
            continue
        quantize = producers[dequantize.input[0]]
        assert quantize.op_type == "QuantizeLinear"
        pairs.append((quantize, dequantize))
    return pairs


def _assert_pair(
    model: onnx.ModelProto,
    pair: tuple[onnx.NodeProto, onnx.NodeProto],
    *,
    float_dtype: int,
    symmetric: bool,
    axis: int | None,
    parameter_length: int | None,
) -> None:
    quantize, dequantize = pair
    assert quantize.input[1:] == dequantize.input[1:]
    assert _attribute(quantize, "axis") == axis
    assert _attribute(dequantize, "axis") == axis
    if symmetric:
        assert len(quantize.input) == 2 or quantize.input[2] == ""
        assert len(dequantize.input) == 2 or dequantize.input[2] == ""
        assert _attribute(quantize, "output_dtype") == TensorProto.INT8
    else:
        assert len(quantize.input) == 3 and quantize.input[2]
        assert len(dequantize.input) == 3 and dequantize.input[2]
        assert _initializer(model, quantize.input[2]).data_type == TensorProto.INT8

    scale_initializer = _initializer(model, quantize.input[1])
    assert scale_initializer.data_type == float_dtype
    scale = numpy_helper.to_array(scale_initializer)
    expected_shape = () if parameter_length is None else (parameter_length,)
    assert scale.shape == expected_shape
    assert bool((scale > 0).all())

    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.value_info, *model.graph.output)
    }
    assert value_types[quantize.output[0]] == TensorProto.INT8
    assert value_types[dequantize.output[0]] == float_dtype


def test_quantized_export_fixture_uses_opset21_without_changing_float_fixture() -> None:
    from tests.integration.onnx.qwen3_export_fixtures import onnx_export_config

    assert quantized_onnx_export_config().opset_version == 21
    assert onnx_export_config().opset_version == 18


def test_export_and_lowering_matrix_cardinality() -> None:
    assert len(MINMAX_EXPORT_CASES) == 72
    assert len(LOWERING_SUPPORTED_CASES) == 16
    assert len(LOWERING_UNSUPPORTED_CASES) == 56


@pytest.mark.integration
@pytest.mark.parametrize("case", MINMAX_EXPORT_CASES, ids=lambda case: case.id)
def test_raw_qdq_matrix_exports_standard_opset21(case: MinMaxExportCase) -> None:
    model = export_quantized_linear(case)

    assert next(opset.version for opset in model.opset_import if opset.domain == "") == 21
    onnx.checker.check_model(model)
    pairs = _qdq_pairs(model)
    assert len(pairs) == int(case.config.activation) + int(case.config.weight)

    pair_by_input = {pair[0].input[0]: pair for pair in pairs}
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    weight_pair = next(
        (pair for input_name, pair in pair_by_input.items() if input_name in initializer_names),
        None,
    )
    activation_pair = next(
        (pair for input_name, pair in pair_by_input.items() if input_name not in initializer_names),
        None,
    )
    float_dtype = {
        torch.float16: TensorProto.FLOAT16,
        torch.bfloat16: TensorProto.BFLOAT16,
        torch.float32: TensorProto.FLOAT,
    }[case.dtype]

    if case.config.weight:
        assert weight_pair is not None
        weight_per_channel = case.config.weight_granularity == "per_channel"
        _assert_pair(
            model,
            weight_pair,
            float_dtype=float_dtype,
            symmetric=case.config.weight_symmetric,
            axis=0 if weight_per_channel else None,
            parameter_length=3 if weight_per_channel else None,
        )
    else:
        assert weight_pair is None

    if case.config.activation:
        assert activation_pair is not None
        activation_per_token = case.config.activation_granularity == "per_token"
        _assert_pair(
            model,
            activation_pair,
            float_dtype=float_dtype,
            symmetric=case.config.activation_symmetric,
            axis=-2 if activation_per_token else None,
            parameter_length=3 if activation_per_token else None,
        )
    else:
        assert activation_pair is None


@pytest.mark.integration
@pytest.mark.parametrize("case", LOWERING_SUPPORTED_CASES, ids=lambda case: case.id)
def test_supported_qdq_subset_lowers_without_residual_qdq(case: MinMaxExportCase) -> None:
    model = export_quantized_linear(case)

    returned = OnnxAdapter(AdapterConfig())(model)

    assert returned is model
    assert next(opset.version for opset in model.opset_import if opset.domain == "") == 18
    assert not any(
        node.domain in ("", "ai.onnx") and node.op_type in _QDQ_OPS
        for node in model.graph.node
    )
    assert sum(node.op_type == "NPUAscendQuantV2" for node in model.graph.node) == 1
    assert sum(node.op_type == "AscendDequant" for node in model.graph.node) == 1
    onnx.checker.check_model(model)


@pytest.mark.integration
@pytest.mark.parametrize("case", LOWERING_UNSUPPORTED_CASES, ids=lambda case: case.id)
def test_unsupported_qdq_matrix_fails_adapter_explicitly(case: MinMaxExportCase) -> None:
    model = export_quantized_linear(case)
    original = model.SerializeToString()

    with pytest.raises(ValueError, match=_expected_lowering_error(case)):
        OnnxAdapter(AdapterConfig())(model)

    assert model.SerializeToString() == original


def _expected_lowering_error(case: MinMaxExportCase) -> str:
    if not case.config.weight or not case.config.activation:
        return "activation and weight must both use supported QDQ"
    if case.dtype is torch.bfloat16:
        return "scale must be floating point"
    return "weight quantization must be symmetric"
