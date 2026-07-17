"""Tests for call-local ONNX quantization sharing."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.graph.metadata import QuantizedTarget
from mdc_llm_deploy.onnx.transform.quantization import request_quant
from mdc_llm_deploy.onnx.transform.support import OnnxLoweringContext


def _target(
    fqn: str,
    *,
    scale: tuple[float, ...] = (0.25,),
    zero_point: tuple[int, ...] = (0,),
) -> QuantizedTarget:
    return QuantizedTarget(
        fqn=fqn,
        target_type="linear",
        algorithm="minmax",
        bits=8,
        granularity="per_tensor",
        symmetric=True,
        scale=scale,
        zero_point=zero_point,
    )


def _model(
    *,
    dtype: int = TensorProto.FLOAT,
    graph_output: bool = False,
) -> onnx.ModelProto:
    source = helper.make_tensor_value_info("source", dtype, (1, 2, 4))
    nodes: list[onnx.NodeProto] = []
    outputs: list[onnx.ValueInfoProto] = []
    if graph_output:
        nodes.append(helper.make_node("Identity", ["source"], ["cache"], name="cache"))
        outputs.append(helper.make_tensor_value_info("cache", dtype, (1, 2, 4)))
    return helper.make_model(helper.make_graph(nodes, "quant", [source], outputs))


def _quantizers(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    return [
        node for node in model.graph.node if node.op_type == "NPUAscendQuantV2"
    ]


def test_context_centralizes_indexes_and_deterministic_names() -> None:
    model = _model()
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(1, dtype=np.float32), name="taken"),
            numpy_helper.from_array(np.asarray(2, dtype=np.float32), name="taken"),
        ]
    )
    context = OnnxLoweringContext.from_model(model)

    assert context.unique_name("taken") == "taken.1"
    assert context.unique_name("taken") == "taken.2"
    assert numpy_helper.to_array(context.first_initializer("taken")) == 1
    context.append_initializer(
        numpy_helper.from_array(np.ones((3,), dtype=np.int32), name="new")
    )

    assert context.types["new"] == (TensorProto.INT32, (3,))
    assert context.first_initializer("new") is model.graph.initializer[-1]


def test_equivalent_emitted_contract_reuses_first_quantizer() -> None:
    model = _model()
    context = OnnxLoweringContext.from_model(model)
    first = context.request_quant(
        "source",
        _target("q_proj", scale=(0.1,)),
        axis=-1,
        name="quant.q",
    )
    equivalent_float32 = float(np.float32(0.1))
    second = request_quant(
        context,
        "source",
        _target("k_proj", scale=(equivalent_float32,)),
        axis=-1,
        name="quant.k",
    )

    assert second == first
    assert len(_quantizers(model)) == 1
    assert len(model.graph.initializer) == 2
    assert _quantizers(model)[0].name == "quant.q"


def test_target_families_share_one_effective_quantizer() -> None:
    model = _model()
    context = OnnxLoweringContext.from_model(model)
    targets = (
        _target("linear"),
        replace(_target("attention"), target_type="attention"),
        replace(_target("moe"), target_type="moe"),
    )

    outputs = {
        context.request_quant("source", target, axis=-1)
        for target in targets
    }

    assert len(outputs) == 1
    assert len(_quantizers(model)) == 1


@pytest.mark.parametrize(
    ("source", "scale", "zero_point", "axis"),
    [
        ("other", (0.25,), (0,), -1),
        ("source", (0.5,), (0,), -1),
        ("source", (0.25,), (1,), -1),
        ("source", (0.25,), (0,), -2),
        ("source", (0.25, 0.5), (0, 0), -1),
    ],
)
def test_different_effective_contracts_are_isolated(
    source: str,
    scale: tuple[float, ...],
    zero_point: tuple[int, ...],
    axis: int,
) -> None:
    model = _model()
    model.graph.input.append(
        helper.make_tensor_value_info("other", TensorProto.FLOAT, (1, 2, 4))
    )
    context = OnnxLoweringContext.from_model(model)
    first = context.request_quant("source", _target("first"), axis=-1)
    second = context.request_quant(
        source,
        _target("second", scale=scale, zero_point=zero_point),
        axis=axis,
    )

    assert second != first
    assert len(_quantizers(model)) == 2


def test_source_dtype_controls_actual_parameter_contract() -> None:
    float_model = _model()
    fp16_model = _model(dtype=TensorProto.FLOAT16)
    float_context = OnnxLoweringContext.from_model(float_model)
    fp16_context = OnnxLoweringContext.from_model(fp16_model)

    float_context.request_quant("source", _target("value"), axis=-1)
    fp16_context.request_quant("source", _target("value"), axis=-1)

    assert float_model.graph.initializer[0].data_type == TensorProto.FLOAT
    assert fp16_model.graph.initializer[0].data_type == TensorProto.FLOAT16


def test_name_collisions_are_allocated_in_first_use_order() -> None:
    model = _model()
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray(1, dtype=np.float32),
            name="quant.scale",
        )
    )
    model.graph.node.append(
        helper.make_node("Identity", ["source"], ["occupied"], name="quant")
    )
    context = OnnxLoweringContext.from_model(model)

    output = context.request_quant(
        "source",
        _target("value"),
        axis=-1,
        name="quant",
    )

    quant = _quantizers(model)[0]
    assert quant.name == "quant.1"
    assert quant.input[1:] == ["quant.scale.1", "quant.offset"]
    assert output == "quant.output"


def test_graph_output_rebind_preserves_abi_name_and_topological_order() -> None:
    model = _model(graph_output=True)
    context = OnnxLoweringContext.from_model(model)

    internal = context.rebind_graph_output("cache", output_dtype=TensorProto.INT8)
    output = context.request_quant(
        internal,
        _target("cache"),
        axis=-2,
        preferred_output="cache",
        name="cache.quant",
    )

    assert internal == "cache.float"
    assert output == "cache"
    assert [node.op_type for node in model.graph.node] == [
        "Identity",
        "NPUAscendQuantV2",
    ]
    assert model.graph.node[0].output == [internal]
    assert model.graph.node[1].input[0] == internal
    assert model.graph.node[1].output == ["cache"]
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.INT8
    assert context.types[internal] == (TensorProto.FLOAT, (1, 2, 4))
    assert context.types["cache"] == (TensorProto.INT8, (1, 2, 4))
    assert context.rebind_graph_output("cache") == internal


def test_preferred_output_requires_explicit_graph_output_rebind() -> None:
    model = _model(graph_output=True)
    context = OnnxLoweringContext.from_model(model)

    with pytest.raises(OnnxExportError, match="was not rebound"):
        context.request_quant(
            "source",
            _target("cache"),
            axis=-1,
            preferred_output="cache",
        )


@pytest.mark.parametrize(
    ("model", "source", "target", "output_dtype", "message"),
    [
        (_model(), "missing", _target("x"), TensorProto.INT8, "no static type"),
        (
            _model(dtype=TensorProto.INT32),
            "source",
            _target("x"),
            TensorProto.INT8,
            "floating-point dtype",
        ),
        (_model(), "source", _target("x"), TensorProto.UINT8, "only supports INT8"),
        (
            _model(),
            "source",
            _target("x", scale=(0.0,)),
            TensorProto.INT8,
            "finite and positive",
        ),
        (
            _model(),
            "source",
            _target("x", scale=(0.25, 0.5), zero_point=(0,)),
            TensorProto.INT8,
            "mismatched emitted shapes",
        ),
    ],
)
def test_invalid_requests_leave_graph_without_quantizer(
    model: onnx.ModelProto,
    source: str,
    target: QuantizedTarget,
    output_dtype: int,
    message: str,
) -> None:
    context = OnnxLoweringContext.from_model(model)
    before = model.SerializeToString()

    with pytest.raises(OnnxExportError, match=message):
        context.request_quant(
            source,
            target,
            axis=-1,
            output_dtype=output_dtype,
        )

    assert model.SerializeToString() == before
    assert not _quantizers(model)


def test_conflicting_preferred_name_does_not_duplicate_quantizer() -> None:
    model = _model()
    context = OnnxLoweringContext.from_model(model)
    context.request_quant("source", _target("first"), axis=-1)

    with pytest.raises(OnnxExportError, match="conflicting preferred outputs"):
        context.request_quant(
            "source",
            _target("second"),
            axis=-1,
            preferred_output="cache",
        )

    assert len(_quantizers(model)) == 1
