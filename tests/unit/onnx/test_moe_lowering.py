"""Behavioral tests for quantized MoE ONNX lowering."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.graph.metadata import (
    GraphMetadata,
    GraphStage,
    QuantizedTarget,
)
from mdc_llm_deploy.onnx.transform import moe, support


def _target(fqn: str) -> QuantizedTarget:
    return QuantizedTarget(
        fqn=fqn,
        target_type="moe",
        algorithm="minmax",
        bits=8,
        granularity="per_tensor",
        symmetric=True,
        scale=(0.25,),
        zero_point=(0,),
    )


def _metadata(
    *fqns: str,
    activation_scales: dict[str, float] | None = None,
) -> GraphMetadata:
    activation_scales = activation_scales or {}
    return GraphMetadata(
        schema_version=1,
        stage=GraphStage.QUANTIZED_DECODE,
        model_kind="test",
        input_abi=(),
        output_abi=(),
        quantized_targets=tuple(_target(fqn) for fqn in fqns),
        sequence_length=1,
        properties={
            "hidden_size": 4,
            "num_experts_per_tok": 2,
            "activation_qparams": {
                fqn: {
                    "bits": 8,
                    "granularity": "per_tensor",
                    "mode": None,
                    "symmetric": True,
                    "scale": (activation_scales.get(fqn, 0.125),),
                    "zero_point": (0,),
                    "intermediate_scale": (float(index + 1),) * 2,
                }
                for index, fqn in enumerate(sorted(fqns))
            },
        },
    )


def _model(
    count: int = 1,
    *,
    ids_dtype: int | None = TensorProto.INT64,
    duplicate_node_names: bool = False,
) -> onnx.ModelProto:
    inputs = [
        helper.make_tensor_value_info(
            "source",
            TensorProto.FLOAT16,
            (1, 1, 4),
        ),
        helper.make_tensor_value_info(
            "score",
            TensorProto.FLOAT16,
            (1, 2),
        ),
    ]
    if ids_dtype is not None:
        inputs.append(
            helper.make_tensor_value_info(
                "topk_ids",
                ids_dtype,
                (1, 2),
            )
        )
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    for index in range(count):
        weight = f"weight.{index}"
        scales = f"scales.{index}"
        nodes.append(
            helper.make_node(
                "MoeExpert",
                ["source", "topk_ids", "score", weight, scales],
                [f"output.{index}"],
                name="moe" if duplicate_node_names else f"moe.{index}",
            )
        )
        initializers.extend(
            [
                numpy_helper.from_array(
                    np.arange(24, dtype=np.int8).reshape(2, 12),
                    name=weight,
                ),
                numpy_helper.from_array(
                    np.full((2, 3), index + 0.25, dtype=np.float32),
                    name=scales,
                ),
            ]
        )
    return helper.make_model(
        helper.make_graph(nodes, "moe_fixture", inputs, [], initializers)
    )


def _initializer_names(model: onnx.ModelProto) -> list[str]:
    return [item.name for item in model.graph.initializer]


def test_name_allocation_preserves_occupancy_boundary_and_append_order() -> None:
    model = _model()
    prefixes = {
        "atc": "moe.0.atc_scales",
        "scale": "moe.0.input_quant.scale",
        "offset": "moe.0.input_quant.offset",
        "output": "moe.0.input_quant.output",
        "cast": "moe.0.topk_ids_int16",
    }
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.ones((1,), dtype=np.float32),
            name=prefixes["atc"],
        )
    )
    model.graph.input.append(
        helper.make_tensor_value_info(
            prefixes["scale"],
            TensorProto.FLOAT,
            (1,),
        )
    )
    model.graph.node.insert(
        0,
        helper.make_node(
            "Identity",
            ["source"],
            [f"{prefixes['scale']}.1"],
            name=prefixes["cast"],
        ),
    )
    model.graph.node.insert(
        1,
        helper.make_node(
            "Identity",
            ["source"],
            [prefixes["offset"]],
            name="offset.conflict",
        ),
    )
    model.graph.output.append(
        helper.make_tensor_value_info(
            prefixes["output"],
            TensorProto.INT8,
            (1, 1, 4),
        )
    )
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info(
                prefixes["cast"],
                TensorProto.INT16,
                (1, 2),
            ),
            helper.make_tensor_value_info(
                f"{prefixes['scale']}.2",
                TensorProto.FLOAT,
                (1,),
            ),
        ]
    )

    moe.adapt_quantized_moe(model, _metadata("layer"))

    adapted = next(node for node in model.graph.node if node.op_type == "MoeExpert")
    quant = next(
        node for node in model.graph.node if node.op_type == "NPUAscendQuantV2"
    )
    cast = next(node for node in model.graph.node if node.op_type == "Cast")
    assert adapted.input[4] == f"{prefixes['atc']}.1"
    assert quant.input[1:] == [f"{prefixes['scale']}.2", f"{prefixes['offset']}.1"]
    assert quant.output == [prefixes["output"]]
    assert cast.output == [prefixes["cast"]]
    assert _initializer_names(model)[-3:] == [
        f"{prefixes['atc']}.1",
        f"{prefixes['scale']}.2",
        f"{prefixes['offset']}.1",
    ]
    assert [node.op_type for node in model.graph.node[-2:]] == [
        "NPUAscendQuantV2",
        "Cast",
    ]


@pytest.mark.parametrize(
    ("ids_dtype", "cast_expected"),
    [
        (TensorProto.INT16, False),
        (TensorProto.INT64, True),
        (None, True),
    ],
)
def test_cast_condition_uses_initial_type_snapshot(
    ids_dtype: int | None,
    cast_expected: bool,
) -> None:
    model = _model(ids_dtype=ids_dtype)

    moe.adapt_quantized_moe(model, _metadata("layer"))

    adapted = next(node for node in model.graph.node if node.op_type == "MoeExpert")
    casts = [node for node in model.graph.node if node.op_type == "Cast"]
    assert bool(casts) is cast_expected
    if cast_expected:
        assert len(casts) == 1
        assert adapted.input[1] == casts[0].output[0]
        assert helper.get_attribute_value(casts[0].attribute[0]) == TensorProto.INT16
    else:
        assert adapted.input[1] == "topk_ids"


def test_multiple_targets_keep_sorted_pairing_and_share_reservations() -> None:
    model = _model(2, duplicate_node_names=True)

    moe.adapt_quantized_moe(model, _metadata("layer.z", "layer.a"))

    experts = [node for node in model.graph.node if node.op_type == "MoeExpert"]
    first_scales = numpy_helper.to_array(
        next(item for item in model.graph.initializer if item.name == experts[0].input[4])
    )
    second_scales = numpy_helper.to_array(
        next(item for item in model.graph.initializer if item.name == experts[1].input[4])
    )
    assert experts[0].input[4] == "moe.atc_scales"
    assert experts[1].input[4] == "moe.atc_scales.1"
    np.testing.assert_array_equal(first_scales[[3, 7]], [1.0, 1.0])
    np.testing.assert_array_equal(second_scales[[3, 7]], [2.0, 2.0])
    assert [node.op_type for node in model.graph.node[-4:]] == [
        "NPUAscendQuantV2",
        "Cast",
        "NPUAscendQuantV2",
        "Cast",
    ]


def test_failure_keeps_prior_target_and_current_prefix_only() -> None:
    model = _model(3)
    third_before = model.graph.node[2].SerializeToString()

    with pytest.raises(
        OnnxExportError,
        match=r"FP16 quantization scale for 'layer.b' is too small",
    ):
        moe.adapt_quantized_moe(
            model,
            _metadata(
                "layer.a",
                "layer.b",
                "layer.c",
                activation_scales={"layer.b": 1e-10},
            ),
        )

    experts = [node for node in model.graph.node if node.op_type == "MoeExpert"]
    assert experts[0].input[0] == "moe.0.input_quant.output"
    assert experts[0].input[1] == "moe.0.topk_ids_int16"
    assert experts[0].input[4] == "moe.0.atc_scales"
    assert tuple(model.graph.initializer[0].dims) == (24,)
    assert experts[1].input == [
        "source",
        "topk_ids",
        "score",
        "weight.1",
        "scales.1",
    ]
    assert tuple(model.graph.initializer[2].dims) == (2, 12)
    assert "moe.1.atc_scales" in _initializer_names(model)
    assert "moe.1.input_quant.scale" not in _initializer_names(model)
    assert experts[2].SerializeToString() == third_before
    assert tuple(model.graph.initializer[4].dims) == (2, 12)


def test_target_count_error_leaves_graph_unchanged() -> None:
    model = _model(2)
    before = model.SerializeToString()

    with pytest.raises(
        OnnxExportError,
        match="Quantized MoeExpert targets do not match ONNX nodes",
    ):
        moe.adapt_quantized_moe(model, _metadata("only"))

    assert model.SerializeToString() == before


def test_append_quant_uses_injected_allocator_in_execution_order() -> None:
    model = _model(0)
    target = _target("layer")
    calls: list[str] = []

    def allocate(base: str) -> str:
        calls.append(base)
        return f"allocated.{len(calls)}"

    output = support.append_quant(
        model,
        "source",
        (1, 1, 4),
        target,
        "quant",
        name_allocator=allocate,
    )

    assert calls == ["quant.scale", "quant.offset", "quant.output"]
    assert output == "allocated.3"
    assert _initializer_names(model) == ["allocated.1", "allocated.2"]
    assert model.graph.node[-1].input == [
        "source",
        "allocated.1",
        "allocated.2",
    ]
    assert model.graph.node[-1].output == ["allocated.3"]


def test_append_quant_default_allocator_behavior_is_unchanged() -> None:
    model = _model(0)
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.ones((1,), dtype=np.float16),
            name="quant.scale",
        )
    )

    output = support.append_quant(
        model,
        "source",
        (1, 1, 4),
        _target("layer"),
        "quant",
    )

    assert _initializer_names(model)[-2:] == ["quant.scale.1", "quant.offset"]
    assert output == "quant.output"


def test_adaptation_builds_one_context_and_skips_default_name_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(3)
    builds = 0
    allocations: list[str] = []
    real_from_model = moe._MoeAdaptationContext.from_model.__func__
    real_unique_name = moe._MoeAdaptationContext.unique_name

    def counted_from_model(
        cls: type[moe._MoeAdaptationContext],
        value: onnx.ModelProto,
    ) -> moe._MoeAdaptationContext:
        nonlocal builds
        builds += 1
        return real_from_model(cls, value)

    def counted_unique_name(
        context: moe._MoeAdaptationContext,
        base: str,
    ) -> str:
        allocations.append(base)
        return real_unique_name(context, base)

    def reject_scan(model: onnx.ModelProto, base: str) -> str:
        raise AssertionError(f"default unique_name called for {base}")

    monkeypatch.setattr(
        moe._MoeAdaptationContext,
        "from_model",
        classmethod(counted_from_model),
    )
    monkeypatch.setattr(
        moe._MoeAdaptationContext,
        "unique_name",
        counted_unique_name,
    )
    monkeypatch.setattr(support, "unique_name", reject_scan)

    moe.adapt_quantized_moe(
        model,
        _metadata("layer.c", "layer.a", "layer.b"),
    )

    assert builds == 1
    assert len(allocations) == 15
    assert allocations[:5] == [
        "moe.0.atc_scales",
        "moe.0.input_quant.scale",
        "moe.0.input_quant.offset",
        "moe.0.input_quant.output",
        "moe.0.topk_ids_int16",
    ]
