"""Behavioral tests for quantized linear ONNX lowering."""

from __future__ import annotations

from collections.abc import Callable

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
from mdc_llm_deploy.onnx.transform import linear, support

SHAPE = (1, 1, 2)


def _target(
    fqn: str,
    *,
    granularity: str = "per_tensor",
    scale: tuple[float, ...] = (0.25,),
) -> QuantizedTarget:
    return QuantizedTarget(
        fqn=fqn,
        target_type="linear",
        algorithm="minmax",
        bits=8,
        granularity=granularity,
        symmetric=True,
        scale=scale,
        zero_point=(0,),
    )


def _metadata(*targets: QuantizedTarget) -> GraphMetadata:
    return GraphMetadata(
        schema_version=1,
        stage=GraphStage.QUANTIZED_DECODE,
        model_kind="test",
        input_abi=(),
        output_abi=(),
        quantized_targets=targets,
    )


def _linear_model(
    fqns: tuple[str, ...],
    *,
    node_order: tuple[int, ...] | None = None,
    output_types: bool = True,
    op_type: str = "MatMul",
    bias: bool = False,
) -> onnx.ModelProto:
    inputs: list[onnx.ValueInfoProto] = []
    outputs: list[onnx.ValueInfoProto] = []
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    value_info: list[onnx.ValueInfoProto] = []
    linear_nodes: list[onnx.NodeProto] = []
    for index, fqn in enumerate(fqns):
        source = f"source.{index}"
        output = f"output.{index}"
        weight_name = f"graph.{fqn}.weight"
        inputs.append(
            helper.make_tensor_value_info(source, TensorProto.FLOAT16, SHAPE)
        )
        if output_types:
            value_info.append(
                helper.make_tensor_value_info(
                    output,
                    TensorProto.FLOAT16,
                    SHAPE,
                )
            )
        initializers.append(
            numpy_helper.from_array(
                np.eye(2, dtype=np.float16),
                name=weight_name,
            )
        )
        node_inputs = [source, weight_name]
        attributes: dict[str, int] = {}
        if bias:
            bias_name = f"bias.{index}"
            node_inputs.append(bias_name)
            initializers.append(
                numpy_helper.from_array(
                    np.ones((2,), dtype=np.float16),
                    name=bias_name,
                )
            )
            attributes["transB"] = 1
        linear_nodes.append(
            helper.make_node(
                op_type,
                node_inputs,
                [output],
                name=f"linear.{index}",
                **attributes,
            )
        )
    order = node_order or tuple(range(len(linear_nodes)))
    nodes.extend(linear_nodes[index] for index in order)
    graph = helper.make_graph(
        nodes,
        "linear_fixture",
        inputs,
        outputs,
        initializers,
    )
    graph.value_info.extend(value_info)
    return helper.make_model(graph)


def _node_names(model: onnx.ModelProto) -> list[str]:
    return [node.name for node in model.graph.node]


def test_multiple_targets_replace_original_positions_and_cleanup_weights() -> None:
    model = _linear_model(("first", "second"), node_order=(1, 0))
    model.graph.node.insert(
        1,
        helper.make_node("Identity", ["source.0"], ["padding"], name="padding"),
    )
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.ones((1,), dtype=np.float32),
                name="graph.keep.weight",
            ),
            numpy_helper.from_array(
                np.ones((1,), dtype=np.float32),
                name="auxiliary",
            ),
        ]
    )
    model.graph.node.append(
        helper.make_node(
            "Identity",
            ["graph.keep.weight"],
            ["kept"],
            name="keep",
        )
    )

    linear.append_quantized_linears(
        model,
        _metadata(_target("first"), _target("second")),
    )

    assert _node_names(model) == [
        "mdc.linear.second.quant",
        "mdc.linear.second.matmul",
        "mdc.linear.second.dequant",
        "padding",
        "mdc.linear.first.quant",
        "mdc.linear.first.matmul",
        "mdc.linear.first.dequant",
        "keep",
    ]
    assert [
        sum(node.op_type == op_type for node in model.graph.node)
        for op_type in ("NPUAscendQuantV2", "MatMul", "AscendDequant")
    ] == [2, 2, 2]
    assert {node.output[-1] for node in model.graph.node if node.op_type == "AscendDequant"} == {
        "output.0",
        "output.1",
    }
    initializer_names = [item.name for item in model.graph.initializer]
    assert "graph.first.weight" not in initializer_names
    assert "graph.second.weight" not in initializer_names
    assert "graph.keep.weight" in initializer_names
    assert "auxiliary" in initializer_names


def test_gemm_bias_allocates_seven_names_and_appends_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _linear_model(("projection",), op_type="Gemm", bias=True)
    allocated: list[str] = []
    real_allocate = linear._LinearLoweringContext.unique_name

    def counted_allocate(
        context: linear._LinearLoweringContext,
        base: str,
    ) -> str:
        allocated.append(base)
        return real_allocate(context, base)

    monkeypatch.setattr(
        linear._LinearLoweringContext,
        "unique_name",
        counted_allocate,
    )

    linear.append_quantized_linears(model, _metadata(_target("projection")))

    prefix = "mdc.linear.projection"
    assert allocated == [
        f"{prefix}.quant_scale",
        f"{prefix}.quant_offset",
        f"{prefix}.weight",
        f"{prefix}.dequant_scale",
        f"{prefix}.quantized",
        f"{prefix}.accumulator",
        f"{prefix}.dequantized",
    ]
    assert [node.op_type for node in model.graph.node] == [
        "NPUAscendQuantV2",
        "MatMul",
        "AscendDequant",
        "Add",
    ]
    assert model.graph.node[-2].output == [f"{prefix}.dequantized"]
    assert model.graph.node[-1].input == [f"{prefix}.dequantized", "bias.0"]
    assert model.graph.node[-1].output == ["output.0"]


def test_name_allocation_uses_only_original_occupancy_categories() -> None:
    model = _linear_model(("projection",))
    base = "mdc.linear.projection.quant_scale"
    model.graph.initializer.append(
        numpy_helper.from_array(np.ones((1,), dtype=np.float32), name=base)
    )
    model.graph.input.append(
        helper.make_tensor_value_info(f"{base}.1", TensorProto.FLOAT, (1,))
    )
    model.graph.node.insert(
        0,
        helper.make_node(
            "Identity",
            ["source.0"],
            [f"{base}.3"],
            name=f"{base}.2",
        ),
    )
    model.graph.output.append(
        helper.make_tensor_value_info(f"{base}.2", TensorProto.FLOAT, (1,))
    )
    model.graph.value_info.append(
        helper.make_tensor_value_info(f"{base}.2", TensorProto.FLOAT, (1,))
    )

    linear.append_quantized_linears(model, _metadata(_target("projection")))

    assert f"{base}.2" in {
        item.name for item in model.graph.initializer
    }


def test_duplicate_initializer_lookup_remains_first_wins() -> None:
    model = _linear_model(("projection",))
    name = "graph.projection.weight"
    del model.graph.initializer[:]
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.full((2, 2), 0.25, dtype=np.float16),
                name=name,
            ),
            numpy_helper.from_array(
                np.full((2, 2), 0.5, dtype=np.float16),
                name=name,
            ),
        ]
    )

    linear.append_quantized_linears(model, _metadata(_target("projection")))

    packed = next(
        item
        for item in model.graph.initializer
        if item.name == "mdc.linear.projection.weight"
    )
    np.testing.assert_array_equal(
        numpy_helper.to_array(packed),
        np.ones((2, 2), dtype=np.int8),
    )


def test_context_types_match_model_types_and_append_last_wins() -> None:
    name = "shared"
    graph = helper.make_graph(
        [],
        "types",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, (1,))],
        [helper.make_tensor_value_info(name, TensorProto.INT8, (2,))],
        [
            numpy_helper.from_array(
                np.ones((4,), dtype=np.int32),
                name=name,
            )
        ],
    )
    graph.value_info.append(
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, (3,))
    )
    model = helper.make_model(graph)
    context = linear._LinearLoweringContext.from_model(model)

    assert context.types == support.model_types(model)
    assert context.types[name] == (TensorProto.INT32, (4,))
    context.append_value(name, TensorProto.INT64, (5,))
    assert context.types[name] == (TensorProto.INT64, (5,))


def test_context_registers_replacement_bucket_in_graph_order() -> None:
    nodes = [
        helper.make_node("MatMul", ["source", "weight.b"], ["before"], name="before"),
        helper.make_node("MatMul", ["source", "weight.a"], ["old"], name="old"),
        helper.make_node("MatMul", ["source", "weight.b"], ["after"], name="after"),
    ]
    model = helper.make_model(helper.make_graph(nodes, "buckets", [], []))
    context = linear._LinearLoweringContext.from_model(model)
    replacement = helper.make_node(
        "MatMul",
        ["source", "weight.b"],
        ["replacement"],
        name="replacement",
    )

    context.replace_node(context.linear_nodes("weight.a")[0], [replacement])

    assert [node.name for node in context.linear_nodes("weight.b")] == [
        "before",
        "replacement",
        "after",
    ]
    assert context.linear_nodes("weight.a") == []


def test_new_value_type_is_visible_to_later_target() -> None:
    model = _linear_model(("first", "second"))
    model.graph.node[1].input[0] = "mdc.linear.first.quantized"

    with pytest.raises(
        OnnxExportError,
        match=r"Linear target 'second' has unsupported dtypes",
    ):
        linear.append_quantized_linears(
            model,
            _metadata(_target("first"), _target("second")),
        )

    assert sum(node.op_type == "NPUAscendQuantV2" for node in model.graph.node) == 1
    assert any(item.name == "graph.first.weight" for item in model.graph.initializer)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda model: model.graph.initializer.clear(),
            "Cannot locate ONNX weight for linear target 'projection'",
        ),
        (
            lambda model: model.graph.node.clear(),
            "Linear target 'projection' maps to 0 standard ONNX nodes",
        ),
        (
            lambda model: model.graph.node.append(
                onnx.NodeProto().FromString(
                    model.graph.node[0].SerializeToString()
                )
            ),
            "Linear target 'projection' maps to 2 standard ONNX nodes",
        ),
        (
            lambda model: model.graph.node[0].output.clear(),
            "Linear node for 'projection' has an invalid ABI",
        ),
    ],
)
def test_early_errors_leave_graph_unchanged(
    mutate: Callable[[onnx.ModelProto], object],
    message: str,
) -> None:
    model = _linear_model(("projection",))
    mutate(model)
    before = model.SerializeToString()

    with pytest.raises(OnnxExportError, match=message.replace("'", r"\x27")):
        linear.append_quantized_linears(model, _metadata(_target("projection")))

    assert model.SerializeToString() == before


def test_missing_output_type_appends_metadata_before_activation_error() -> None:
    model = _linear_model(("projection",), output_types=False)
    before_initializers = [
        item.SerializeToString() for item in model.graph.initializer
    ]
    before_nodes = [node.SerializeToString() for node in model.graph.node]

    with pytest.raises(
        OnnxExportError,
        match=r"Linear activation for 'projection' must be symmetric per-tensor",
    ):
        linear.append_quantized_linears(
            model,
            _metadata(_target("projection", granularity="per_channel")),
        )

    assert [item.name for item in model.graph.value_info] == [
        "source.0",
        "output.0",
    ]
    assert [
        item.SerializeToString() for item in model.graph.initializer
    ] == before_initializers
    assert [
        node.SerializeToString() for node in model.graph.node
    ] == before_nodes


def test_offset_failure_preserves_appended_scale_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _linear_model(("projection",))
    before_nodes = [node.SerializeToString() for node in model.graph.node]

    def fail_offset(*args: object, **kwargs: object) -> str:
        raise RuntimeError("offset sentinel")

    monkeypatch.setattr(linear, "offset_initializer", fail_offset)

    with pytest.raises(RuntimeError, match="offset sentinel"):
        linear.append_quantized_linears(model, _metadata(_target("projection")))

    assert [item.name for item in model.graph.initializer] == [
        "graph.projection.weight",
        "mdc.linear.projection.quant_scale",
    ]
    assert not model.graph.value_info[-1].name.startswith("mdc.linear.")
    assert [
        node.SerializeToString() for node in model.graph.node
    ] == before_nodes


def test_duplicate_target_fails_at_second_match_before_later_missing_weight() -> None:
    model = _linear_model(("projection",))

    with pytest.raises(
        OnnxExportError,
        match=r"Linear target 'projection' maps to 0 standard ONNX nodes",
    ):
        linear.append_quantized_linears(
            model,
            _metadata(
                _target("projection"),
                _target("projection"),
                _target("missing"),
            ),
        )

    assert sum(node.op_type == "NPUAscendQuantV2" for node in model.graph.node) == 1
    assert any(
        item.name == "graph.projection.weight"
        for item in model.graph.initializer
    )
    assert any(
        item.name == "mdc.linear.projection.weight"
        for item in model.graph.initializer
    )


def test_initial_multiple_matches_fail_before_later_missing_weight() -> None:
    model = _linear_model(("projection",))
    duplicate = onnx.NodeProto()
    duplicate.CopyFrom(model.graph.node[0])
    model.graph.node.append(duplicate)
    before = model.SerializeToString()

    with pytest.raises(
        OnnxExportError,
        match=r"Linear target 'projection' maps to 2 standard ONNX nodes",
    ):
        linear.append_quantized_linears(
            model,
            _metadata(_target("projection"), _target("missing")),
        )

    assert model.SerializeToString() == before


def test_multiple_targets_build_types_once_and_skip_default_name_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _linear_model(("first", "second", "third"))
    calls = 0
    real_model_types = linear.model_types

    def counted_model_types(
        value: onnx.ModelProto,
    ) -> dict[str, tuple[int, tuple[int, ...]]]:
        nonlocal calls
        calls += 1
        return real_model_types(value)

    def reject_default_name_scan(
        model: onnx.ModelProto,
        base: str,
    ) -> str:
        raise AssertionError(f"default unique_name called for {base}")

    monkeypatch.setattr(linear, "model_types", counted_model_types)
    monkeypatch.setattr(support, "unique_name", reject_default_name_scan)

    linear.append_quantized_linears(
        model,
        _metadata(_target("third"), _target("first"), _target("second")),
    )

    assert calls == 1


def test_support_initializers_keep_default_allocator_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _linear_model(())
    target = _target("projection")
    bases: list[str] = []
    real_unique_name = support.unique_name

    def counted_unique_name(value: onnx.ModelProto, base: str) -> str:
        bases.append(base)
        return real_unique_name(value, base)

    monkeypatch.setattr(support, "unique_name", counted_unique_name)

    scale = support.scale_initializer(
        model,
        "scale",
        target,
        inverse=False,
    )
    offset = support.offset_initializer(model, "offset", target)

    assert (scale, offset) == ("scale", "offset")
    assert bases == ["scale", "offset"]
