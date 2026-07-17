"""Behavioral tests for exact-identity linear initializer binding."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn
from torch.fx import Graph, GraphModule

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.onnx.export.linear_binding import (
    canonicalize_linear_initializers,
)


def _linear_graph(
    weights: dict[str, torch.Tensor],
    *,
    calls: tuple[str, ...] | None = None,
) -> GraphModule:
    root = nn.Module()
    for name, value in weights.items():
        root.register_parameter(name, nn.Parameter(value.clone()))
    graph = Graph()
    value = graph.placeholder("x")
    for name in calls or tuple(weights):
        weight = graph.get_attr(name)
        value = graph.call_function(
            torch.ops.aten.linear.default,
            (value, weight, None),
        )
    graph.output(value)
    return GraphModule(root, graph)


def _initializer(
    value: torch.Tensor,
    name: str,
    *,
    transpose: bool,
) -> onnx.TensorProto:
    stored = value.T if transpose else value
    stored = stored.detach().contiguous().cpu()
    if stored.dtype == torch.bfloat16:
        initializer = onnx.TensorProto()
        initializer.name = name
        initializer.data_type = TensorProto.BFLOAT16
        initializer.dims.extend(stored.shape)
        initializer.raw_data = stored.view(torch.uint16).numpy().tobytes()
        return initializer
    return numpy_helper.from_array(stored.numpy(), name=name)


def _linear_model(
    specifications: tuple[tuple[str, torch.Tensor, str, int | None], ...],
    *,
    initializer_order: tuple[int, ...] | None = None,
    node_order: tuple[int, ...] | None = None,
) -> onnx.ModelProto:
    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []
    for index, (name, value, op_type, trans_b) in enumerate(specifications):
        transpose = op_type == "MatMul" or trans_b == 0
        initializers.append(_initializer(value, name, transpose=transpose))
        attributes = {} if trans_b is None else {"transB": trans_b}
        nodes.append(
            helper.make_node(
                op_type,
                [f"x_{index}", name],
                [f"y_{index}"],
                name=f"linear_{index}",
                **attributes,
            )
        )
    initializer_indices = initializer_order or tuple(range(len(initializers)))
    node_indices = node_order or tuple(range(len(nodes)))
    graph = helper.make_graph(
        [nodes[index] for index in node_indices],
        "linear_binding",
        [
            helper.make_tensor_value_info(
                f"x_{index}",
                TensorProto.FLOAT,
                (1, int(specification[1].shape[1])),
            )
            for index, specification in enumerate(specifications)
        ],
        [
            helper.make_tensor_value_info(
                f"y_{index}",
                TensorProto.FLOAT,
                (1, int(specification[1].shape[0])),
            )
            for index, specification in enumerate(specifications)
        ],
        initializer=[initializers[index] for index in initializer_indices],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _serialized(model: onnx.ModelProto) -> bytes:
    return model.SerializeToString(deterministic=True)


def test_binding_uses_content_across_node_and_initializer_reordering() -> None:
    first = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    second = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    unrelated = torch.tensor([[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])
    graph = _linear_graph({"first": first, "second": second})
    model = _linear_model(
        (
            ("temporary_second", second, "MatMul", None),
            ("embed_constant", unrelated, "MatMul", None),
            ("temporary_first", first, "MatMul", None),
        ),
        initializer_order=(2, 0, 1),
        node_order=(1, 0, 2),
    )

    canonicalize_linear_initializers(model, graph)

    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    assert set(initializers) == {
        "graph.first",
        "graph.second",
        "embed_constant",
    }
    np.testing.assert_array_equal(initializers["graph.first"], first.T.numpy())
    np.testing.assert_array_equal(initializers["graph.second"], second.T.numpy())
    assert next(node for node in model.graph.node if node.name == "linear_1").input[1] == (
        "embed_constant"
    )


@pytest.mark.parametrize(
    ("dtype", "op_type", "trans_b"),
    [
        pytest.param(torch.float16, "MatMul", None, id="float16-matmul"),
        pytest.param(torch.float32, "Gemm", 0, id="float32-gemm-transB-0"),
        pytest.param(torch.bfloat16, "Gemm", 1, id="bfloat16-gemm-transB-1"),
    ],
)
def test_binding_understands_dtype_and_effective_weight_layout(
    dtype: torch.dtype,
    op_type: str,
    trans_b: int | None,
) -> None:
    bits = torch.tensor(
        [[0x0000, 0x8000, 0x3F80], [0x7FC1, 0x4000, 0x4040]],
        dtype=torch.uint32 if dtype == torch.float32 else torch.uint16,
    )
    value = bits.view(dtype)
    graph = _linear_graph({"projection": value})
    model = _linear_model((("temporary", value, op_type, trans_b),))

    canonicalize_linear_initializers(model, graph)

    assert model.graph.initializer[0].name == "graph.projection"
    assert model.graph.node[0].input[1] == "graph.projection"


def test_shared_fx_parameter_updates_every_onnx_reference() -> None:
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    graph = _linear_graph({"shared": weight}, calls=("shared", "shared"))
    model = _linear_model((("temporary", weight, "MatMul", None),))
    model.graph.node.append(
        helper.make_node("MatMul", ["other", "temporary"], ["other_output"])
    )
    model.graph.node.append(
        helper.make_node("Identity", ["temporary"], ["weight_copy"])
    )

    canonicalize_linear_initializers(model, graph)

    assert model.graph.initializer[0].name == "graph.shared"
    assert [
        input_name
        for node in model.graph.node
        for input_name in node.input
        if input_name == "graph.shared"
    ] == ["graph.shared", "graph.shared", "graph.shared"]


@pytest.mark.parametrize(
    ("build_graph", "build_model", "message"),
    [
        pytest.param(
            lambda weight: _linear_graph(
                {"missing": weight},
                calls=("missing", "missing"),
            ),
            lambda weight: _linear_model(
                (("unrelated", weight + 1, "MatMul", None),)
            ),
            r"Missing ONNX initializer for FX linear parameter 'missing' "
            r"\(FX call count: 2\)",
            id="missing",
        ),
        pytest.param(
            lambda weight: _linear_graph({"first": weight, "second": weight}),
            lambda weight: _linear_model(
                (("temporary", weight, "MatMul", None),)
            ),
            "Ambiguous FX linear parameters have identical tensor identity",
            id="ambiguous-fx",
        ),
        pytest.param(
            lambda weight: _linear_graph(
                {"projection": weight},
                calls=("projection", "projection"),
            ),
            lambda weight: _linear_model(
                (
                    ("first_copy", weight, "MatMul", None),
                    ("second_copy", weight, "MatMul", None),
                )
            ),
            r"Ambiguous ONNX initializers for FX linear parameter 'projection' "
            r"\(FX call count: 2\)",
            id="ambiguous-onnx",
        ),
    ],
)
def test_binding_failures_are_transactional(
    build_graph: Callable[[torch.Tensor], GraphModule],
    build_model: Callable[[torch.Tensor], onnx.ModelProto],
    message: str,
) -> None:
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    graph = build_graph(weight)
    model = build_model(weight)
    before = _serialized(model)

    with pytest.raises(OnnxExportError, match=message):
        canonicalize_linear_initializers(model, graph)

    assert _serialized(model) == before


@pytest.mark.parametrize(
    ("replace_initializer", "message"),
    [
        pytest.param(
            lambda value: numpy_helper.from_array(
                value.T.contiguous().numpy().reshape(-1),
                name="temporary",
            ),
            r"Unsupported ONNX linear weight representation while binding "
            r"FX linear parameter 'projection' \(FX call count: 2\): "
            r"'temporary' \(shape=\(6,\), dtype=FLOAT\)",
            id="rank",
        ),
        pytest.param(
            lambda value: numpy_helper.from_array(
                np.arange(value.numel(), dtype=np.int32).reshape(
                    value.shape[1],
                    value.shape[0],
                ),
                name="temporary",
            ),
            r"Unsupported ONNX linear weight representation while binding "
            r"FX linear parameter 'projection' \(FX call count: 2\): "
            r"'temporary' \(shape=\(3, 2\), dtype=INT32\)",
            id="dtype",
        ),
        pytest.param(
            lambda value: None,
            r"Missing ONNX initializer for FX linear parameter 'projection' "
            r"\(FX call count: 2\)",
            id="missing",
        ),
    ],
)
def test_unsupported_and_missing_onnx_weights_are_distinct_and_transactional(
    replace_initializer: Callable[[torch.Tensor], onnx.TensorProto | None],
    message: str,
) -> None:
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    graph = _linear_graph(
        {"projection": weight},
        calls=("projection", "projection"),
    )
    model = _linear_model((("temporary", weight, "MatMul", None),))
    replacement = replace_initializer(weight)
    model.graph.initializer.clear()
    if replacement is not None:
        model.graph.initializer.append(replacement)
    before = _serialized(model)

    with pytest.raises(OnnxExportError, match=message):
        canonicalize_linear_initializers(model, graph)

    assert _serialized(model) == before


def test_unrelated_unsupported_matmul_constant_is_ignored() -> None:
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    graph = _linear_graph({"projection": weight})
    model = _linear_model((("temporary", weight, "MatMul", None),))
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.ones((5,), dtype=np.int64),
            name="unrelated",
        )
    )
    model.graph.node.append(
        helper.make_node("MatMul", ["other", "unrelated"], ["other_output"])
    )

    canonicalize_linear_initializers(model, graph)

    assert {item.name for item in model.graph.initializer} == {
        "graph.projection",
        "unrelated",
    }


def test_rename_conflict_is_transactional() -> None:
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    graph = _linear_graph({"projection": weight})
    model = _linear_model((("temporary", weight, "MatMul", None),))
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.ones((1,), dtype=np.float32),
            name="graph.projection",
        )
    )
    before = _serialized(model)

    with pytest.raises(
        OnnxExportError,
        match=r"target names are occupied: 'graph\.projection'",
    ):
        canonicalize_linear_initializers(model, graph)

    assert _serialized(model) == before


def test_conflicting_gemm_layout_is_rejected_transactionally() -> None:
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    graph = _linear_graph({"projection": weight})
    model = _linear_model((("temporary", weight, "Gemm", 0),))
    model.graph.node.append(
        helper.make_node(
            "Gemm",
            ["other", "temporary"],
            ["other_output"],
            transB=1,
        )
    )
    before = _serialized(model)

    with pytest.raises(
        OnnxExportError,
        match="initializer is used with conflicting layouts",
    ):
        canonicalize_linear_initializers(model, graph)

    assert _serialized(model) == before
