from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import mdc_llm_deploy.onnx.export.constant_folding as folding_module
from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.onnx.export.constant_folding import (
    ConstantFoldingStats,
    fold_constant_subgraphs,
)


def _initializer(name: str, value: object, dtype: np.dtype[object]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def _model(
    nodes: Sequence[onnx.NodeProto],
    initializers: Sequence[onnx.TensorProto],
    *,
    inputs: Sequence[onnx.ValueInfoProto] = (),
    outputs: Sequence[str] = ("output",),
    output_dtype: int = TensorProto.FLOAT,
    output_shape: Sequence[int | None] | None = None,
    opset: int = 18,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        list(nodes),
        "constant-folding",
        list(inputs),
        [
            helper.make_tensor_value_info(name, output_dtype, output_shape)
            for name in outputs
        ],
        initializer=list(initializers),
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _array(model: onnx.ModelProto, name: str) -> np.ndarray[object, object]:
    initializer = next(item for item in model.graph.initializer if item.name == name)
    return np.asarray(numpy_helper.to_array(initializer))


def test_fixed_point_folds_reverse_order_broadcast_chain_deterministically() -> None:
    model = _model(
        [
            helper.make_node("Neg", ["sum"], ["output"], name="second"),
            helper.make_node("Add", ["matrix", "row"], ["sum"], name="first"),
        ],
        [
            _initializer("matrix", [[1.0], [2.0]], np.dtype(np.float32)),
            _initializer("row", [[10.0, 20.0, 30.0]], np.dtype(np.float32)),
        ],
        output_shape=(2, 3),
    )

    stats = fold_constant_subgraphs(model)

    assert stats == ConstantFoldingStats(
        folded_nodes=2,
        materialized_initializers=2,
        materialized_bytes=48,
        skipped_nodes=0,
        skipped_by_reason=(),
    )
    assert not model.graph.node
    np.testing.assert_array_equal(
        _array(model, "output"),
        -np.asarray([[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]], dtype=np.float32),
    )
    serialized = model.SerializeToString(deterministic=True)
    assert fold_constant_subgraphs(model).folded_nodes == 0
    assert model.SerializeToString(deterministic=True) == serialized


@pytest.mark.parametrize(
    ("node", "initializers", "expected"),
    [
        pytest.param(
            helper.make_node("Transpose", ["data"], ["output"], perm=[1, 0]),
            [_initializer("data", [[1, 2, 3], [4, 5, 6]], np.dtype(np.int64))],
            [[1, 4], [2, 5], [3, 6]],
            id="transpose",
        ),
        pytest.param(
            helper.make_node("Reshape", ["data", "shape"], ["output"]),
            [
                _initializer("data", [1, 2, 3, 4], np.dtype(np.int64)),
                _initializer("shape", [2, -1], np.dtype(np.int64)),
            ],
            [[1, 2], [3, 4]],
            id="reshape",
        ),
        pytest.param(
            helper.make_node("Squeeze", ["data", "axes"], ["output"]),
            [
                _initializer("data", [[[1], [2]]], np.dtype(np.int64)),
                _initializer("axes", [0, -1], np.dtype(np.int64)),
            ],
            [1, 2],
            id="squeeze-negative-axis",
        ),
        pytest.param(
            helper.make_node("Unsqueeze", ["data", "axes"], ["output"]),
            [
                _initializer("data", [1, 2], np.dtype(np.int64)),
                _initializer("axes", [0, -1], np.dtype(np.int64)),
            ],
            [[[1], [2]]],
            id="unsqueeze-negative-axis",
        ),
        pytest.param(
            helper.make_node("Concat", ["left", "right"], ["output"], axis=-1),
            [
                _initializer("left", [[1], [2]], np.dtype(np.int64)),
                _initializer("right", [[3, 4], [5, 6]], np.dtype(np.int64)),
            ],
            [[1, 3, 4], [2, 5, 6]],
            id="concat-negative-axis",
        ),
        pytest.param(
            helper.make_node("Gather", ["data", "indices"], ["output"], axis=-1),
            [
                _initializer("data", [[10, 20, 30], [40, 50, 60]], np.dtype(np.int64)),
                _initializer("indices", [-1, 0], np.dtype(np.int64)),
            ],
            [[30, 10], [60, 40]],
            id="gather-negative-index",
        ),
        pytest.param(
            helper.make_node("Sub", ["left", "right"], ["output"]),
            [
                _initializer("left", [5, 8], np.dtype(np.int64)),
                _initializer("right", [2, 3], np.dtype(np.int64)),
            ],
            [3, 5],
            id="sub",
        ),
        pytest.param(
            helper.make_node("Mul", ["left", "right"], ["output"]),
            [
                _initializer("left", [5, 8], np.dtype(np.int64)),
                _initializer("right", [2, 3], np.dtype(np.int64)),
            ],
            [10, 24],
            id="mul",
        ),
        pytest.param(
            helper.make_node("Div", ["left", "right"], ["output"]),
            [
                _initializer("left", [-5, 8], np.dtype(np.int64)),
                _initializer("right", [2, 3], np.dtype(np.int64)),
            ],
            [-2, 2],
            id="integer-div-truncates-toward-zero",
        ),
    ],
)
def test_whitelist_evaluators(
    node: onnx.NodeProto,
    initializers: list[onnx.TensorProto],
    expected: object,
) -> None:
    model = _model(
        [node],
        initializers,
        output_dtype=TensorProto.INT64,
        output_shape=None,
    )

    stats = fold_constant_subgraphs(model)

    assert stats.folded_nodes == 1
    np.testing.assert_array_equal(_array(model, "output"), expected)


def test_slice_supports_negative_indices_steps_and_omitted_axes() -> None:
    model = _model(
        [
            helper.make_node(
                "Slice",
                ["data", "starts", "ends", "", "steps"],
                ["output"],
            )
        ],
        [
            _initializer("data", [0, 1, 2, 3, 4, 5], np.dtype(np.int64)),
            _initializer("starts", [-2], np.dtype(np.int64)),
            _initializer("ends", [-7], np.dtype(np.int64)),
            _initializer("steps", [-2], np.dtype(np.int64)),
        ],
        output_dtype=TensorProto.INT64,
        output_shape=(3,),
    )

    fold_constant_subgraphs(model)

    np.testing.assert_array_equal(_array(model, "output"), [4, 2, 0])


def test_reshape_allowzero_preserves_zero_dimension() -> None:
    model = _model(
        [
            helper.make_node(
                "Reshape",
                ["data", "shape"],
                ["output"],
                allowzero=1,
            )
        ],
        [
            _initializer("data", np.empty((0, 3)), np.dtype(np.float32)),
            _initializer("shape", [0, 3], np.dtype(np.int64)),
        ],
        output_shape=(0, 3),
    )

    fold_constant_subgraphs(model)

    assert _array(model, "output").shape == (0, 3)


def test_cast_materializes_bfloat16_initializer() -> None:
    model = _model(
        [
            helper.make_node(
                "Cast",
                ["data"],
                ["output"],
                to=TensorProto.BFLOAT16,
            )
        ],
        [_initializer("data", [1.5, -2.25], np.dtype(np.float32))],
        output_dtype=TensorProto.BFLOAT16,
        output_shape=(2,),
    )

    fold_constant_subgraphs(model)

    initializer = next(item for item in model.graph.initializer if item.name == "output")
    assert initializer.data_type == TensorProto.BFLOAT16
    np.testing.assert_array_equal(
        numpy_helper.to_array(initializer).astype(np.float32),
        np.asarray([1.5, -2.25], dtype=np.float32),
    )


def test_shared_constant_chain_is_folded_once_and_reused() -> None:
    model = _model(
        [
            helper.make_node("Identity", ["source"], ["shared"], name="alias"),
            helper.make_node("Add", ["shared", "shared"], ["output"], name="consumer"),
        ],
        [_initializer("source", [1.0, 2.0], np.dtype(np.float32))],
        output_shape=(2,),
    )

    stats = fold_constant_subgraphs(model)

    assert stats.folded_nodes == 2
    assert [item.name for item in model.graph.initializer].count("shared") == 1
    np.testing.assert_array_equal(_array(model, "output"), [2.0, 4.0])


def test_graph_output_can_fold_but_overridable_graph_input_cannot() -> None:
    input_info = helper.make_tensor_value_info("source", TensorProto.FLOAT, (2,))
    model = _model(
        [helper.make_node("Identity", ["source"], ["output"])],
        [_initializer("source", [1.0, 2.0], np.dtype(np.float32))],
        inputs=[input_info],
        output_shape=(2,),
    )

    stats = fold_constant_subgraphs(model)

    assert stats.folded_nodes == 0
    assert len(model.graph.node) == 1
    assert not any(item.name == "output" for item in model.graph.initializer)


@pytest.mark.parametrize(
    ("budget", "expected_reason"),
    [
        pytest.param(
            folding_module._FoldingBudget(max_output_bytes=4),
            "output_bytes_limit",
            id="single-output",
        ),
        pytest.param(
            folding_module._FoldingBudget(max_total_bytes=4),
            "total_bytes_limit",
            id="total-materialization",
        ),
        pytest.param(
            folding_module._FoldingBudget(max_rank=0),
            "rank_limit",
            id="rank",
        ),
        pytest.param(
            folding_module._FoldingBudget(max_nodes=0),
            "node_limit",
            id="node-count",
        ),
        pytest.param(
            folding_module._FoldingBudget(max_expansion=0),
            "expansion_limit",
            id="expansion",
        ),
    ],
)
def test_budget_rejection_leaves_node_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    budget: folding_module._FoldingBudget,
    expected_reason: str,
) -> None:
    model = _model(
        [helper.make_node("Identity", ["source"], ["output"], name="bounded")],
        [_initializer("source", [1.0, 2.0], np.dtype(np.float32))],
        output_shape=(2,),
    )
    original = copy.deepcopy(model)
    monkeypatch.setattr(folding_module, "_DEFAULT_BUDGET", budget)

    stats = fold_constant_subgraphs(model)

    assert stats.folded_nodes == 0
    assert stats.skipped_nodes == 1
    assert stats.skipped_by_reason == ((expected_reason, 1),)
    assert model.SerializeToString(deterministic=True) == original.SerializeToString(
        deterministic=True
    )


def test_invalid_constant_node_failure_is_atomic() -> None:
    model = _model(
        [
            helper.make_node("Identity", ["source"], ["valid"], name="valid"),
            helper.make_node(
                "Reshape",
                ["valid", "bad_shape"],
                ["output"],
                name="invalid",
            ),
        ],
        [
            _initializer("source", [1.0, 2.0], np.dtype(np.float32)),
            _initializer("bad_shape", [3], np.dtype(np.int64)),
        ],
        output_shape=(3,),
    )
    original = model.SerializeToString(deterministic=True)

    with pytest.raises(OnnxExportError, match=r"invalid.*element count"):
        fold_constant_subgraphs(model)

    assert model.SerializeToString(deterministic=True) == original


def test_non_whitelist_and_custom_domain_nodes_are_not_folded() -> None:
    model = _model(
        [
            helper.make_node("Exp", ["source"], ["standard_hidden"]),
            helper.make_node(
                "Add",
                ["source", "source"],
                ["output"],
                domain="vendor",
            ),
        ],
        [_initializer("source", [1.0, 2.0], np.dtype(np.float32))],
        output_shape=(2,),
    )

    stats = fold_constant_subgraphs(model)

    assert stats.folded_nodes == 0
    assert len(model.graph.node) == 2
