from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

from mdc_llm_deploy.onnx.fusion_pass import (
    FusionPassResult,
    FusionReport,
    fuse_rms_norm,
)
from mdc_llm_deploy.onnx.schemas import RMS_NORM_OP, register_schemas

_SHAPE = (2, 3, 4)
_RSTD_SHAPE = (2, 3, 1)


class NPURmsNorm(OpRun):
    op_domain = ""

    def _run(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        epsilon: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        eps = 1e-6 if epsilon is None else epsilon
        squared_mean = np.mean(np.square(x.astype(np.float32)), axis=-1)
        rstd = np.reciprocal(np.sqrt(squared_mean + eps))
        y = x.astype(np.float32) * rstd[..., None] * gamma.astype(np.float32)
        return y.astype(x.dtype), rstd


@pytest.mark.parametrize(
    "elem_type",
    [TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.FLOAT],
)
def test_fuses_qwen3_fp32_accumulation_shape(elem_type: int) -> None:
    model = _rms_norm_model(elem_type)

    result = fuse_rms_norm(model)

    assert result == FusionPassResult("rms_norm", 1, ("y_npu_rms_norm",))
    fused = _only_node(model, RMS_NORM_OP)
    assert list(fused.input) == ["x", "gamma"]
    assert fused.output[0] == "y"
    assert len(fused.output) == 2
    assert fused.output[1] != "rstd_keepdims"
    assert helper.get_attribute_value(fused.attribute[0]) == pytest.approx(1e-6)
    assert not _nodes(model, "Pow")
    assert not _nodes(model, "ReduceMean")
    assert not _nodes(model, "Reciprocal")
    assert not _nodes(model, "Cast")
    assert {initializer.name for initializer in model.graph.initializer} == {"gamma"}
    _check_model(model)


def test_fused_graph_preserves_numerical_result_and_consumed_rstd() -> None:
    model = _rms_norm_model(TensorProto.FLOAT, consume_rstd=True)
    x = np.random.default_rng(7).normal(size=_SHAPE).astype(np.float32)
    expected = ReferenceEvaluator(model).run(None, {"x": x})

    result = fuse_rms_norm(model)
    actual = ReferenceEvaluator(model, new_ops=[NPURmsNorm]).run(None, {"x": x})

    assert result.fused_count == 1
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-6)
    unsqueeze = _only_node(model, "Unsqueeze")
    assert unsqueeze.output[0] == "rstd_keepdims"
    assert _only_node(model, "Identity").input[0] == "rstd_keepdims"
    _check_model(model)


def test_rejects_shared_non_rstd_intermediate_without_modifying_graph() -> None:
    model = _rms_norm_model(TensorProto.FLOAT, shared_value="mean")
    before = model.SerializeToString()

    result = fuse_rms_norm(model)

    assert result == FusionPassResult("rms_norm", 0)
    assert model.SerializeToString() == before


@pytest.mark.parametrize(
    ("power", "axes", "epsilon"),
    [
        (3.0, (-1,), 1e-6),
        (2.0, (-2,), 1e-6),
        (2.0, (-1,), 0.0),
    ],
)
def test_rejects_unproven_statistics_semantics(
    power: float,
    axes: tuple[int, ...],
    epsilon: float,
) -> None:
    model = _rms_norm_model(
        TensorProto.FLOAT,
        power=power,
        axes=axes,
        epsilon=epsilon,
    )
    before = model.SerializeToString()

    result = fuse_rms_norm(model)

    assert result.fused_count == 0
    assert model.SerializeToString() == before


def test_fuses_multiple_matches_and_resolves_name_collision() -> None:
    first = _rms_norm_nodes(TensorProto.FLOAT, prefix="first_")
    second = _rms_norm_nodes(TensorProto.FLOAT, prefix="second_")
    collision = helper.make_node("Identity", ["first_x"], ["collision"], name="first_y_npu_rms_norm")
    graph = helper.make_graph(
        [*first.nodes, *second.nodes, collision],
        "multi",
        [
            helper.make_tensor_value_info("first_x", TensorProto.FLOAT, _SHAPE),
            helper.make_tensor_value_info("second_x", TensorProto.FLOAT, _SHAPE),
        ],
        [
            helper.make_tensor_value_info("first_y", TensorProto.FLOAT, _SHAPE),
            helper.make_tensor_value_info("second_y", TensorProto.FLOAT, _SHAPE),
            helper.make_tensor_value_info("collision", TensorProto.FLOAT, _SHAPE),
        ],
        [*first.initializers, *second.initializers],
        value_info=[*first.value_info, *second.value_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    result = fuse_rms_norm(model)

    assert result.fused_count == 2
    assert len(set(result.fused_node_names)) == 2
    assert "first_y_npu_rms_norm_1" in result.fused_node_names
    assert len(_nodes(model, RMS_NORM_OP)) == 2
    _check_model(model)


def test_report_exposes_immutable_stable_counts() -> None:
    report = FusionReport(
        (
            FusionPassResult("rms_norm", 2, ("rms_0", "rms_1")),
            FusionPassResult("rope", 0),
        )
    )

    assert report.total_fused_count == 2
    assert report.counts == {"rms_norm": 2, "rope": 0}
    with pytest.raises(TypeError):
        report.counts["rms_norm"] = 3  # type: ignore[index]


class _PatternParts:
    def __init__(
        self,
        nodes: Sequence[onnx.NodeProto],
        initializers: Sequence[onnx.TensorProto],
        value_info: Sequence[onnx.ValueInfoProto],
    ) -> None:
        self.nodes = list(nodes)
        self.initializers = list(initializers)
        self.value_info = list(value_info)


def _rms_norm_model(
    elem_type: int,
    *,
    power: float = 2.0,
    axes: tuple[int, ...] = (-1,),
    epsilon: float = 1e-6,
    consume_rstd: bool = False,
    shared_value: str | None = None,
) -> onnx.ModelProto:
    parts = _rms_norm_nodes(
        elem_type,
        power=power,
        axes=axes,
        epsilon=epsilon,
    )
    nodes = list(parts.nodes)
    outputs = [helper.make_tensor_value_info("y", elem_type, _SHAPE)]
    if consume_rstd:
        nodes.append(helper.make_node("Identity", ["rstd_keepdims"], ["observed_rstd"]))
        outputs.append(
            helper.make_tensor_value_info("observed_rstd", TensorProto.FLOAT, _RSTD_SHAPE)
        )
    if shared_value is not None:
        nodes.append(helper.make_node("Identity", [shared_value], ["shared_output"]))
        outputs.append(
            helper.make_tensor_value_info("shared_output", TensorProto.FLOAT, _RSTD_SHAPE)
        )
    graph = helper.make_graph(
        nodes,
        "rms_norm",
        [helper.make_tensor_value_info("x", elem_type, _SHAPE)],
        outputs,
        parts.initializers,
        value_info=parts.value_info,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _rms_norm_nodes(
    elem_type: int,
    *,
    prefix: str = "",
    power: float = 2.0,
    axes: tuple[int, ...] = (-1,),
    epsilon: float = 1e-6,
) -> _PatternParts:
    def name(value: str) -> str:
        return f"{prefix}{value}"

    initializers = [
        helper.make_tensor(name("gamma"), elem_type, [4], [1.0, 0.5, 1.5, 2.0]),
        helper.make_tensor(name("power"), TensorProto.FLOAT, [], [power]),
        helper.make_tensor(name("axes"), TensorProto.INT64, [len(axes)], axes),
        helper.make_tensor(name("epsilon"), TensorProto.FLOAT, [], [epsilon]),
    ]
    nodes: list[onnx.NodeProto] = []
    accumulator = name("x")
    normalized = name("normalized")
    value_info: list[onnx.ValueInfoProto] = []
    if elem_type != TensorProto.FLOAT:
        accumulator = name("accumulator")
        nodes.append(
            helper.make_node("Cast", [name("x")], [accumulator], to=TensorProto.FLOAT)
        )
        value_info.append(helper.make_tensor_value_info(accumulator, TensorProto.FLOAT, _SHAPE))

    nodes.extend(
        [
            helper.make_node("Pow", [accumulator, name("power")], [name("squared")]),
            helper.make_node(
                "ReduceMean",
                [name("squared"), name("axes")],
                [name("mean")],
                keepdims=1,
                noop_with_empty_axes=0,
            ),
            helper.make_node("Add", [name("mean"), name("epsilon")], [name("variance")]),
            helper.make_node("Sqrt", [name("variance")], [name("root")]),
            helper.make_node("Reciprocal", [name("root")], [name("rstd_keepdims")]),
            helper.make_node(
                "Mul",
                [accumulator, name("rstd_keepdims")],
                [name("normalized_fp32")],
            ),
        ]
    )
    if elem_type != TensorProto.FLOAT:
        nodes.append(
            helper.make_node(
                "Cast",
                [name("normalized_fp32")],
                [normalized],
                to=elem_type,
            )
        )
    else:
        normalized = name("normalized_fp32")
    nodes.append(helper.make_node("Mul", [name("gamma"), normalized], [name("y")]))

    value_info.extend(
        [
            helper.make_tensor_value_info(name("squared"), TensorProto.FLOAT, _SHAPE),
            helper.make_tensor_value_info(name("mean"), TensorProto.FLOAT, _RSTD_SHAPE),
            helper.make_tensor_value_info(name("variance"), TensorProto.FLOAT, _RSTD_SHAPE),
            helper.make_tensor_value_info(name("root"), TensorProto.FLOAT, _RSTD_SHAPE),
            helper.make_tensor_value_info(name("rstd_keepdims"), TensorProto.FLOAT, _RSTD_SHAPE),
            helper.make_tensor_value_info(name("normalized_fp32"), TensorProto.FLOAT, _SHAPE),
        ]
    )
    if elem_type != TensorProto.FLOAT:
        value_info.append(helper.make_tensor_value_info(normalized, elem_type, _SHAPE))
    return _PatternParts(nodes, initializers, value_info)


def _nodes(model: onnx.ModelProto, op_type: str) -> list[onnx.NodeProto]:
    return [node for node in model.graph.node if node.op_type == op_type]


def _only_node(model: onnx.ModelProto, op_type: str) -> onnx.NodeProto:
    nodes = _nodes(model, op_type)
    assert len(nodes) == 1
    return nodes[0]


def _check_model(model: onnx.ModelProto) -> None:
    register_schemas(RMS_NORM_OP)
    onnx.checker.check_model(model, full_check=True)
