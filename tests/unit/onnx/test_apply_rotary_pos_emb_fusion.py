from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

from mdc_llm_deploy.onnx.fusion_pass.apply_rotary_pos_emb import (
    fuse_apply_rotary_pos_emb,
)
from mdc_llm_deploy.onnx.schemas import (
    ROTARY_POSITION_EMBEDDING_OP,
    register_schemas,
)

_SEQUENCE_LENGTH = 3
_HEAD_DIMENSION = 8


class ApplyRotaryPosEmb(OpRun):
    op_domain = ""

    def _run(
        self,
        query: np.ndarray,
        key: np.ndarray,
        cos: np.ndarray,
        sin: np.ndarray,
        layout: int | None = None,
        rotary_mode: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert layout == 3
        assert rotary_mode == "half"

        def rotate_half(value: np.ndarray) -> np.ndarray:
            first, second = np.split(value, 2, axis=-1)
            return np.concatenate((-second, first), axis=-1)

        return (
            query * cos + rotate_half(query) * sin,
            key * cos + rotate_half(key) * sin,
        )


@pytest.mark.parametrize(
    "elem_type",
    [TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.FLOAT],
)
@pytest.mark.parametrize(("query_heads", "key_heads"), [(4, 4), (4, 2)])
@pytest.mark.parametrize("coefficient_variant", ["direct", "unsqueeze", "cast_unsqueeze"])
def test_fuses_bnsd_mha_and_gqa_variants(
    elem_type: int,
    query_heads: int,
    key_heads: int,
    coefficient_variant: str,
) -> None:
    model = _rope_model(
        elem_type,
        query_heads=query_heads,
        key_heads=key_heads,
        coefficient_variant=coefficient_variant,
    )

    result = fuse_apply_rotary_pos_emb(model)

    assert result.fused_count == 1
    fused = _only_node(model, ROTARY_POSITION_EMBEDDING_OP)
    assert list(fused.input) == ["query", "key", "cos", "sin"]
    assert list(fused.output) == ["query_out", "key_out"]
    assert _attribute(fused, "layout") == 3
    assert _attribute(fused, "rotary_mode") == b"half"
    assert not _nodes(model, "Slice")
    assert not _nodes(model, "Neg")
    assert not _nodes(model, "Concat")
    assert not _nodes(model, "Add")
    _check_model(model)


def test_fused_graph_preserves_fp32_results_and_output_names() -> None:
    model = _rope_model(TensorProto.FLOAT, query_heads=4, key_heads=2)
    generator = np.random.default_rng(11)
    feeds = {
        "query": generator.normal(size=(1, 4, 3, 8)).astype(np.float32),
        "key": generator.normal(size=(1, 2, 3, 8)).astype(np.float32),
        "cos": generator.normal(size=(1, 1, 3, 8)).astype(np.float32),
        "sin": generator.normal(size=(1, 1, 3, 8)).astype(np.float32),
    }
    expected = ReferenceEvaluator(model).run(None, feeds)

    fuse_apply_rotary_pos_emb(model)
    actual = ReferenceEvaluator(model, new_ops=[ApplyRotaryPosEmb]).run(None, feeds)

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-6)
    assert [output.name for output in model.graph.output] == ["query_out", "key_out"]


def test_fuses_commuted_arithmetic_and_slice_default_step() -> None:
    model = _rope_model(
        TensorProto.FLOAT,
        query_heads=4,
        key_heads=2,
        commute_arithmetic=True,
        omit_slice_steps=True,
    )

    result = fuse_apply_rotary_pos_emb(model)

    assert result.fused_count == 1
    _check_model(model)


def test_preserves_shared_coefficient_producer_and_external_consumer() -> None:
    model = _rope_model(
        TensorProto.FLOAT16,
        query_heads=4,
        key_heads=2,
        coefficient_variant="cast_unsqueeze",
        observe_cos=True,
    )

    result = fuse_apply_rotary_pos_emb(model)

    assert result.fused_count == 1
    assert _only_node(model, "Identity").input[0] == "cos"
    assert len(_nodes(model, "Cast")) == 2
    assert len(_nodes(model, "Unsqueeze")) == 2
    _check_model(model)


@pytest.mark.parametrize(
    "mutation",
    [
        "shared_rotation",
        "invalid_broadcast",
        "invalid_gqa_heads",
        "wrong_half_split",
        "third_branch",
    ],
)
def test_rejects_unproven_or_open_patterns_without_modification(mutation: str) -> None:
    model = _rope_model(
        TensorProto.FLOAT,
        query_heads=4,
        key_heads=2 if mutation != "invalid_gqa_heads" else 3,
        cos_shape=(2, 1, 3, 8) if mutation == "invalid_broadcast" else (1, 1, 3, 8),
        shared_rotation=mutation == "shared_rotation",
        wrong_half_split=mutation == "wrong_half_split",
        third_branch=mutation == "third_branch",
    )
    before = model.SerializeToString()

    result = fuse_apply_rotary_pos_emb(model)

    assert result.fused_count == 0
    assert model.SerializeToString() == before


def test_resolves_fused_node_name_collision() -> None:
    model = _rope_model(TensorProto.FLOAT, query_heads=4, key_heads=2)
    model.graph.node.append(
        helper.make_node(
            "Identity",
            ["query"],
            ["collision_output"],
            name="query_out_apply_rotary_pos_emb",
        )
    )
    model.graph.output.append(
        helper.make_tensor_value_info(
            "collision_output",
            TensorProto.FLOAT,
            (1, 4, 3, 8),
        )
    )

    result = fuse_apply_rotary_pos_emb(model)

    assert result.fused_node_names == ("query_out_apply_rotary_pos_emb_1",)
    _check_model(model)


@dataclass
class _GraphParts:
    nodes: list[onnx.NodeProto]
    inputs: list[onnx.ValueInfoProto]
    outputs: list[onnx.ValueInfoProto]
    value_info: list[onnx.ValueInfoProto]
    initializers: list[onnx.TensorProto]


def _rope_model(
    elem_type: int,
    *,
    query_heads: int,
    key_heads: int,
    coefficient_variant: str = "direct",
    cos_shape: tuple[int, int, int, int] = (1, 1, 3, 8),
    observe_cos: bool = False,
    shared_rotation: bool = False,
    wrong_half_split: bool = False,
    third_branch: bool = False,
    commute_arithmetic: bool = False,
    omit_slice_steps: bool = False,
) -> onnx.ModelProto:
    parts = _coefficient_parts(elem_type, coefficient_variant, cos_shape)
    parts.inputs.extend(
        [
            _value("query", elem_type, (1, query_heads, 3, 8)),
            _value("key", elem_type, (1, key_heads, 3, 8)),
        ]
    )
    parts.nodes.extend(
        _rotation_branch(
            "query",
            "query_out",
            elem_type,
            (1, query_heads, 3, 8),
            wrong_half_split=wrong_half_split,
            commute_arithmetic=commute_arithmetic,
            omit_slice_steps=omit_slice_steps,
        )
    )
    parts.nodes.extend(
        _rotation_branch(
            "key",
            "key_out",
            elem_type,
            (1, key_heads, 3, 8),
            commute_arithmetic=commute_arithmetic,
            omit_slice_steps=omit_slice_steps,
        )
    )
    parts.outputs.extend(
        [
            _value("query_out", elem_type, (1, query_heads, 3, 8)),
            _value("key_out", elem_type, (1, key_heads, 3, 8)),
        ]
    )
    for prefix, heads in (("query", query_heads), ("key", key_heads)):
        parts.value_info.extend(_rotation_value_info(prefix, elem_type, heads))

    if shared_rotation:
        parts.nodes.append(helper.make_node("Identity", ["query_concat"], ["observed_rotation"]))
        parts.outputs.append(_value("observed_rotation", elem_type, (1, query_heads, 3, 8)))
    if observe_cos:
        parts.nodes.append(helper.make_node("Identity", ["cos"], ["observed_cos"]))
        parts.outputs.append(_value("observed_cos", elem_type, cos_shape))
    if third_branch:
        parts.inputs.append(_value("extra", elem_type, (1, key_heads, 3, 8)))
        parts.nodes.extend(
            _rotation_branch("extra", "extra_out", elem_type, (1, key_heads, 3, 8))
        )
        parts.outputs.append(_value("extra_out", elem_type, (1, key_heads, 3, 8)))
        parts.value_info.extend(_rotation_value_info("extra", elem_type, key_heads))

    graph = helper.make_graph(
        parts.nodes,
        "rope",
        parts.inputs,
        parts.outputs,
        parts.initializers,
        value_info=parts.value_info,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _coefficient_parts(
    elem_type: int,
    variant: str,
    final_shape: tuple[int, int, int, int],
) -> _GraphParts:
    nodes: list[onnx.NodeProto] = []
    inputs: list[onnx.ValueInfoProto] = []
    value_info: list[onnx.ValueInfoProto] = []
    initializers = [
        helper.make_tensor("starts_zero", TensorProto.INT64, [1], [0]),
        helper.make_tensor("starts_half", TensorProto.INT64, [1], [4]),
        helper.make_tensor("ends_half", TensorProto.INT64, [1], [4]),
        helper.make_tensor("ends_all", TensorProto.INT64, [1], [np.iinfo(np.int64).max]),
        helper.make_tensor("slice_axis", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("slice_step", TensorProto.INT64, [1], [1]),
    ]
    if variant == "direct":
        inputs.extend([_value("cos", elem_type, final_shape), _value("sin", elem_type, final_shape)])
    else:
        base_shape = (final_shape[0], final_shape[2], final_shape[3])
        source_type = TensorProto.FLOAT if variant == "cast_unsqueeze" else elem_type
        axes_name = "coefficient_axes"
        initializers.append(helper.make_tensor(axes_name, TensorProto.INT64, [1], [1]))
        for coefficient in ("cos", "sin"):
            source_name = f"{coefficient}_source"
            unsqueezed_input = source_name
            inputs.append(_value(source_name, source_type, base_shape))
            if variant == "cast_unsqueeze":
                unsqueezed_input = f"{coefficient}_cast"
                nodes.append(
                    helper.make_node(
                        "Cast",
                        [source_name],
                        [unsqueezed_input],
                        to=elem_type,
                    )
                )
                value_info.append(_value(unsqueezed_input, elem_type, base_shape))
            nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    [unsqueezed_input, axes_name],
                    [coefficient],
                )
            )
            value_info.append(_value(coefficient, elem_type, final_shape))
    return _GraphParts(nodes, inputs, [], value_info, initializers)


def _rotation_branch(
    input_name: str,
    output_name: str,
    elem_type: int,
    shape: tuple[int, int, int, int],
    *,
    wrong_half_split: bool = False,
    commute_arithmetic: bool = False,
    omit_slice_steps: bool = False,
) -> list[onnx.NodeProto]:
    first_end = "ends_all" if wrong_half_split else "ends_half"
    slice_suffix = ["slice_axis"] if omit_slice_steps else ["slice_axis", "slice_step"]
    direct_inputs = ["cos", input_name] if commute_arithmetic else [input_name, "cos"]
    rotated_inputs = (
        ["sin", f"{input_name}_concat"]
        if commute_arithmetic
        else [f"{input_name}_concat", "sin"]
    )
    add_inputs = (
        [f"{input_name}_rotated", f"{input_name}_direct"]
        if commute_arithmetic
        else [f"{input_name}_direct", f"{input_name}_rotated"]
    )
    return [
        helper.make_node(
            "Slice",
            [input_name, "starts_zero", first_end, *slice_suffix],
            [f"{input_name}_first"],
        ),
        helper.make_node(
            "Slice",
            [input_name, "starts_half", "ends_all", *slice_suffix],
            [f"{input_name}_second"],
        ),
        helper.make_node("Neg", [f"{input_name}_second"], [f"{input_name}_negative"]),
        helper.make_node(
            "Concat",
            [f"{input_name}_negative", f"{input_name}_first"],
            [f"{input_name}_concat"],
            axis=-1,
        ),
        helper.make_node("Mul", direct_inputs, [f"{input_name}_direct"]),
        helper.make_node(
            "Mul",
            rotated_inputs,
            [f"{input_name}_rotated"],
        ),
        helper.make_node(
            "Add",
            add_inputs,
            [output_name],
        ),
    ]


def _rotation_value_info(
    prefix: str,
    elem_type: int,
    heads: int,
) -> list[onnx.ValueInfoProto]:
    full_shape = (1, heads, _SEQUENCE_LENGTH, _HEAD_DIMENSION)
    half_shape = (1, heads, _SEQUENCE_LENGTH, _HEAD_DIMENSION // 2)
    return [
        _value(f"{prefix}_first", elem_type, half_shape),
        _value(f"{prefix}_second", elem_type, half_shape),
        _value(f"{prefix}_negative", elem_type, half_shape),
        _value(f"{prefix}_concat", elem_type, full_shape),
        _value(f"{prefix}_direct", elem_type, full_shape),
        _value(f"{prefix}_rotated", elem_type, full_shape),
    ]


def _value(
    name: str,
    elem_type: int,
    shape: tuple[int, ...],
) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def _nodes(model: onnx.ModelProto, op_type: str) -> list[onnx.NodeProto]:
    return [node for node in model.graph.node if node.op_type == op_type]


def _only_node(model: onnx.ModelProto, op_type: str) -> onnx.NodeProto:
    nodes = _nodes(model, op_type)
    assert len(nodes) == 1
    return nodes[0]


def _attribute(node: onnx.NodeProto, name: str) -> object:
    return helper.get_attribute_value(next(attr for attr in node.attribute if attr.name == name))


def _check_model(model: onnx.ModelProto) -> None:
    register_schemas(ROTARY_POSITION_EMBEDDING_OP)
    onnx.checker.check_model(model, full_check=True)
