from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from mdc_llm_deploy.onnx.compatibility_lowering import lower_opset_compatibility_core
from mdc_llm_deploy.onnx.fusion_pass.fused_infer_attention_score import (
    fuse_fused_infer_attention_score,
)
from mdc_llm_deploy.onnx.schemas import (
    FUSED_INFER_ATTENTION_SCORE_OP,
    register_schemas,
)

_ATTRIBUTE_DEFAULTS = {
    "antiquant_mode": 0,
    "block_size": 0,
    "inner_precise": 1,
    "input_layout": b"BNSD",
    "key_antiquant_mode": 0,
    "next_tokens": 2_147_483_647,
    "out_dtype": 0,
    "pre_tokens": 2_147_483_647,
    "pse_type": 0,
    "query_quant_mode": 0,
    "softmax_lse_flag": 0,
    "sparse_mode": 0,
    "value_antiquant_mode": 0,
}


@dataclass(frozen=True)
class _AttentionCase:
    elem_type: int = TensorProto.FLOAT16
    backend: str = "eager"
    query_length: int = 3
    query_heads: int = 4
    kv_heads: int = 2
    key_length: int = 5
    mask: str = "none"
    shared_repeat: bool = False
    dropout: bool = False


@pytest.mark.parametrize("elem_type", [TensorProto.FLOAT16, TensorProto.BFLOAT16])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("query_length", [3, 1], ids=["prefill", "decode"])
@pytest.mark.parametrize(("query_heads", "kv_heads"), [(4, 4), (4, 2)], ids=["mha", "gqa"])
def test_fuses_supported_attention_matrix_with_exact_fia_abi(
    elem_type: int,
    backend: str,
    query_length: int,
    query_heads: int,
    kv_heads: int,
) -> None:
    model = _attention_model(
        _AttentionCase(
            elem_type=elem_type,
            backend=backend,
            query_length=query_length,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )
    )

    result = fuse_fused_infer_attention_score(model)

    assert result.fused_count == 1
    fused = _only_node(model, FUSED_INFER_ATTENTION_SCORE_OP)
    assert len(fused.input) == 31
    assert list(fused.input[:5]) == ["query", "key", "value", "", ""]
    assert all(not name for name in fused.input[5:])
    assert fused.output[0] == "output"
    assert len(fused.output) == 2
    attributes = {
        attribute.name: helper.get_attribute_value(attribute) for attribute in fused.attribute
    }
    assert len(attributes) == 16
    assert attributes == {
        **_ATTRIBUTE_DEFAULTS,
        "num_heads": query_heads,
        "num_key_value_heads": kv_heads,
        "scale": pytest.approx(_expected_scale(backend, elem_type)),
    }
    assert not _nodes(model, "Softmax")
    assert not _nodes(model, "Expand")
    _check_model(model)


def test_fuses_sdpa_after_expand_compatibility_lowering() -> None:
    model = _attention_model(_AttentionCase(backend="sdpa"))
    lower_opset_compatibility_core(model)
    assert len(_nodes(model, "Tile")) == 2

    result = fuse_fused_infer_attention_score(model)

    assert result.fused_count == 1
    fused = _only_node(model, FUSED_INFER_ATTENTION_SCORE_OP)
    assert list(fused.input[:3]) == ["query", "key", "value"]
    attributes = {
        attribute.name: helper.get_attribute_value(attribute) for attribute in fused.attribute
    }
    assert attributes["num_heads"] == 4
    assert attributes["num_key_value_heads"] == 2
    assert not _nodes(model, "Tile")
    _check_model(model)


@pytest.mark.parametrize(
    ("mask", "preparation_op"),
    [
        ("native", None),
        ("visibility", "Not"),
        ("additive_min", "Equal"),
        ("additive_inf", "Equal"),
    ],
)
def test_normalizes_supported_masks_to_bool(mask: str, preparation_op: str | None) -> None:
    model = _attention_model(_AttentionCase(mask=mask))

    result = fuse_fused_infer_attention_score(model)

    assert result.fused_count == 1
    fused = _only_node(model, FUSED_INFER_ATTENTION_SCORE_OP)
    assert fused.input[4]
    if preparation_op is None:
        assert fused.input[4] == "mask"
    else:
        preparation = _only_node(model, preparation_op)
        assert preparation.output[0] == fused.input[4]
    _check_model(model)


@pytest.mark.parametrize(
    "case",
    [
        _AttentionCase(elem_type=TensorProto.FLOAT),
        _AttentionCase(mask="finite_bias"),
        _AttentionCase(mask="pse"),
        _AttentionCase(mask="additive_other"),
        _AttentionCase(dropout=True),
        _AttentionCase(shared_repeat=True),
    ],
    ids=[
        "fp32",
        "alibi-finite-bias",
        "pse",
        "non-binary-additive",
        "dropout",
        "shared-repeat",
    ],
)
def test_rejects_unsafe_attention_without_modifying_graph(case: _AttentionCase) -> None:
    model = _attention_model(case)
    before = model.SerializeToString()

    result = fuse_fused_infer_attention_score(model)

    assert result.fused_count == 0
    assert model.SerializeToString() == before


def test_eager_gqa_visibility_mask_matches_pytorch_oracle() -> None:
    case = _AttentionCase(query_length=2, key_length=3, mask="visibility")
    model = _attention_model(case)
    rng = np.random.default_rng(11)
    query = rng.normal(size=(1, 4, 2, 8)).astype(np.float16)
    key = rng.normal(size=(1, 2, 3, 8)).astype(np.float16)
    value = rng.normal(size=(1, 2, 3, 8)).astype(np.float16)
    visibility = np.asarray([[[[True, True, False], [True, True, True]]]])
    evaluated = ReferenceEvaluator(model).run(
        None,
        {"query": query, "key": key, "value": value, "mask": visibility},
    )
    assert isinstance(evaluated, list)
    expected = evaluated[0]

    result = fuse_fused_infer_attention_score(model)
    fused = _only_node(model, FUSED_INFER_ATTENTION_SCORE_OP)
    attributes = {
        attribute.name: helper.get_attribute_value(attribute) for attribute in fused.attribute
    }
    expanded_key = torch.from_numpy(key).repeat_interleave(2, dim=1)
    expanded_value = torch.from_numpy(value).repeat_interleave(2, dim=1)
    scores = torch.matmul(
        torch.from_numpy(query),
        expanded_key.transpose(-1, -2),
    )
    scores = scores * attributes["scale"]
    scores = scores.masked_fill(~torch.from_numpy(visibility), -torch.inf)
    actual = torch.matmul(torch.softmax(scores, dim=-1), expanded_value).numpy()

    assert result.fused_count == 1
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_fuses_multiple_attention_subgraphs_with_unique_names() -> None:
    first = _attention_model(_AttentionCase())
    second = _attention_model(_AttentionCase())
    _prefix_graph(second, "second_")
    graph = helper.make_graph(
        [*first.graph.node, *second.graph.node],
        "multiple_attention",
        [*first.graph.input, *second.graph.input],
        [*first.graph.output, *second.graph.output],
        [*first.graph.initializer, *second.graph.initializer],
        value_info=[*first.graph.value_info, *second.graph.value_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    result = fuse_fused_infer_attention_score(model)

    assert result.fused_count == 2
    assert len(set(result.fused_node_names)) == 2
    assert len(_nodes(model, FUSED_INFER_ATTENTION_SCORE_OP)) == 2
    _check_model(model)


def _attention_model(case: _AttentionCase) -> onnx.ModelProto:
    batch, head_dim = 1, 8
    query_shape = (batch, case.query_heads, case.query_length, head_dim)
    kv_shape = (batch, case.kv_heads, case.key_length, head_dim)
    score_shape = (batch, case.query_heads, case.query_length, case.key_length)
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    value_info: list[onnx.ValueInfoProto] = []
    inputs = [
        helper.make_tensor_value_info("query", case.elem_type, query_shape),
        helper.make_tensor_value_info("key", case.elem_type, kv_shape),
        helper.make_tensor_value_info("value", case.elem_type, kv_shape),
    ]

    repeated_key = _repeat_kv(
        nodes,
        initializers,
        value_info,
        "key",
        "key_repeated",
        case,
    )
    repeated_value = _repeat_kv(
        nodes,
        initializers,
        value_info,
        "value",
        "value_repeated",
        case,
    )
    if case.shared_repeat and case.kv_heads != case.query_heads:
        nodes.append(helper.make_node("Identity", [repeated_key], ["shared_key"]))
        value_info.append(
            helper.make_tensor_value_info(
                "shared_key",
                case.elem_type,
                (batch, case.query_heads, case.key_length, head_dim),
            )
        )

    if case.backend == "eager":
        nodes.append(
            helper.make_node(
                "Transpose",
                [repeated_key],
                ["key_transposed"],
                perm=[0, 1, 3, 2],
            )
        )
        value_info.append(
            helper.make_tensor_value_info(
                "key_transposed",
                case.elem_type,
                (batch, case.query_heads, head_dim, case.key_length),
            )
        )
        nodes.append(helper.make_node("MatMul", ["query", "key_transposed"], ["raw_scores"]))
        initializers.append(_scalar_tensor("scale", case.elem_type, 1.0 / math.sqrt(head_dim)))
        nodes.append(helper.make_node("Mul", ["raw_scores", "scale"], ["scores"]))
    else:
        initializers.extend(
            [
                numpy_helper.from_array(
                    np.asarray(
                        [batch * case.query_heads, case.key_length, head_dim],
                        dtype=np.int64,
                    ),
                    "flattened_key_shape",
                ),
                numpy_helper.from_array(
                    np.asarray(
                        [batch, case.query_heads, head_dim, case.key_length],
                        dtype=np.int64,
                    ),
                    "restored_key_shape",
                ),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Reshape",
                    [repeated_key, "flattened_key_shape"],
                    ["key_flattened"],
                ),
                helper.make_node(
                    "Transpose",
                    ["key_flattened"],
                    ["key_flattened_transposed"],
                    perm=[0, 2, 1],
                ),
                helper.make_node(
                    "Reshape",
                    ["key_flattened_transposed", "restored_key_shape"],
                    ["key_transposed"],
                ),
            ]
        )
        value_info.extend(
            [
                helper.make_tensor_value_info(
                    "key_flattened",
                    case.elem_type,
                    (batch * case.query_heads, case.key_length, head_dim),
                ),
                helper.make_tensor_value_info(
                    "key_flattened_transposed",
                    case.elem_type,
                    (batch * case.query_heads, head_dim, case.key_length),
                ),
                helper.make_tensor_value_info(
                    "key_transposed",
                    case.elem_type,
                    (batch, case.query_heads, head_dim, case.key_length),
                ),
            ]
        )
        initializers.append(
            _scalar_tensor("sqrt_scale", case.elem_type, math.sqrt(1.0 / math.sqrt(head_dim)))
        )
        nodes.extend(
            [
                helper.make_node("Mul", ["query", "sqrt_scale"], ["scaled_query"]),
                helper.make_node(
                    "Mul",
                    ["key_transposed", "sqrt_scale"],
                    ["scaled_key"],
                ),
                helper.make_node("MatMul", ["scaled_query", "scaled_key"], ["scores"]),
            ]
        )
        value_info.extend(
            [
                helper.make_tensor_value_info("scaled_query", case.elem_type, query_shape),
                helper.make_tensor_value_info(
                    "scaled_key",
                    case.elem_type,
                    (batch, case.query_heads, head_dim, case.key_length),
                ),
            ]
        )

    value_info.extend(
        [
            helper.make_tensor_value_info("raw_scores", case.elem_type, score_shape),
            helper.make_tensor_value_info("scores", case.elem_type, score_shape),
        ]
    )
    softmax_input = _add_mask(
        nodes,
        initializers,
        value_info,
        inputs,
        case,
        score_shape,
    )
    nodes.append(helper.make_node("Softmax", [softmax_input], ["probabilities"], axis=-1))
    probability_name = "probabilities"
    value_info.append(helper.make_tensor_value_info("probabilities", case.elem_type, score_shape))
    if case.backend == "sdpa":
        initializers.append(_scalar_tensor("zero", case.elem_type, 0.0))
        nodes.extend(
            [
                helper.make_node("IsNaN", ["probabilities"], ["nan_probabilities"]),
                helper.make_node(
                    "Where",
                    ["nan_probabilities", "zero", "probabilities"],
                    ["normalized_probabilities"],
                ),
            ]
        )
        value_info.extend(
            [
                helper.make_tensor_value_info(
                    "nan_probabilities",
                    TensorProto.BOOL,
                    score_shape,
                ),
                helper.make_tensor_value_info(
                    "normalized_probabilities",
                    case.elem_type,
                    score_shape,
                ),
            ]
        )
        probability_name = "normalized_probabilities"
    if case.dropout:
        initializers.append(_scalar_tensor("dropout_ratio", TensorProto.FLOAT, 0.1))
        nodes.append(
            helper.make_node(
                "Dropout",
                [probability_name, "dropout_ratio"],
                ["dropped_probabilities"],
            )
        )
        probability_name = "dropped_probabilities"
        value_info.append(
            helper.make_tensor_value_info(
                probability_name,
                case.elem_type,
                score_shape,
            )
        )
    nodes.append(helper.make_node("MatMul", [probability_name, repeated_value], ["output"]))
    if case.shared_repeat and case.kv_heads != case.query_heads:
        nodes.append(helper.make_node("Identity", ["shared_key"], ["shared_output"]))

    outputs = [helper.make_tensor_value_info("output", case.elem_type, query_shape)]
    if case.shared_repeat and case.kv_heads != case.query_heads:
        outputs.append(
            helper.make_tensor_value_info(
                "shared_output",
                case.elem_type,
                (batch, case.query_heads, case.key_length, head_dim),
            )
        )
    graph = helper.make_graph(
        nodes,
        "attention",
        inputs,
        outputs,
        initializers,
        value_info=value_info,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _repeat_kv(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    value_info: list[onnx.ValueInfoProto],
    source: str,
    output: str,
    case: _AttentionCase,
) -> str:
    if case.kv_heads == case.query_heads:
        return source
    repeats = case.query_heads // case.kv_heads
    unsqueezed = f"{source}_unsqueezed"
    expanded = f"{source}_expanded"
    axes_name = f"{source}_axes"
    expanded_shape_name = f"{source}_expanded_shape"
    repeated_shape_name = f"{source}_repeated_shape"
    initializers.extend(
        [
            numpy_helper.from_array(np.asarray([2], dtype=np.int64), axes_name),
            numpy_helper.from_array(
                np.asarray(
                    [1, case.kv_heads, repeats, case.key_length, 8],
                    dtype=np.int64,
                ),
                expanded_shape_name,
            ),
            numpy_helper.from_array(
                np.asarray([1, case.query_heads, case.key_length, 8], dtype=np.int64),
                repeated_shape_name,
            ),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Unsqueeze", [source, axes_name], [unsqueezed]),
            helper.make_node("Expand", [unsqueezed, expanded_shape_name], [expanded]),
            helper.make_node("Reshape", [expanded, repeated_shape_name], [output]),
        ]
    )
    value_info.extend(
        [
            helper.make_tensor_value_info(
                unsqueezed,
                case.elem_type,
                (1, case.kv_heads, 1, case.key_length, 8),
            ),
            helper.make_tensor_value_info(
                expanded,
                case.elem_type,
                (1, case.kv_heads, repeats, case.key_length, 8),
            ),
            helper.make_tensor_value_info(
                output,
                case.elem_type,
                (1, case.query_heads, case.key_length, 8),
            ),
        ]
    )
    return output


def _add_mask(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    value_info: list[onnx.ValueInfoProto],
    inputs: list[onnx.ValueInfoProto],
    case: _AttentionCase,
    score_shape: tuple[int, int, int, int],
) -> str:
    if case.mask == "none":
        return "scores"
    mask_shape = (1, 1, case.query_length, case.key_length)
    negative = _minimum(case.elem_type)
    if case.mask == "native":
        inputs.append(helper.make_tensor_value_info("mask", TensorProto.BOOL, mask_shape))
        initializers.append(_scalar_tensor("negative", case.elem_type, negative))
        nodes.append(helper.make_node("Where", ["mask", "negative", "scores"], ["masked_scores"]))
    elif case.mask == "visibility":
        inputs.append(helper.make_tensor_value_info("mask", TensorProto.BOOL, mask_shape))
        initializers.extend(
            [
                _scalar_tensor("zero", case.elem_type, 0.0),
                _scalar_tensor("negative", case.elem_type, negative),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Where", ["mask", "zero", "negative"], ["additive_mask"]),
                helper.make_node("Add", ["scores", "additive_mask"], ["masked_scores"]),
            ]
        )
        value_info.append(
            helper.make_tensor_value_info("additive_mask", case.elem_type, mask_shape)
        )
    elif case.mask == "pse":
        inputs.append(helper.make_tensor_value_info("mask", case.elem_type, mask_shape))
        nodes.append(helper.make_node("Add", ["scores", "mask"], ["masked_scores"]))
    else:
        if case.mask == "additive_inf":
            values = [0.0, -math.inf] * (case.query_length * case.key_length // 2)
        elif case.mask == "additive_min":
            values = [0.0, negative] * (case.query_length * case.key_length // 2)
        elif case.mask == "finite_bias":
            values = [0.0, -1.0] * (case.query_length * case.key_length // 2)
        else:
            values = [0.0, negative, 1.0] * (case.query_length * case.key_length // 3)
        size = case.query_length * case.key_length
        values = (values + [0.0] * size)[:size]
        initializers.append(
            helper.make_tensor(
                "mask",
                case.elem_type,
                mask_shape,
                values,
            )
        )
        nodes.append(helper.make_node("Add", ["scores", "mask"], ["masked_scores"]))
    value_info.append(helper.make_tensor_value_info("masked_scores", case.elem_type, score_shape))
    return "masked_scores"


def _scalar_tensor(name: str, elem_type: int, value: float) -> onnx.TensorProto:
    return helper.make_tensor(name, elem_type, [], [value])


def _minimum(elem_type: int) -> float:
    if elem_type == TensorProto.FLOAT16:
        return float(np.finfo(np.float16).min)
    if elem_type == TensorProto.BFLOAT16:
        return -3.3895313892515355e38
    return float(np.finfo(np.float32).min)


def _expected_scale(backend: str, elem_type: int) -> float:
    scalar_type = np.float16 if elem_type == TensorProto.FLOAT16 else np.float32
    if backend == "eager":
        value = float(scalar_type(1.0 / math.sqrt(8)))
        return value if elem_type == TensorProto.FLOAT16 else 0.353515625
    split = float(scalar_type(math.sqrt(1.0 / math.sqrt(8))))
    if elem_type == TensorProto.BFLOAT16:
        split = 0.59375
    return split * split


def _prefix_graph(model: onnx.ModelProto, prefix: str) -> None:
    names = {name for node in model.graph.node for name in (*node.input, *node.output) if name}
    names.update(
        value.name for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
    )
    names.update(tensor.name for tensor in model.graph.initializer)
    mapping = {name: f"{prefix}{name}" for name in names}
    for node in model.graph.node:
        node.input[:] = [mapping.get(name, name) for name in node.input]
        node.output[:] = [mapping.get(name, name) for name in node.output]
        if node.name:
            node.name = f"{prefix}{node.name}"
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        value.name = mapping[value.name]
    for tensor in model.graph.initializer:
        tensor.name = mapping[tensor.name]


def _nodes(model: onnx.ModelProto, op_type: str) -> list[onnx.NodeProto]:
    return [node for node in model.graph.node if node.op_type == op_type]


def _only_node(model: onnx.ModelProto, op_type: str) -> onnx.NodeProto:
    nodes = _nodes(model, op_type)
    assert len(nodes) == 1
    return nodes[0]


def _check_model(model: onnx.ModelProto) -> None:
    register_schemas(FUSED_INFER_ATTENTION_SCORE_OP)
    onnx.checker.check_model(model, full_check=True)
