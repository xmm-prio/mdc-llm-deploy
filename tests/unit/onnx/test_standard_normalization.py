from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn
from torch.fx import Graph, GraphModule

import mdc_llm_deploy.onnx.export.normalization as normalization_module
from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.graph.lifecycle import (
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    TensorAbi,
)
from mdc_llm_deploy.onnx.export.normalization import (
    _fold_linear_weight_transposes,
    _fold_rms_norm_initializers,
    _InitializerAliasFoldPlan,
    normalize_standard_onnx,
)


def _metadata() -> GraphMetadata:
    return GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="dense",
        input_abi=(TensorAbi("x", "float32", (1, 2)),),
        output_abi=(TensorAbi("output", "float32", (1, 2)),),
        sequence_length=2,
        properties={"input_devices": {"x": "cpu"}},
    )


def _linear_graph(count: int = 2) -> GraphModule:
    root = nn.Module()
    graph = Graph()
    value = graph.placeholder("x")
    for index in range(count):
        name = f"weight_{index}"
        root.register_parameter(
            name,
            nn.Parameter(torch.full((2, 2), index + 1.0)),
        )
        weight = graph.get_attr(name)
        value = graph.call_function(
            torch.ops.aten.linear.default,
            (value, weight, None),
        )
    graph.output(value)
    return GraphModule(root, graph)


def _linear_model(
    *,
    weight_count: int = 2,
    include_embedding: bool = False,
    invalid_weight: bool = False,
) -> onnx.ModelProto:
    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []
    source = "x"
    for index in range(weight_count):
        name = f"onnx_weight_{index}"
        array = (
            np.ones(2, dtype=np.int64)
            if invalid_weight and index == weight_count - 1
            else np.full((2, 2), index + 1, dtype=np.float32)
        )
        initializers.append(numpy_helper.from_array(array, name=name))
        target = "output" if index == weight_count - 1 else f"hidden_{index}"
        nodes.append(helper.make_node("MatMul", [source, name], [target]))
        source = target
    if include_embedding:
        initializers.append(
            numpy_helper.from_array(
                np.ones((2, 2), dtype=np.float32),
                name="embed_tokens",
            )
        )
        nodes.insert(
            0,
            helper.make_node("MatMul", ["x", "embed_tokens"], ["embedded"]),
        )
    graph = helper.make_graph(
        nodes,
        "standard",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 2))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 2))],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_normalize_standard_onnx_folds_transpose_and_validates_standard() -> None:
    model = _linear_model()
    linear_nodes = list(model.graph.node)
    transposes: list[onnx.NodeProto] = []
    for index, node in enumerate(linear_nodes):
        old_name = node.input[1]
        transposed_name = f"transposed_{index}"
        transposes.append(
            helper.make_node("Transpose", [old_name], [transposed_name]),
        )
        node.input[1] = transposed_name
    del model.graph.node[:]
    for transpose, linear in zip(transposes, linear_nodes, strict=True):
        model.graph.node.extend((transpose, linear))

    result = normalize_standard_onnx(model, _linear_graph(), _metadata())

    assert all(node.op_type != "Transpose" for node in result.graph.node)
    assert [item.name for item in result.graph.initializer] == [
        "graph.weight_0",
        "graph.weight_1",
    ]
    onnx.checker.check_model(result, full_check=True)


def test_fold_linear_transpose_ignores_invalid_initializer() -> None:
    model = _linear_model(weight_count=1, invalid_weight=True)
    model.graph.node.insert(
        0,
        helper.make_node("Transpose", ["onnx_weight_0"], ["transposed"]),
    )
    model.graph.node[1].input[1] = "transposed"
    _fold_linear_weight_transposes(model)
    assert model.graph.node[0].op_type == "Transpose"


def _rms_metadata(*fqns: str) -> GraphMetadata:
    value = _metadata()
    return GraphMetadata(
        schema_version=value.schema_version,
        stage=value.stage,
        model_kind=value.model_kind,
        input_abi=value.input_abi,
        output_abi=value.output_abi,
        boundaries=tuple(
            FusionBoundary("rms_norm", fqn, (f"node_{index}",)) for index, fqn in enumerate(fqns)
        ),
        sequence_length=value.sequence_length,
        properties=value.properties,
    )


def _reference_fold_initializer_alias(
    model: onnx.ModelProto,
    canonical_name: str,
) -> str | None:
    initializers = {item.name: item for item in model.graph.initializer}
    if canonical_name in initializers:
        return None
    producers = {output: node for node in model.graph.node for output in node.output}
    aliases: set[str] = set()
    source = canonical_name
    identities: list[onnx.NodeProto] = []
    while source not in initializers:
        producer = producers.get(source)
        if (
            producer is None
            or producer.op_type != "Identity"
            or len(producer.input) != 1
            or len(producer.output) != 1
        ):
            raise OnnxExportError(f"Parameter {canonical_name!r} is not backed by an initializer")
        identities.append(producer)
        aliases.add(source)
        source = producer.input[0]
    initializer = onnx.TensorProto()
    initializer.CopyFrom(initializers[source])
    initializer.name = canonical_name
    model.graph.initializer.append(initializer)
    identity_ids = {id(node) for node in identities}
    for node in model.graph.node:
        if id(node) in identity_ids:
            continue
        for index, input_name in enumerate(node.input):
            if input_name in aliases:
                node.input[index] = canonical_name
    for identity in identities:
        model.graph.node.remove(identity)
    retained_values = [
        item
        for item in model.graph.value_info
        if item.name not in aliases or item.name == canonical_name
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_values)
    return source


def _reference_fold_rms_norm_initializers(
    model: onnx.ModelProto,
    metadata: GraphMetadata,
) -> None:
    alias_sources: set[str] = set()
    for boundary in metadata.boundaries:
        if boundary.kind != "rms_norm":
            continue
        source = _reference_fold_initializer_alias(
            model,
            f"graph.{boundary.fqn}.weight",
        )
        if source is not None:
            alias_sources.add(source)
    used_inputs = {
        input_name for node in model.graph.node for input_name in node.input if input_name
    }
    retained = [
        item
        for item in model.graph.initializer
        if item.name not in alias_sources or item.name in used_inputs
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)


def _independent_alias_model(*, shared_source: bool = False) -> onnx.ModelProto:
    first_source = "shared" if shared_source else "source_first"
    second_source = "shared" if shared_source else "source_second"
    initializer_names = ("shared",) if shared_source else ("source_first", "source_second")
    nodes = [
        helper.make_node("Identity", [first_source], ["first_inner"], name="first_0"),
        helper.make_node(
            "Identity",
            ["first_inner"],
            ["graph.first.weight"],
            name="first_1",
        ),
        helper.make_node("Add", ["x", "first_inner"], ["unrelated"]),
        helper.make_node(
            "Identity",
            [second_source],
            ["second_inner"],
            name="second_0",
        ),
        helper.make_node(
            "Identity",
            ["second_inner"],
            ["graph.second.weight"],
            name="second_1",
        ),
        helper.make_node(
            "Add",
            ["graph.first.weight", "graph.first.weight"],
            ["first_used"],
        ),
        helper.make_node(
            "Identity",
            ["second_inner"],
            ["chain_external"],
            name="external_identity",
        ),
        helper.make_node(
            "Add",
            ["graph.second.weight", "chain_external"],
            ["output"],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "independent_aliases",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (2,))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2,))],
        initializer=[
            numpy_helper.from_array(np.ones(2, dtype=np.float32), name=name)
            for name in initializer_names
        ],
        value_info=[
            helper.make_tensor_value_info(name, TensorProto.FLOAT, (2,))
            for name in (
                "first_inner",
                "first_inner",
                "graph.first.weight",
                "unrelated",
                "second_inner",
                "graph.second.weight",
            )
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


@pytest.mark.parametrize("shared_source", [False, True])
def test_rms_initializer_alias_fast_path_matches_frozen_reference(
    shared_source: bool,
) -> None:
    model = _independent_alias_model(shared_source=shared_source)
    metadata = _rms_metadata("first", "second")
    assert (
        _InitializerAliasFoldPlan.try_build(
            model,
            ("graph.first.weight", "graph.second.weight"),
        )
        is not None
    )
    reference = copy.deepcopy(model)
    actual = copy.deepcopy(model)
    _reference_fold_rms_norm_initializers(reference, metadata)
    _fold_rms_norm_initializers(actual, metadata)
    assert actual.SerializeToString(deterministic=True) == (
        reference.SerializeToString(deterministic=True)
    )


@pytest.mark.parametrize(
    "invalid_node",
    [
        pytest.param(None, id="missing-producer"),
        pytest.param(
            helper.make_node("Add", ["x", "x"], ["graph.invalid.weight"]),
            id="non-identity",
        ),
        pytest.param(
            helper.make_node(
                "Identity",
                ["source_invalid", "x"],
                ["graph.invalid.weight"],
            ),
            id="invalid-arity",
        ),
    ],
)
def test_rms_initializer_failure_preserves_reference_partial_mutation(
    invalid_node: onnx.NodeProto | None,
) -> None:
    model = _independent_alias_model()
    if invalid_node is not None:
        model.graph.node.append(invalid_node)
        if invalid_node.op_type == "Identity":
            model.graph.initializer.append(
                numpy_helper.from_array(
                    np.ones(2, dtype=np.float32),
                    name="source_invalid",
                )
            )
    metadata = _rms_metadata("first", "invalid")
    reference = copy.deepcopy(model)
    actual = copy.deepcopy(model)
    with pytest.raises(OnnxExportError) as reference_error:
        _reference_fold_rms_norm_initializers(reference, metadata)
    with pytest.raises(OnnxExportError) as actual_error:
        _fold_rms_norm_initializers(actual, metadata)
    assert str(actual_error.value) == str(reference_error.value)
    assert actual.SerializeToString(deterministic=True) == (
        reference.SerializeToString(deterministic=True)
    )


def _custom_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node("MoeExpert", ["x"], ["custom_hidden"]),
            helper.make_node("Identity", ["custom_hidden"], ["output"]),
        ],
        "custom",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 2))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 2))],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_normalize_custom_branch_seeds_value_info_and_uses_structure_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def validate_structure(model: onnx.ModelProto) -> None:
        assert any(item.name == "custom_hidden" for item in model.graph.value_info)
        calls.append("structure")

    def reject_standard(*args: object, **kwargs: object) -> None:
        raise AssertionError("standard checker must not validate custom graph")

    monkeypatch.setattr(
        normalization_module,
        "validate_mdc_model_structure",
        validate_structure,
    )
    monkeypatch.setattr(onnx.checker, "check_model", reject_standard)

    normalize_standard_onnx(_custom_model(), _linear_graph(0), _metadata())

    assert calls == ["structure"]


def test_normalize_standard_branch_uses_standard_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def validate_standard(model: onnx.ModelProto) -> None:
        del model
        calls.append("standard")

    def reject_mdc_structure(model: onnx.ModelProto) -> None:
        del model
        raise AssertionError("MDC structure validator must not validate standard graph")

    monkeypatch.setattr(
        normalization_module,
        "validate_standard_model",
        validate_standard,
    )
    monkeypatch.setattr(
        normalization_module,
        "validate_mdc_model_structure",
        reject_mdc_structure,
    )

    normalize_standard_onnx(_linear_model(), _linear_graph(), _metadata())

    assert calls == ["standard"]
