from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn
from torch.fx import Graph, GraphModule

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.graph.lifecycle import (
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    TensorAbi,
)
from mdc_llm_deploy.onnx.export.standard import (
    _example_arguments,
    _fold_rms_norm_initializers,
    _InitializerAliasFoldPlan,
    _rename_initializer_references,
    export_standard_onnx,
)
from mdc_llm_deploy.operators.contracts.onnx import MDC_ONNX_OPSET


def _metadata(*, properties: dict[str, Any] | None = None) -> GraphMetadata:
    return GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="dense",
        input_abi=(TensorAbi("x", "float32", (1, 2)),),
        output_abi=(TensorAbi("output", "float32", (1, 2)),),
        sequence_length=2,
        properties=(
            {"input_devices": {"x": "cpu"}}
            if properties is None
            else properties
        ),
    )


def _linear_graph() -> GraphModule:
    root = nn.Module()
    root.register_parameter("first_weight", nn.Parameter(torch.eye(2)))
    root.register_parameter("second_weight", nn.Parameter(torch.eye(2)))
    graph = Graph()
    value = graph.placeholder("x")
    first_weight = graph.get_attr("first_weight")
    value = graph.call_function(
        torch.ops.aten.linear.default,
        (value, first_weight, None),
    )
    second_weight = graph.get_attr("second_weight")
    value = graph.call_function(
        torch.ops.aten.linear.default,
        (value, second_weight, None),
    )
    graph.output(value)
    return GraphModule(root, graph)


def _rms_graph() -> GraphModule:
    root = nn.Module()
    root.add_module("norm", nn.Module())
    root.norm.register_parameter(  # type: ignore[attr-defined]
        "weight",
        nn.Parameter(torch.ones(2)),
    )
    graph = Graph()
    value = graph.placeholder("x")
    weight = graph.get_attr("norm.weight")
    value = graph.call_function(torch.ops.aten.mul.Tensor, (value, weight))
    graph.output(value)
    return GraphModule(root, graph)


def _standard_model() -> onnx.ModelProto:
    first = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="onnx_first")
    second = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="onnx_second")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "onnx_first"], ["hidden"]),
            helper.make_node("Identity", ["onnx_first"], ["weight_copy"]),
            helper.make_node("MatMul", ["hidden", "onnx_second"], ["output"]),
            helper.make_node("Identity", ["onnx_second"], ["second_weight_copy"]),
        ],
        "standard",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 2))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 2))],
        initializer=[first, second],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _rename_reference_model() -> onnx.ModelProto:
    initializers = [
        numpy_helper.from_array(np.ones(2, dtype=np.float32), name=name)
        for name in ("a", "b", "c")
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["a"], ["declared"]),
            helper.make_node("Identity", ["a"], ["a_copy"]),
            helper.make_node("Identity", ["b"], ["b_copy"]),
            helper.make_node("Identity", ["c"], ["c_copy"]),
        ],
        "rename_references",
        [],
        [helper.make_tensor_value_info("declared", TensorProto.FLOAT, (2,))],
        initializer=initializers,
        value_info=[
            helper.make_tensor_value_info("a_copy", TensorProto.FLOAT, (2,))
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _ordered_renames(
    model: onnx.ModelProto,
    specification: list[tuple[int, str, str]],
) -> list[tuple[onnx.TensorProto, str, str]]:
    return [
        (model.graph.initializer[index], old_name, new_name)
        for index, old_name, new_name in specification
    ]


def _reference_rename_initializer_references(
    model: onnx.ModelProto,
    renames: list[tuple[onnx.TensorProto, str, str]],
) -> None:
    for initializer, old_name, new_name in renames:
        initializer.name = new_name
        for node in model.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name == old_name:
                    node.input[index] = new_name


def _unfolded_standard_model() -> onnx.ModelProto:
    first = numpy_helper.from_array(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        name="raw_first",
    )
    second = numpy_helper.from_array(
        np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        name="raw_second",
    )
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["raw_first"], ["first_weight"]),
            helper.make_node("MatMul", ["x", "first_weight"], ["hidden"]),
            helper.make_node("Transpose", ["raw_second"], ["second_weight"]),
            helper.make_node("MatMul", ["hidden", "second_weight"], ["output"]),
        ],
        "unfolded",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 2))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 2))],
        initializer=[first, second],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _rms_standard_model() -> onnx.ModelProto:
    weight = numpy_helper.from_array(
        np.ones(2, dtype=np.float32),
        name="onnx_norm",
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["onnx_norm"],
                ["graph.norm.weight"],
            ),
            helper.make_node(
                "Mul",
                ["x", "graph.norm.weight"],
                ["output"],
            ),
        ],
        "rms",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 2))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 2))],
        initializer=[weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _reference_fold_initializer_alias(
    model: onnx.ModelProto,
    canonical_name: str,
) -> str | None:
    initializers = {item.name: item for item in model.graph.initializer}
    if canonical_name in initializers:
        return None
    producers = {
        output: node for node in model.graph.node for output in node.output
    }
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
            raise OnnxExportError(
                f"Parameter {canonical_name!r} is not backed by an initializer"
            )
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
        if boundary.kind == "rms_norm":
            source = _reference_fold_initializer_alias(
                model,
                f"graph.{boundary.fqn}.weight",
            )
            if source is not None:
                alias_sources.add(source)
    used_inputs = {
        input_name
        for node in model.graph.node
        for input_name in node.input
        if input_name
    }
    retained_initializers = [
        item
        for item in model.graph.initializer
        if item.name not in alias_sources or item.name in used_inputs
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)


def _rms_metadata(*fqns: str) -> GraphMetadata:
    value = _metadata()
    return GraphMetadata(
        schema_version=value.schema_version,
        stage=value.stage,
        model_kind=value.model_kind,
        input_abi=value.input_abi,
        output_abi=value.output_abi,
        boundaries=tuple(
            FusionBoundary("rms_norm", fqn, (f"node_{index}",))
            for index, fqn in enumerate(fqns)
        ),
        sequence_length=value.sequence_length,
        properties=value.properties,
    )


def _assert_rms_fold_matches_reference(
    model: onnx.ModelProto,
    metadata: GraphMetadata,
) -> onnx.ModelProto:
    reference = copy.deepcopy(model)
    actual = copy.deepcopy(model)
    _reference_fold_rms_norm_initializers(reference, metadata)
    _fold_rms_norm_initializers(actual, metadata)
    assert actual.SerializeToString(deterministic=True) == (
        reference.SerializeToString(deterministic=True)
    )
    return actual


def _independent_alias_model(*, shared_source: bool = False) -> onnx.ModelProto:
    first_source = "shared" if shared_source else "source_first"
    second_source = "shared" if shared_source else "source_second"
    initializer_names = (
        ("shared",) if shared_source else ("source_first", "source_second")
    )
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


def test_example_arguments_follow_per_input_device_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[torch.device] = []
    original_zeros = torch.zeros

    def record_zeros(
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        captured.append(device)
        return original_zeros(shape, dtype=dtype)

    monkeypatch.setattr(torch, "zeros", record_zeros)
    value = GraphMetadata(
        schema_version=1,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="dense",
        input_abi=(
            TensorAbi("tokens", "int64", (1, 2)),
            TensorAbi("mask", "bool", (1, 2)),
        ),
        output_abi=(TensorAbi("output", "float32", (1, 2)),),
        properties={
            "input_devices": {
                "tokens": "cuda:1",
                "mask": "cpu",
            }
        },
    )

    arguments = _example_arguments(value)

    assert captured == [torch.device("cuda:1"), torch.device("cpu")]
    assert [argument.dtype for argument in arguments] == [torch.int64, torch.bool]


@pytest.mark.parametrize(
    ("properties", "message"),
    [
        ({}, "contract is missing"),
        ({"input_devices": {"other": "cpu"}}, "missing=\\['x'\\]"),
        ({"input_devices": {"x": 0}}, "for 'x' must be a string"),
        ({"input_devices": {"x": "not-a-device"}}, "for 'x' is invalid"),
    ],
)
def test_example_arguments_reject_invalid_input_device_contract(
    properties: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(OnnxExportError, match=message):
        _example_arguments(_metadata(properties=properties))


def test_standard_export_wraps_external_failure_and_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = RuntimeError("boom")

    def fail_export(*args: Any, **kwargs: Any) -> None:
        raise original

    monkeypatch.setattr(torch.onnx, "export", fail_export)

    with pytest.raises(
        OnnxExportError,
        match="Standard ONNX validation failed: boom",
    ) as captured:
        export_standard_onnx(_linear_graph(), _metadata(), tmp_path)

    assert captured.value.__cause__ is original
    assert list(tmp_path.iterdir()) == []


def test_standard_export_restores_initializer_fqns_and_all_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_options: dict[str, Any] = {}

    def save_standard(
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: Path,
        **kwargs: Any,
    ) -> None:
        del args
        export_options["model_training"] = model.training
        export_options.update(kwargs)
        onnx.save_model(_standard_model(), output)

    monkeypatch.setattr(torch.onnx, "export", save_standard)

    model = export_standard_onnx(_linear_graph(), _metadata(), tmp_path)

    assert [item.name for item in model.graph.initializer] == [
        "graph.first_weight",
        "graph.second_weight",
    ]
    assert [list(node.input) for node in model.graph.node] == [
        ["x", "graph.first_weight"],
        ["graph.first_weight"],
        ["hidden", "graph.second_weight"],
        ["graph.second_weight"],
    ]
    assert export_options["opset_version"] == MDC_ONNX_OPSET
    assert export_options["export_params"] is True
    assert export_options["model_training"] is False
    assert export_options["do_constant_folding"] is False
    assert export_options["training"] is torch.onnx.TrainingMode.PRESERVE
    assert export_options["dynamo"] is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "specification",
    [
        pytest.param([(0, "a", "b"), (1, "b", "a")], id="swap"),
        pytest.param([(0, "a", "b"), (1, "b", "next")], id="chain"),
        pytest.param([(0, "a", "first"), (0, "a", "second")], id="repeated-old"),
        pytest.param([(0, "a", "same"), (1, "b", "same")], id="repeated-target"),
        pytest.param([(0, "a", "c")], id="occupied-initializer"),
        pytest.param([(0, "a", "declared")], id="occupied-declaration"),
    ],
)
def test_initializer_reference_rename_conflicts_match_ordered_reference(
    specification: list[tuple[int, str, str]],
) -> None:
    reference = _rename_reference_model()
    actual = copy.deepcopy(reference)

    _reference_rename_initializer_references(
        reference,
        _ordered_renames(reference, specification),
    )
    _rename_initializer_references(
        actual,
        _ordered_renames(actual, specification),
    )

    assert actual.SerializeToString(deterministic=True) == (
        reference.SerializeToString(deterministic=True)
    )


def test_initializer_reference_rename_fast_path_matches_ordered_reference() -> None:
    specification = [(0, "a", "first"), (1, "b", "second")]
    reference = _rename_reference_model()
    actual = copy.deepcopy(reference)

    _reference_rename_initializer_references(
        reference,
        _ordered_renames(reference, specification),
    )
    _rename_initializer_references(
        actual,
        _ordered_renames(actual, specification),
    )

    assert actual.SerializeToString(deterministic=True) == (
        reference.SerializeToString(deterministic=True)
    )
    onnx.checker.check_model(actual, full_check=True)


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

    actual = _assert_rms_fold_matches_reference(model, metadata)

    assert [node.name for node in actual.graph.node] == [
        "",
        "",
        "external_identity",
        "",
    ]
    assert [item.name for item in actual.graph.value_info] == [
        "graph.first.weight",
        "unrelated",
        "graph.second.weight",
    ]
    onnx.checker.check_model(actual, full_check=True)


def test_rms_initializer_alias_duplicate_initializer_uses_last_wins_fallback() -> None:
    model = _independent_alias_model()
    model.graph.initializer.insert(
        0,
        numpy_helper.from_array(
            np.zeros(2, dtype=np.float32),
            name="source_first",
        ),
    )
    assert (
        _InitializerAliasFoldPlan.try_build(
            model,
            ("graph.first.weight", "graph.second.weight"),
        )
        is None
    )
    actual = _assert_rms_fold_matches_reference(
        model,
        _rms_metadata("first", "second"),
    )
    first = next(
        item
        for item in actual.graph.initializer
        if item.name == "graph.first.weight"
    )
    np.testing.assert_array_equal(
        numpy_helper.to_array(first),
        np.ones(2, dtype=np.float32),
    )


def test_rms_initializer_alias_duplicate_producer_preserves_earlier_node() -> None:
    model = _independent_alias_model()
    model.graph.node.insert(
        0,
        helper.make_node(
            "Identity",
            ["source_first"],
            ["graph.first.weight"],
            name="earlier_duplicate",
        ),
    )
    assert (
        _InitializerAliasFoldPlan.try_build(
            model,
            ("graph.first.weight", "graph.second.weight"),
        )
        is None
    )
    actual = _assert_rms_fold_matches_reference(
        model,
        _rms_metadata("first", "second"),
    )
    assert "earlier_duplicate" in [node.name for node in actual.graph.node]


@pytest.mark.parametrize(
    ("fqns", "fails"),
    [
        (("first", "second"), False),
        (("second", "first"), True),
    ],
)
def test_rms_initializer_shared_chain_falls_back_in_boundary_order(
    fqns: tuple[str, str],
    fails: bool,
) -> None:
    source = numpy_helper.from_array(
        np.ones(2, dtype=np.float32),
        name="source",
    )
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["source"], ["shared"]),
            helper.make_node(
                "Identity",
                ["shared"],
                ["graph.first.weight"],
            ),
            helper.make_node(
                "Identity",
                ["graph.first.weight"],
                ["graph.second.weight"],
            ),
            helper.make_node("Add", ["graph.second.weight", "x"], ["output"]),
        ],
        "shared_chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (2,))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2,))],
        initializer=[source],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    assert (
        _InitializerAliasFoldPlan.try_build(
            model,
            tuple(f"graph.{fqn}.weight" for fqn in fqns),
        )
        is None
    )
    metadata = _rms_metadata(*fqns)
    if not fails:
        _assert_rms_fold_matches_reference(model, metadata)
        return

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


def test_rms_initializer_existing_canonical_no_op_matches_reference() -> None:
    model = _independent_alias_model()
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.full(2, 3.0, dtype=np.float32),
            name="graph.first.weight",
        )
    )
    _assert_rms_fold_matches_reference(
        model,
        _rms_metadata("first", "second"),
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

    assert type(actual_error.value) is type(reference_error.value)
    assert str(actual_error.value) == str(reference_error.value)
    assert actual_error.value.__cause__ is reference_error.value.__cause__ is None
    assert actual.SerializeToString(deterministic=True) == (
        reference.SerializeToString(deterministic=True)
    )


def test_standard_export_folds_linear_transposes_without_jit_constant_folding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def save_unfolded(
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: Path,
        **kwargs: Any,
    ) -> None:
        del model, args, kwargs
        onnx.save_model(_unfolded_standard_model(), output)

    monkeypatch.setattr(torch.onnx, "export", save_unfolded)

    model = export_standard_onnx(_linear_graph(), _metadata(), tmp_path)

    initializers = {item.name: item for item in model.graph.initializer}
    assert set(initializers) == {
        "graph.first_weight",
        "graph.second_weight",
    }
    np.testing.assert_array_equal(
        numpy_helper.to_array(initializers["graph.first_weight"]),
        np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
    )
    assert all(node.op_type != "Transpose" for node in model.graph.node)
    assert [node.input[1] for node in model.graph.node if node.op_type == "MatMul"] == [
        "graph.first_weight",
        "graph.second_weight",
    ]
    assert list(tmp_path.iterdir()) == []


def test_standard_export_folds_rms_norm_weight_into_initializer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def save_standard(
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: Path,
        **kwargs: Any,
    ) -> None:
        del model, args, kwargs
        onnx.save_model(_rms_standard_model(), output)

    monkeypatch.setattr(torch.onnx, "export", save_standard)
    value = _metadata()
    value = GraphMetadata(
        schema_version=value.schema_version,
        stage=value.stage,
        model_kind=value.model_kind,
        input_abi=value.input_abi,
        output_abi=value.output_abi,
        boundaries=(
            FusionBoundary("rms_norm", "norm", ("mul",)),
        ),
        sequence_length=value.sequence_length,
        properties=value.properties,
    )

    model = export_standard_onnx(_rms_graph(), value, tmp_path)

    assert [item.name for item in model.graph.initializer] == [
        "graph.norm.weight"
    ]
    assert all(node.op_type != "Identity" for node in model.graph.node)
    assert next(node for node in model.graph.node if node.op_type == "Mul").input[1] == (
        "graph.norm.weight"
    )
    assert list(tmp_path.iterdir()) == []


def test_standard_export_materializes_external_data_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def save_external_standard(
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: Path,
        **kwargs: Any,
    ) -> None:
        del model, args, kwargs
        onnx.save_model(
            _standard_model(),
            output,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.data",
            size_threshold=0,
        )

    monkeypatch.setattr(torch.onnx, "export", save_external_standard)

    model = export_standard_onnx(_linear_graph(), _metadata(), tmp_path)

    assert all(
        item.data_location == TensorProto.DEFAULT
        and bool(item.raw_data)
        and not item.external_data
        for item in model.graph.initializer
    )
    assert list(tmp_path.iterdir()) == []
