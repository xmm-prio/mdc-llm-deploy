"""Release-specific semantic acceptance contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

from mdc_llm_deploy.capabilities import (
    Algorithm,
    Artifact,
    Capability,
    MaskMode,
    ModelKind,
    Phase,
    Target,
)
from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.graph.metadata import GraphStage
from mdc_llm_deploy.onnx.validation.metadata import ValidatedMetadata
from mdc_llm_deploy.onnx.validation.model import ValidatedMdcArtifact
from mdc_llm_deploy.onnx.validation.topology import (
    QuantizationTopologyEvidence,
    quantized_target_families,
    validate_custom_node_reachability,
    validate_graph_topology,
)
from mdc_llm_deploy.operators.contracts.attention import (
    ATTENTION_INPUT_COUNT,
    AttentionInput,
)
from tools.release import validation as release_validation


def _capability(
    *,
    model: ModelKind = ModelKind.DENSE,
    algorithm: Algorithm = Algorithm.FP16,
    target: Target | None = None,
    phase: Phase = Phase.PREFILL,
) -> Capability:
    return Capability(
        model=model,
        algorithm=algorithm,
        target=target,
        phase=phase,
        mask_mode=MaskMode.MASKED,
        artifacts=frozenset({Artifact.ONNX}),
    )


def _metadata(
    capability: Capability,
    *,
    save_kv_cache: bool | None = True,
) -> ValidatedMetadata:
    stage = {
        (Algorithm.FP16, Phase.PREFILL): GraphStage.FLOAT_PREFILL,
        (Algorithm.FP16, Phase.DECODE): GraphStage.FLOAT_DECODE,
        (Algorithm.MINMAX, Phase.PREFILL): GraphStage.QUANTIZED_PREFILL,
        (Algorithm.MINMAX, Phase.DECODE): GraphStage.QUANTIZED_DECODE,
    }[(capability.algorithm, capability.phase)].value
    target = (
        Algorithm.FP16.value
        if capability.algorithm is Algorithm.FP16
        else capability.target.value
    )
    properties = {"mdc.model_kind": capability.model.value}
    if save_kv_cache is not None:
        properties["save_kv_cache"] = str(save_kv_cache).lower()
    return ValidatedMetadata(
        properties=properties,
        stage=stage,
        mask_mode=capability.mask_mode.value,
        algorithms=frozenset({capability.algorithm.value}),
        targets=frozenset({target}),
    )


def _model(
    capability: Capability,
    *,
    save_kv_cache: bool = True,
) -> onnx.ModelProto:
    if capability.phase is Phase.PREFILL:
        inputs = [
            helper.make_tensor_value_info(
                "input_ids",
                onnx.TensorProto.INT64,
                [1, 8],
            )
        ]
        output_shape = [1, 8, 128]
        cache_shape = [1, 2, 8, 64]
    else:
        cache_dtype = (
            onnx.TensorProto.INT8
            if capability.target is Target.ATTENTION
            else onnx.TensorProto.FLOAT16
        )
        inputs = [
            helper.make_tensor_value_info(
                "input_ids",
                onnx.TensorProto.INT64,
                [1, 1],
            ),
            *[
                helper.make_tensor_value_info(
                    f"past.{layer}.{kind}",
                    cache_dtype,
                    [1, 2, 7, 64],
                )
                for layer in range(2)
                for kind in ("key", "value")
            ],
        ]
        output_shape = [1, 1, 128]
        cache_shape = [1, 2, 8, 64]
    cache_dtype = (
        onnx.TensorProto.INT8
        if capability.target is Target.ATTENTION
        else onnx.TensorProto.FLOAT16
    )
    outputs = [
        helper.make_tensor_value_info(
            "logits",
            onnx.TensorProto.FLOAT16,
            output_shape,
        )
    ]
    if save_kv_cache:
        outputs.extend(
            helper.make_tensor_value_info(
                f"present.{layer}.{kind}",
                cache_dtype,
                cache_shape,
            )
            for layer in range(2)
            for kind in ("key", "value")
        )
    graph = helper.make_graph(
        [],
        "release",
        inputs,
        outputs,
    )
    return helper.make_model(graph)


def _stub_validation_layers(
    monkeypatch: pytest.MonkeyPatch,
    model: onnx.ModelProto,
    metadata: ValidatedMetadata,
    observed_targets: frozenset[str],
) -> None:
    monkeypatch.setattr(
        release_validation,
        "load_validated_mdc_artifact",
        lambda path: ValidatedMdcArtifact(
            model=model,
            metadata=metadata,
            topology=QuantizationTopologyEvidence(
                operator_counts=tuple(
                    sorted(
                        {
                            "FusedInferAttentionScore": 2,
                            "ApplyRotaryPosEmb": 2,
                            "NPURmsNorm": 1,
                        }.items()
                    )
                ),
                observed_quantized_targets=observed_targets,
            ),
        ),
    )


def test_release_artifact_returns_immutable_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _capability()
    model = _model(capability)
    _stub_validation_layers(
        monkeypatch,
        model,
        _metadata(capability),
        frozenset(),
    )

    evidence = release_validation.validate_release_artifact(
        Path("artifact.onnx"),
        capability,
    )

    assert evidence.input_names == ("input_ids",)
    assert evidence.output_names == (
        "logits",
        "present.0.key",
        "present.0.value",
        "present.1.key",
        "present.1.value",
    )
    assert evidence.declared_targets == {"fp16"}
    assert evidence.observed_quantized_targets == frozenset()
    assert evidence.operator_counts == (
        ("ApplyRotaryPosEmb", 2),
        ("FusedInferAttentionScore", 2),
        ("NPURmsNorm", 1),
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"properties": {"mdc.model_kind": "moe"}}, "model kind"),
        ({"stage": GraphStage.FLOAT_DECODE.value}, "graph stage"),
        ({"mask_mode": "maskless"}, "mask mode"),
        ({"algorithms": frozenset({"minmax"})}, "algorithm"),
        ({"targets": frozenset({"linear"})}, "target"),
    ],
)
def test_release_artifact_rejects_metadata_capability_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    message: str,
) -> None:
    capability = _capability()
    model = _model(capability)
    _stub_validation_layers(
        monkeypatch,
        model,
        replace(_metadata(capability), **changes),
        frozenset(),
    )

    with pytest.raises(
        OnnxExportError,
        match=rf"model=dense.*{message}",
    ):
        release_validation.validate_release_artifact("artifact.onnx", capability)


def test_release_artifact_rejects_decode_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _capability(phase=Phase.DECODE)
    model = _model(capability)
    model.graph.input[1].name = "past.0.value"
    _stub_validation_layers(
        monkeypatch,
        model,
        _metadata(capability),
        frozenset(),
    )

    with pytest.raises(OnnxExportError, match="past KV ABI"):
        release_validation.validate_release_artifact("artifact.onnx", capability)


@pytest.mark.parametrize("phase", [Phase.PREFILL, Phase.DECODE])
@pytest.mark.parametrize("serialized", ["false", None])
def test_release_artifact_accepts_logits_only_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    phase: Phase,
    serialized: str | None,
) -> None:
    capability = _capability(phase=phase)
    model = _model(capability, save_kv_cache=False)
    metadata = _metadata(capability, save_kv_cache=None)
    if serialized is not None:
        metadata.properties["save_kv_cache"] = serialized
    _stub_validation_layers(monkeypatch, model, metadata, frozenset())

    evidence = release_validation.validate_release_artifact(
        "artifact.onnx",
        capability,
    )

    assert evidence.output_names == ("logits",)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "key/value pairs"),
        ("reordered", "ordered contiguous key/value pairs"),
    ],
)
def test_release_artifact_rejects_invalid_present_cache_order(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    capability = _capability()
    model = _model(capability)
    if mutation == "missing":
        del model.graph.output[-1]
    else:
        model.graph.output[1].name = "present.0.value"
        model.graph.output[2].name = "present.0.key"
    _stub_validation_layers(
        monkeypatch,
        model,
        _metadata(capability),
        frozenset(),
    )

    with pytest.raises(OnnxExportError, match=message):
        release_validation.validate_release_artifact("artifact.onnx", capability)


def test_release_artifact_wraps_unreadable_path_with_capability(
    tmp_path: Path,
) -> None:
    path = tmp_path / "broken.onnx"
    path.write_bytes(b"not-onnx")

    with pytest.raises(
        OnnxExportError,
        match=r"model=dense.*Cannot read ONNX protobuf",
    ):
        release_validation.validate_release_artifact(path, _capability())


def test_release_artifact_wraps_missing_external_data(
    tmp_path: Path,
) -> None:
    weight = numpy_helper.from_array(
        np.ones((1,), dtype=np.float32),
        name="weight",
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Add", ["input", "weight"], ["output"])],
            "external",
            [
                helper.make_tensor_value_info(
                    "input",
                    onnx.TensorProto.FLOAT,
                    [1],
                )
            ],
            [
                helper.make_tensor_value_info(
                    "output",
                    onnx.TensorProto.FLOAT,
                    [1],
                )
            ],
            initializer=[weight],
        )
    )
    path = tmp_path / "external.onnx"
    data_path = tmp_path / "external.onnx.data"
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=0,
    )
    data_path.unlink()

    with pytest.raises(
        OnnxExportError,
        match=r"model=dense.*Cannot read ONNX protobuf",
    ):
        release_validation.validate_release_artifact(path, _capability())


def _moe_model(weight_dtypes: tuple[np.dtype[np.generic], ...]) -> onnx.ModelProto:
    initializers = []
    nodes = []
    for index, dtype in enumerate(weight_dtypes):
        weight_name = f"weight.{index}"
        scale_name = f"scale.{index}"
        initializers.extend(
            (
                numpy_helper.from_array(
                    np.ones((1,), dtype=dtype),
                    name=weight_name,
                ),
                numpy_helper.from_array(
                    np.ones((1,), dtype=np.float32),
                    name=scale_name,
                ),
            )
        )
        nodes.append(
            helper.make_node(
                "MoeExpert",
                ["x", "ids", "scores", weight_name, scale_name, ""],
                [f"output.{index}"],
            )
        )
    return helper.make_model(
        helper.make_graph(nodes, "moe", [], [], initializer=initializers)
    )


def _reference_quantized_target_families(
    model: onnx.ModelProto,
) -> frozenset[str]:
    initializers = {
        item.name: item for item in model.graph.initializer
    }
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    result: set[str] = set()
    moe_nodes = [
        node for node in model.graph.node if node.op_type == "MoeExpert"
    ]
    quantized_moe_nodes = [
        node
        for node in moe_nodes
        if (
            len(node.input) > 4
            and bool(node.input[4])
            and (weight := initializers.get(node.input[3])) is not None
            and weight.data_type == onnx.TensorProto.INT8
        )
    ]
    if quantized_moe_nodes and len(quantized_moe_nodes) != len(moe_nodes):
        raise OnnxExportError(
            "MoeExpert quantization coverage is inconsistent"
        )
    if quantized_moe_nodes:
        result.add("moe")
    attention_quantization_inputs = (
        AttentionInput.DEQUANT_SCALE1,
        AttentionInput.QUANT_SCALE1,
        AttentionInput.DEQUANT_SCALE2,
        AttentionInput.QUANT_SCALE2,
        AttentionInput.QUANT_OFFSET2,
        AttentionInput.ANTIQUANT_SCALE,
        AttentionInput.ANTIQUANT_OFFSET,
        AttentionInput.KEY_ANTIQUANT_SCALE,
        AttentionInput.KEY_ANTIQUANT_OFFSET,
        AttentionInput.VALUE_ANTIQUANT_SCALE,
        AttentionInput.VALUE_ANTIQUANT_OFFSET,
        AttentionInput.KEY_ROPE_ANTIQUANT_SCALE,
        AttentionInput.DEQUANT_SCALE_QUERY,
    )
    attention_nodes = [
        node
        for node in model.graph.node
        if node.op_type == "FusedInferAttentionScore"
    ]
    quantized_attention_nodes = [
        node
        for node in attention_nodes
        if any(
            index < len(node.input) and bool(node.input[index])
            for index in attention_quantization_inputs
        )
    ]
    if (
        quantized_attention_nodes
        and len(quantized_attention_nodes) != len(attention_nodes)
    ):
        raise OnnxExportError(
            "Attention quantization coverage is inconsistent"
        )
    if quantized_attention_nodes:
        result.add("attention")
    for node in model.graph.node:
        if node.op_type != "AscendDequant" or not node.input:
            continue
        accumulator = producers.get(node.input[0])
        if (
            accumulator is None
            or accumulator.op_type != "MatMul"
            or not accumulator.input
        ):
            continue
        quantizer = producers.get(accumulator.input[0])
        if (
            quantizer is not None
            and quantizer.op_type == "NPUAscendQuantV2"
        ):
            result.add("linear")
    return frozenset(result)


def _target_model(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto] | None = None,
) -> onnx.ModelProto:
    return helper.make_model(
        helper.make_graph(
            nodes,
            "targets",
            [],
            [],
            initializer=initializers or [],
        )
    )


def _attention_node(name: str, *, quantized: bool) -> onnx.NodeProto:
    inputs = [""] * ATTENTION_INPUT_COUNT
    if quantized:
        inputs[AttentionInput.DEQUANT_SCALE1] = f"{name}.scale"
    return helper.make_node(
        "FusedInferAttentionScore",
        inputs,
        [f"{name}.output", f"{name}.lse"],
        name=name,
    )


def _linear_nodes(prefix: str) -> list[onnx.NodeProto]:
    return [
        helper.make_node(
            "NPUAscendQuantV2",
            [f"{prefix}.input"],
            [f"{prefix}.quantized"],
        ),
        helper.make_node(
            "MatMul",
            [f"{prefix}.quantized", f"{prefix}.weight"],
            [f"{prefix}.accumulator"],
        ),
        helper.make_node(
            "AscendDequant",
            [f"{prefix}.accumulator"],
            [f"{prefix}.output"],
        ),
    ]


def test_float_moe_is_not_a_quantized_target() -> None:
    model = _moe_model((np.dtype(np.float16), np.dtype(np.float16)))

    assert quantized_target_families(model) == frozenset()


def test_int8_moe_is_a_quantized_target() -> None:
    model = _moe_model((np.dtype(np.int8), np.dtype(np.int8)))

    assert quantized_target_families(model) == {"moe"}


def test_mixed_moe_quantization_is_rejected() -> None:
    model = _moe_model((np.dtype(np.int8), np.dtype(np.float16)))

    with pytest.raises(
        OnnxExportError,
        match=r"^MoeExpert quantization coverage is inconsistent$",
    ):
        quantized_target_families(model)


def test_mixed_attention_quantization_is_rejected() -> None:
    float_inputs = [""] * ATTENTION_INPUT_COUNT
    quantized_inputs = [""] * ATTENTION_INPUT_COUNT
    quantized_inputs[AttentionInput.DEQUANT_SCALE1] = "scale"
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "FusedInferAttentionScore",
                    float_inputs,
                    ["float", "float_lse"],
                ),
                helper.make_node(
                    "FusedInferAttentionScore",
                    quantized_inputs,
                    ["quantized", "quantized_lse"],
                ),
            ],
            "attention",
            [],
            [],
        )
    )

    with pytest.raises(
        OnnxExportError,
        match=r"^Attention quantization coverage is inconsistent$",
    ):
        quantized_target_families(model)


def test_quantized_target_families_matches_frozen_reference() -> None:
    int8_weight = numpy_helper.from_array(
        np.ones((1,), dtype=np.int8),
        name="moe.weight",
    )
    scale = numpy_helper.from_array(
        np.ones((1,), dtype=np.float32),
        name="moe.scale",
    )
    moe = helper.make_node(
        "MoeExpert",
        ["x", "ids", "scores", "moe.weight", "moe.scale", ""],
        ["moe.output"],
    )
    broken_linear_nodes = [
        helper.make_node("AscendDequant", [], ["no_input"]),
        helper.make_node("AscendDequant", ["missing"], ["no_producer"]),
        helper.make_node("Identity", ["x"], ["not_matmul"]),
        helper.make_node("AscendDequant", ["not_matmul"], ["wrong_producer"]),
        helper.make_node("MatMul", [], ["empty_matmul"]),
        helper.make_node("AscendDequant", ["empty_matmul"], ["empty_input"]),
        helper.make_node("MatMul", ["missing_quantizer"], ["missing_acc"]),
        helper.make_node("AscendDequant", ["missing_acc"], ["missing_quant"]),
        helper.make_node("Identity", ["x"], ["identity_quantizer"]),
        helper.make_node("MatMul", ["identity_quantizer"], ["identity_acc"]),
        helper.make_node("AscendDequant", ["identity_acc"], ["wrong_quant"]),
    ]
    cases = [
        _target_model([]),
        _target_model(
            [
                helper.make_node(
                    "Identity", [f"x.{index}"], [f"y.{index}"]
                )
                for index in range(64)
            ]
        ),
        _moe_model((np.dtype(np.float16), np.dtype(np.float16))),
        _moe_model((np.dtype(np.int8), np.dtype(np.int8))),
        _target_model([_attention_node("float", quantized=False)]),
        _target_model([_attention_node("quantized", quantized=True)]),
        _target_model(_linear_nodes("valid")),
        _target_model(
            [
                *_linear_nodes("all"),
                _attention_node("all", quantized=True),
                moe,
            ],
            [int8_weight, scale],
        ),
        _target_model([*_linear_nodes("valid"), *broken_linear_nodes]),
    ]

    for model in cases:
        actual = quantized_target_families(model)
        assert actual == _reference_quantized_target_families(model)
        assert isinstance(actual, frozenset)
    assert quantized_target_families(cases[-3]) == frozenset({"linear"})
    assert quantized_target_families(cases[-2]) == frozenset(
        {"linear", "attention", "moe"}
    )


@pytest.mark.parametrize(
    ("duplicate_nodes", "expected"),
    [
        (
            [
                helper.make_node("NPUAscendQuantV2", ["x"], ["quantized"]),
                helper.make_node("Identity", ["x"], ["quantized"]),
            ],
            frozenset(),
        ),
        (
            [
                helper.make_node("Identity", ["x"], ["quantized"]),
                helper.make_node("NPUAscendQuantV2", ["x"], ["quantized"]),
            ],
            frozenset({"linear"}),
        ),
    ],
)
def test_quantized_target_producer_index_is_last_wins(
    duplicate_nodes: list[onnx.NodeProto],
    expected: frozenset[str],
) -> None:
    model = _target_model(
        [
            helper.make_node("Identity", ["x"], [""]),
            *duplicate_nodes,
            helper.make_node(
                "MatMul", ["quantized", "weight"], ["accumulator"]
            ),
            helper.make_node(
                "AscendDequant", ["accumulator"], ["output"]
            ),
        ]
    )

    assert quantized_target_families(model) == expected


@pytest.mark.parametrize(
    ("dtypes", "expected"),
    [
        ((np.float16, np.int8), frozenset({"moe"})),
        ((np.int8, np.float16), frozenset()),
    ],
)
def test_quantized_target_initializer_index_is_last_wins(
    dtypes: tuple[type[np.generic], type[np.generic]],
    expected: frozenset[str],
) -> None:
    initializers = [
        numpy_helper.from_array(np.ones((1,), dtype=dtype), name="weight")
        for dtype in dtypes
    ]
    initializers.append(
        numpy_helper.from_array(
            np.ones((1,), dtype=np.float32),
            name="scale",
        )
    )
    model = _target_model(
        [
            helper.make_node(
                "MoeExpert",
                ["x", "ids", "scores", "weight", "scale", ""],
                ["output"],
            )
        ],
        initializers,
    )

    assert quantized_target_families(model) == expected


@pytest.mark.parametrize("attention_first", [False, True])
def test_moe_coverage_error_precedes_attention_coverage_error(
    attention_first: bool,
) -> None:
    moe_model = _moe_model((np.dtype(np.int8), np.dtype(np.float16)))
    moe_nodes = list(moe_model.graph.node)
    attention_nodes = [
        _attention_node("float", quantized=False),
        _attention_node("quantized", quantized=True),
    ]
    sections = (
        [attention_nodes, moe_nodes]
        if attention_first
        else [moe_nodes, attention_nodes]
    )
    model = _target_model(
        [node for section in sections for node in section],
        list(moe_model.graph.initializer),
    )

    with pytest.raises(
        OnnxExportError,
        match=r"^MoeExpert quantization coverage is inconsistent$",
    ):
        quantized_target_families(model)


def _topology_model(
    nodes: list[onnx.NodeProto],
    *,
    initializers: list[onnx.TensorProto] | None = None,
) -> onnx.ModelProto:
    return helper.make_model(
        helper.make_graph(
            nodes,
            "topology",
            [
                helper.make_tensor_value_info(
                    "input",
                    onnx.TensorProto.FLOAT,
                    [1],
                )
            ],
            [
                helper.make_tensor_value_info(
                    "logits",
                    onnx.TensorProto.FLOAT,
                    [1],
                )
            ],
            initializer=initializers,
        )
    )


def test_release_topology_rejects_qdq_nodes() -> None:
    scale = numpy_helper.from_array(
        np.ones((1,), dtype=np.float32),
        name="scale",
    )
    zero_point = numpy_helper.from_array(
        np.zeros((1,), dtype=np.int8),
        name="zero_point",
    )
    model = _topology_model(
        [
            helper.make_node(
                "QuantizeLinear",
                ["input", "scale", "zero_point"],
                ["logits"],
            )
        ],
        initializers=[scale, zero_point],
    )

    with pytest.raises(OnnxExportError, match="must not contain QDQ"):
        validate_graph_topology(model, "masked")


def test_release_topology_rejects_ssa_output_reuse() -> None:
    model = _topology_model(
        [
            helper.make_node("Identity", ["input"], ["shared"]),
            helper.make_node("Identity", ["shared"], ["shared"]),
            helper.make_node("Identity", ["shared"], ["logits"]),
        ]
    )

    with pytest.raises(OnnxExportError, match="SSA violation"):
        validate_graph_topology(model, "masked")


def test_release_topology_rejects_reverse_reference() -> None:
    model = _topology_model(
        [
            helper.make_node("Identity", ["future"], ["logits"], name="early"),
            helper.make_node("Identity", ["input"], ["future"]),
        ]
    )

    with pytest.raises(OnnxExportError, match="not topologically sorted"):
        validate_graph_topology(model, "masked")


def test_release_topology_rejects_constant_output() -> None:
    constant = numpy_helper.from_array(
        np.ones((1,), dtype=np.float32),
        name="value",
    )
    model = _topology_model(
        [helper.make_node("Constant", [], ["logits"], value=constant)]
    )

    with pytest.raises(OnnxExportError, match="constant placeholders"):
        validate_graph_topology(model, "masked")


def test_release_topology_rejects_isolated_custom_node() -> None:
    model = _topology_model(
        [
            helper.make_node("Identity", ["input"], ["logits"]),
            helper.make_node(
                "MoeExpert",
                ["input", "input", "input", "input", "", ""],
                ["dead"],
                name="isolated_moe",
            ),
        ]
    )

    with pytest.raises(OnnxExportError, match="isolated_moe"):
        validate_custom_node_reachability(model, {"mdc.target": "fp16"})
