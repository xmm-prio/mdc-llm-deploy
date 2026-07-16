"""Release-specific semantic acceptance contracts."""

from __future__ import annotations

from collections import Counter
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
from mdc_llm_deploy.onnx.validation.topology import (
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
    return ValidatedMetadata(
        properties={"mdc.model_kind": capability.model.value},
        stage=stage,
        mask_mode=capability.mask_mode.value,
        algorithms=frozenset({capability.algorithm.value}),
        targets=frozenset({target}),
    )


def _model(capability: Capability) -> onnx.ModelProto:
    if capability.phase is Phase.PREFILL:
        inputs = [
            helper.make_tensor_value_info(
                "input_ids",
                onnx.TensorProto.INT64,
                [1, 8],
            )
        ]
        output_shape = [1, 8, 128]
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
    graph = helper.make_graph(
        [],
        "release",
        inputs,
        [
            helper.make_tensor_value_info(
                "logits",
                onnx.TensorProto.FLOAT16,
                output_shape,
            )
        ],
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
        "validate_serialized_model",
        lambda path: model,
    )
    monkeypatch.setattr(
        release_validation,
        "validate_metadata",
        lambda value: metadata,
    )
    monkeypatch.setattr(
        release_validation,
        "validate_graph_topology",
        lambda value, mask_mode: Counter(
            {
                "FusedInferAttentionScore": 2,
                "ApplyRotaryPosEmb": 2,
                "NPURmsNorm": 1,
            }
        ),
    )
    monkeypatch.setattr(
        release_validation,
        "validate_custom_node_reachability",
        lambda value, properties: None,
    )
    monkeypatch.setattr(
        release_validation,
        "quantized_target_families",
        lambda value: observed_targets,
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
    assert evidence.output_names == ("logits",)
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

    with pytest.raises(OnnxExportError, match="incomplete or out of order"):
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


def test_float_moe_is_not_a_quantized_target() -> None:
    model = _moe_model((np.dtype(np.float16), np.dtype(np.float16)))

    assert quantized_target_families(model) == frozenset()


def test_int8_moe_is_a_quantized_target() -> None:
    model = _moe_model((np.dtype(np.int8), np.dtype(np.int8)))

    assert quantized_target_families(model) == {"moe"}


def test_mixed_moe_quantization_is_rejected() -> None:
    model = _moe_model((np.dtype(np.int8), np.dtype(np.float16)))

    with pytest.raises(OnnxExportError, match="coverage is inconsistent"):
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

    with pytest.raises(OnnxExportError, match="coverage is inconsistent"):
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
