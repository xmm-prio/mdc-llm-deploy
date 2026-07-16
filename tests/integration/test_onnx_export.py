"""Tests for standard-intermediate and MDC-dialect ONNX export."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.export import convert_to_decode, export
from mdc_llm_deploy.graph import GraphStage, QuantizedTarget, metadata, set_metadata
from mdc_llm_deploy.models.tiny import TinyConfig, TinyQwen3Dense, TinyQwen3Moe
from mdc_llm_deploy.onnx_export import onnx_export
from mdc_llm_deploy.onnx_export.graph_cleanup import prune_unreachable
from mdc_llm_deploy.onnx_export.validator import validate_mdc_model
from mdc_llm_deploy.quantization import oneshot

pytestmark = pytest.mark.integration


def _graph() -> torch.fx.GraphModule:
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    return export(TinyQwen3Dense().eval(), inputs)


def _quantized_graph() -> torch.fx.GraphModule:
    graph = _graph()
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    return oneshot(
        graph,
        "configs/minmax-linear-w8a8.json",
        [inputs],
    )


def _moe_config() -> TinyConfig:
    return TinyConfig(
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        moe_intermediate_size=128,
    )


@pytest.fixture(scope="module")
def moe_model_bytes(
    tmp_path_factory: pytest.TempPathFactory,
) -> bytes:
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    graph = export(TinyQwen3Moe(_moe_config()).eval(), inputs)
    oneshot(graph, "configs/minmax-moe-w8a8.json", [inputs])
    model = onnx_export(
        graph,
        tmp_path_factory.mktemp("moe-model") / "moe.onnx",
        mask_mode="masked",
    )
    return model.SerializeToString()


@pytest.fixture
def moe_model(moe_model_bytes: bytes) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.ParseFromString(moe_model_bytes)
    return model


def _set_model_property(
    model: onnx.ModelProto,
    key: str,
    value: str,
) -> None:
    properties = {item.key: item.value for item in model.metadata_props}
    properties[key] = value
    helper.set_model_props(model, properties)


def _values_reaching_outputs(model: onnx.ModelProto) -> set[str]:
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
    }
    pending = [item.name for item in model.graph.output]
    result: set[str] = set()
    while pending:
        value = pending.pop()
        if value in result:
            continue
        result.add(value)
        producer = producers.get(value)
        if producer is not None:
            pending.extend(name for name in producer.input if name)
    return result


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "mdc.graph_schema_version",
            "999",
            "schema version",
        ),
        ("mdc.stage", "FLOAT_PREFILL", "declare fp16"),
        ("mdc.model_kind", "unknown", "model kind"),
        ("mdc.algorithm", "gptq", "does not support GPTQ"),
        ("mdc.target", "unknown", "target metadata"),
        ("mdc.target", "moe,moe", "target metadata"),
        (
            "mdc.target",
            "attention",
            "does not match quantized topology",
        ),
    ],
)
def test_validator_rejects_invalid_core_metadata(
    moe_model: onnx.ModelProto,
    key: str,
    value: str,
    message: str,
) -> None:
    _set_model_property(moe_model, key, value)

    with pytest.raises(OnnxExportError, match=message):
        validate_mdc_model(moe_model)


def _assert_compact(model: onnx.ModelProto) -> None:
    reachable = _values_reaching_outputs(model)
    assert all(
        any(output in reachable for output in node.output)
        for node in model.graph.node
    )
    used_initializers = {
        name for node in model.graph.node for name in node.input if name
    }
    used_initializers.update(item.name for item in model.graph.output)
    assert {
        item.name for item in model.graph.initializer
    } <= used_initializers


@pytest.fixture
def exported_pair(tmp_path: Path) -> tuple[onnx.ModelProto, onnx.ModelProto]:
    graph = _graph()
    masked = onnx_export(graph, tmp_path / "masked.onnx", mask_mode="masked")
    maskless = onnx_export(graph, tmp_path / "maskless.onnx", mask_mode="maskless")
    return masked, maskless


def test_mdc_onnx_custom_spine_reaches_outputs_in_both_mask_modes(
    exported_pair: tuple[onnx.ModelProto, onnx.ModelProto],
) -> None:
    for model in exported_pair:
        initializer_names = {item.name for item in model.graph.initializer}
        producers = {
            output: node.op_type for node in model.graph.node for output in node.output
        }
        reachable = _values_reaching_outputs(model)
        custom_nodes = [
            node
            for node in model.graph.node
            if node.op_type
            in {"NPURmsNorm", "ApplyRotaryPosEmb", "FusedInferAttentionScore"}
        ]

        assert not initializer_names.intersection(
            item.name for item in model.graph.output
        )
        assert all(
            producers[item.name] not in {"Constant", "ConstantOfShape"}
            for item in model.graph.output
        )
        assert {node.op_type for node in custom_nodes} == {
            "NPURmsNorm",
            "ApplyRotaryPosEmb",
            "FusedInferAttentionScore",
        }
        assert all(
            any(output in reachable for output in node.output)
            for node in custom_nodes
        )
        attention = next(
            node
            for node in model.graph.node
            if node.op_type == "FusedInferAttentionScore"
        )
        assert len(attention.input) == 29
        assert bool(attention.input[4]) == (
            model is exported_pair[0]
        )
        assert {item.domain: item.version for item in model.opset_import}[""] == 18
        _assert_compact(model)


def test_mask_modes_lower_to_distinct_fused_attention_contracts(
    exported_pair: tuple[onnx.ModelProto, onnx.ModelProto],
) -> None:
    masked, maskless = exported_pair
    masked_attention = next(
        node for node in masked.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    maskless_attention = next(
        node for node in maskless.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    assert masked_attention.input[4]
    assert not maskless_attention.input[4]
    properties = {item.key: item.value for item in maskless.metadata_props}
    assert properties["mdc.mask_semantics"] == "all-visible-non-causal"


def test_pruning_removes_strict_identity_dead_nodes_and_unused_initializers() -> None:
    used = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="used")
    unused = numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), name="unused")
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["input", "used"], ["hidden"], name="add"),
            helper.make_node("Identity", ["hidden"], ["output"], name="identity"),
            helper.make_node("Neg", ["input"], ["dead"], name="dead"),
        ],
        "compact",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1,))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1,))],
        initializer=[used, unused],
        value_info=[
            helper.make_tensor_value_info("hidden", TensorProto.FLOAT, (1,)),
            helper.make_tensor_value_info("dead", TensorProto.FLOAT, (1,)),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    prune_unreachable(model)
    first = model.SerializeToString()
    prune_unreachable(model)

    assert [node.op_type for node in model.graph.node] == ["Add"]
    assert list(model.graph.node[0].output) == ["output"]
    assert [item.name for item in model.graph.initializer] == ["used"]
    assert not model.graph.value_info
    assert model.SerializeToString() == first


def test_pruning_preserves_identity_after_custom_operator() -> None:
    gamma = numpy_helper.from_array(
        np.ones((1,), dtype=np.float32),
        name="gamma",
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "NPURmsNorm",
                ["input", "gamma"],
                ["normalized", "rstd"],
                name="norm",
                epsilon=1e-6,
            ),
            helper.make_node(
                "Identity",
                ["normalized"],
                ["output"],
                name="custom_boundary",
            ),
        ],
        "custom-identity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1,))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1,))],
        initializer=[gamma],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    prune_unreachable(model)

    assert [node.op_type for node in model.graph.node] == [
        "NPURmsNorm",
        "Identity",
    ]


@pytest.mark.parametrize("mask_mode", ["masked", "maskless"])
def test_decode_onnx_is_compact_and_preserves_static_position_contract(
    tmp_path: Path,
    mask_mode: str,
) -> None:
    graph = _graph()
    convert_to_decode(graph)

    model = onnx_export(
        graph,
        tmp_path / f"decode-{mask_mode}.onnx",
        mask_mode=mask_mode,
    )

    _assert_compact(model)
    properties = {item.key: item.value for item in model.metadata_props}
    attention = next(
        node for node in model.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    assert properties["mdc.stage"] == "FLOAT_DECODE"
    assert bool(attention.input[4]) == (mask_mode == "masked")
    assert [item.name for item in model.graph.input] == [
        "input_ids",
        "past_key_values.0.key",
        "past_key_values.0.value",
    ]


def test_standard_and_atomic_temporary_files_are_removed(
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.onnx"

    onnx_export(_graph(), target, mask_mode="masked")

    assert target.is_file()
    assert tuple(path for path in tmp_path.iterdir() if path != target) == ()


def test_failure_preserves_existing_target(tmp_path: Path) -> None:
    graph = _graph()
    value = metadata(graph)
    properties = dict(value.properties)
    properties["rms_norm_epsilon"] = 1e-5
    set_metadata(graph, replace(value, properties=properties))
    target = tmp_path / "existing.onnx"
    target.write_bytes(b"keep-me")

    with pytest.raises(OnnxExportError, match="epsilon"):
        onnx_export(graph, target, mask_mode="masked", overwrite=True)

    assert target.read_bytes() == b"keep-me"
    assert tuple(path for path in tmp_path.iterdir() if path != target) == ()


def test_gptq_w4_rejection_is_stable_and_writes_nothing(tmp_path: Path) -> None:
    graph = _graph()
    value = metadata(graph)
    set_metadata(
        graph,
        replace(
            value,
            stage=GraphStage.QUANTIZED_PREFILL,
            quantized_targets=(
                QuantizedTarget(
                    "lm_head",
                    "linear",
                    "gptq",
                    4,
                    "per_channel",
                    True,
                    (0.1,),
                    (0,),
                ),
            ),
            config_fingerprint="b" * 64,
        ),
    )
    target = tmp_path / "forbidden.onnx"

    with pytest.raises(
        OnnxExportError,
        match="GPTQ is FX-only and does not support ONNX or ATC",
    ):
        onnx_export(graph, target, mask_mode="masked")

    assert not target.exists()


def test_validator_rejects_mask_contract_violation(
    exported_pair: tuple[onnx.ModelProto, onnx.ModelProto],
) -> None:
    masked, _ = exported_pair
    attention = next(
        node for node in masked.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    attention.input[4] = ""

    with pytest.raises(OnnxExportError, match="atten_mask"):
        validate_mdc_model(masked)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("pre_tokens", 0, "pre_tokens"),
        ("next_tokens", 0, "next_tokens"),
        ("softmax_lse_flag", 1, "softmax_lse_flag"),
        ("num_key_value_heads", 3, "head counts"),
        ("scale", float("nan"), "finite and positive"),
    ],
)
def test_validator_rejects_invalid_attention_release_attributes(
    exported_pair: tuple[onnx.ModelProto, onnx.ModelProto],
    attribute: str,
    value: float,
    message: str,
) -> None:
    masked, _ = exported_pair
    attention = next(
        node for node in masked.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    target = next(item for item in attention.attribute if item.name == attribute)
    if attribute == "scale":
        target.f = value
    else:
        target.i = int(value)

    with pytest.raises(OnnxExportError, match=message):
        validate_mdc_model(masked)


def test_validator_rejects_invalid_attention_attribute_wire_type(
    exported_pair: tuple[onnx.ModelProto, onnx.ModelProto],
) -> None:
    masked, _ = exported_pair
    attention = next(
        node for node in masked.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    attribute = next(
        item for item in attention.attribute if item.name == "pre_tokens"
    )
    attribute.CopyFrom(helper.make_attribute("pre_tokens", 2147483647.0))

    with pytest.raises(OnnxExportError, match="invalid ONNX type"):
        validate_mdc_model(masked)


@pytest.mark.parametrize("mutation", ["query", "lse", "duplicate"])
def test_validator_rejects_incomplete_attention_structure(
    exported_pair: tuple[onnx.ModelProto, onnx.ModelProto],
    mutation: str,
) -> None:
    masked, _ = exported_pair
    attention = next(
        node for node in masked.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    if mutation == "query":
        attention.input[0] = ""
        message = "requires query"
    elif mutation == "lse":
        lse = next(
            item
            for item in masked.graph.value_info
            if item.name == attention.output[1]
        )
        lse.type.tensor_type.elem_type = TensorProto.FLOAT16
        message = "LSE output"
    else:
        duplicate = onnx.NodeProto()
        duplicate.CopyFrom(attention)
        duplicate.name = f"{attention.name}.duplicate"
        duplicate.output[:] = ["duplicate.attention", "duplicate.lse"]
        masked.graph.node.append(duplicate)
        message = "exactly one FusedInferAttentionScore"

    with pytest.raises(OnnxExportError, match=message):
        validate_mdc_model(masked)


def test_validator_rejects_mixed_attention_quantization_parameters(
    tmp_path: Path,
) -> None:
    graph = _graph()
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    oneshot(graph, "configs/minmax-attention-a8.json", [inputs])
    model = onnx_export(graph, tmp_path / "attention.onnx", mask_mode="masked")
    attention = next(
        node for node in model.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    query = next(
        item for item in model.graph.value_info if item.name == attention.input[0]
    )
    query.type.tensor_type.elem_type = TensorProto.FLOAT16

    with pytest.raises(
        OnnxExportError,
        match="must not provide dequant_scale_query",
    ):
        validate_mdc_model(model)


def test_quantized_attention_exports_complete_atc_scale_contract(
    tmp_path: Path,
) -> None:
    graph = _graph()
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    oneshot(graph, "configs/minmax-attention-a8.json", [inputs])
    targets = {
        item.fqn.rsplit(".", 1)[-1]: item
        for item in metadata(graph).quantized_targets
    }

    model = onnx_export(graph, tmp_path / "attention.onnx", mask_mode="masked")

    attention = next(
        node for node in model.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    initializers = {item.name: item for item in model.graph.initializer}
    actual = {
        index: float(
            numpy_helper.to_array(initializers[attention.input[index]]).reshape(-1)[0]
        )
        for index in (7, 8, 9, 17, 19, 27)
    }
    expected = {
        7: targets["query"].scale[0] * targets["key"].scale[0],
        8: 1.0 / targets["score"].scale[0],
        9: targets["score"].scale[0] * targets["value"].scale[0],
        17: targets["key"].scale[0],
        19: targets["value"].scale[0],
        27: targets["query"].scale[0],
    }
    assert all(attention.input[index] for index in actual)
    for index, value in expected.items():
        assert np.isclose(actual[index], value, rtol=1e-6, atol=0)


@pytest.mark.parametrize(("slot", "name"), [(7, "dequant_scale1"), (9, "dequant_scale2")])
def test_validator_rejects_missing_attention_accumulator_scale(
    tmp_path: Path,
    slot: int,
    name: str,
) -> None:
    graph = _graph()
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    oneshot(graph, "configs/minmax-attention-a8.json", [inputs])
    model = onnx_export(graph, tmp_path / "attention.onnx", mask_mode="masked")
    attention = next(
        node for node in model.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    attention.input[slot] = ""

    with pytest.raises(OnnxExportError, match=name):
        validate_mdc_model(model)


def test_atc_supported_moe_lowering_replaces_standard_body_and_reaches_logits(
    tmp_path: Path,
) -> None:
    config = _moe_config()
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    graph = export(TinyQwen3Moe(config).eval(), inputs)
    oneshot(graph, "configs/minmax-moe-w8a8.json", [inputs])
    graph_metadata = metadata(graph)
    activation_qparams = graph_metadata.properties["activation_qparams"]
    down_projection_fqns = (
        "experts.0.down_proj",
        "experts.1.down_proj",
        "experts.2.down_proj",
        "experts.3.down_proj",
        "shared_expert.down_proj",
    )
    expected_intermediate_scales = [
        float(
            activation_qparams[
                next(name for name in activation_qparams if fragment in name)
            ]["scale"][0]
        )
        for fragment in down_projection_fqns
    ]
    expected_intermediate_offsets = [
        int(
            activation_qparams[
                next(name for name in activation_qparams if fragment in name)
            ]["zero_point"][0]
        )
        for fragment in down_projection_fqns
    ]

    model = onnx_export(graph, tmp_path / "moe.onnx", mask_mode="masked")

    moe = next(node for node in model.graph.node if node.op_type == "MoeExpert")
    reachable = _values_reaching_outputs(model)
    initializers = {item.name: item for item in model.graph.initializer}
    specs = {
        item.name: tuple(
            dimension.dim_value
            for dimension in item.type.tensor_type.shape.dim
        )
        for item in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        )
    }
    properties = {item.key: item.value for item in model.metadata_props}
    scale_values = numpy_helper.to_array(initializers[moe.input[4]])
    offset_values = numpy_helper.to_array(initializers[moe.input[5]])
    assert moe.output[0] in reachable
    assert specs[moe.input[1]] == (8, 3)
    assert specs[moe.input[2]] == (8, 3)
    assert tuple(initializers[moe.input[3]].dims) == (5 * 3 * 256 * 128,)
    assert tuple(initializers[moe.input[4]].dims) == (21,)
    assert tuple(initializers[moe.input[5]].dims) == (21,)
    np.testing.assert_allclose(
        scale_values[[3, 7, 11, 15, 19]],
        expected_intermediate_scales,
    )
    assert offset_values[[3, 7, 11, 15, 19]].tolist() == (
        expected_intermediate_offsets
    )
    assert properties["mdc.moe.hidden_size"] == "256"
    assert properties["mdc.moe.intermediate_size"] == "128"
    assert properties["mdc.moe.expert_order"] == "0,1,2,3,4(shared)"
    assert (
        properties["mdc.moe.weight_projection_order"]
        == "gate_proj,up_proj,down_proj"
    )
    assert tuple(
        int(item) for item in properties["mdc.moe.weight_offsets"].split(",")
    ) == tuple(index * 32768 for index in range(15))
    assert tuple(
        int(item) for item in properties["mdc.moe.weight_lengths"].split(",")
    ) == (32768,) * 15
    assert properties["mdc.moe.quant_parameter_count"] == "21"
    assert properties["mdc.moe.quant_parameter_order"] == (
        "input,"
        "expert.0.gate,expert.0.up,expert.0.intermediate,expert.0.down,"
        "expert.1.gate,expert.1.up,expert.1.intermediate,expert.1.down,"
        "expert.2.gate,expert.2.up,expert.2.intermediate,expert.2.down,"
        "expert.3.gate,expert.3.up,expert.3.intermediate,expert.3.down,"
        "expert.4.gate,expert.4.up,expert.4.intermediate,expert.4.down"
    )
    assert not any("experts." in name for name in initializers)
    assert not any("shared_expert" in name for name in initializers)


@pytest.mark.parametrize(
    ("field", "actual", "expected"),
    [
        ("num_experts", 3, "must equal 4"),
        ("num_shared_experts", 0, "must equal 1"),
        ("num_experts_per_tok", 1, "must equal 2"),
    ],
)
def test_moe_export_rejects_model_outside_fixed_abi(
    tmp_path: Path,
    field: str,
    actual: int,
    expected: str,
) -> None:
    config = replace(_moe_config(), **{field: actual})
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    graph = export(TinyQwen3Moe(config).eval(), inputs)
    oneshot(graph, "configs/minmax-moe-w8a8.json", [inputs])
    target = tmp_path / f"{field}.onnx"

    with pytest.raises(OnnxExportError, match=expected):
        onnx_export(graph, target, mask_mode="masked")

    assert not target.exists()


@pytest.mark.parametrize("mutation", ["missing", "per-token"])
def test_moe_export_rejects_invalid_intermediate_activation_qparams(
    tmp_path: Path,
    mutation: str,
) -> None:
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    graph = export(TinyQwen3Moe(_moe_config()).eval(), inputs)
    oneshot(graph, "configs/minmax-moe-w8a8.json", [inputs])
    value = metadata(graph)
    properties = dict(value.properties)
    qparams = {
        name: dict(parameters)
        for name, parameters in properties["activation_qparams"].items()
    }
    down_name = next(
        name for name in qparams if "experts.0.down_proj" in name
    )
    if mutation == "missing":
        del qparams[down_name]
        message = "lacks activation qparams"
    else:
        qparams[down_name]["granularity"] = "per_token"
        qparams[down_name]["scale"] = [0.5, 0.5]
        qparams[down_name]["zero_point"] = [0, 0]
        message = "requires scalar static INT8 activation qparams"
    properties["activation_qparams"] = qparams
    set_metadata(graph, replace(value, properties=properties))
    target = tmp_path / f"invalid-{mutation}.onnx"

    with pytest.raises(OnnxExportError, match=message):
        onnx_export(graph, target, mask_mode="masked")

    assert not target.exists()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "mdc.moe.expert_order",
            "1,0,2,3,4(shared)",
            "metadata order",
        ),
        (
            "mdc.moe.weight_projection_order",
            "up_proj,gate_proj,down_proj",
            "metadata order",
        ),
        (
            "mdc.moe.weight_offsets",
            (
                "1,32768,65536,98304,131072,163840,196608,229376,"
                "262144,294912,327680,360448,393216,425984,458752"
            ),
            "does not match tensor shapes",
        ),
        (
            "mdc.moe.weight_lengths",
            (
                "1,32768,32768,32768,32768,32768,32768,32768,"
                "32768,32768,32768,32768,32768,32768,32768"
            ),
            "does not match tensor shapes",
        ),
        (
            "mdc.moe.quant_parameter_count",
            "20",
            "metadata is incomplete",
        ),
        (
            "mdc.moe.quant_parameter_order",
            "input,expert.0.up,expert.0.gate",
            "metadata order",
        ),
    ],
)
def test_validator_rejects_tampered_moe_metadata(
    moe_model: onnx.ModelProto,
    key: str,
    value: str,
    message: str,
) -> None:
    _set_model_property(moe_model, key, value)

    with pytest.raises(OnnxExportError, match=message):
        validate_mdc_model(moe_model)


@pytest.mark.parametrize(
    ("input_index", "value", "message"),
    [
        (4, 0.0, "finite positive"),
        (4, float("nan"), "finite positive"),
        (5, 128, "signed INT8"),
    ],
)
def test_validator_rejects_invalid_moe_quant_parameter_values(
    moe_model: onnx.ModelProto,
    input_index: int,
    value: float,
    message: str,
) -> None:
    moe = next(
        node for node in moe_model.graph.node if node.op_type == "MoeExpert"
    )
    initializers = {
        item.name: item for item in moe_model.graph.initializer
    }
    initializer = initializers[moe.input[input_index]]
    values = numpy_helper.to_array(initializer).copy()
    values[3] = value
    initializer.CopyFrom(
        numpy_helper.from_array(values, name=initializer.name)
    )

    with pytest.raises(OnnxExportError, match=message):
        validate_mdc_model(moe_model)


def test_validator_rejects_tampered_moe_route_width(
    moe_model: onnx.ModelProto,
) -> None:
    moe = next(
        node for node in moe_model.graph.node if node.op_type == "MoeExpert"
    )
    values = {
        item.name: item
        for item in (
            *moe_model.graph.input,
            *moe_model.graph.output,
            *moe_model.graph.value_info,
        )
    }
    for name in moe.input[1:3]:
        values[name].type.tensor_type.shape.dim[-1].dim_value = 2

    with pytest.raises(OnnxExportError, match=r"\[tokenNum, 3\]"):
        validate_mdc_model(moe_model)


@pytest.mark.parametrize(
    ("input_index", "message"),
    [(4, "quant_scales"), (5, "quant_offsets")],
)
def test_validator_rejects_tampered_moe_quant_parameter_count(
    moe_model: onnx.ModelProto,
    input_index: int,
    message: str,
) -> None:
    moe = next(
        node for node in moe_model.graph.node if node.op_type == "MoeExpert"
    )
    initializers = {
        item.name: item for item in moe_model.graph.initializer
    }
    initializers[moe.input[input_index]].dims[0] = 20

    with pytest.raises(OnnxExportError, match=message):
        validate_mdc_model(moe_model)


def test_linear_lowering_replaces_every_fqn_matmul_and_reaches_outputs(
    tmp_path: Path,
) -> None:
    graph = _quantized_graph()
    targets = [
        item
        for item in metadata(graph).quantized_targets
        if item.target_type == "linear"
    ]

    model = onnx_export(graph, tmp_path / "linear.onnx", mask_mode="masked")

    reachable = _values_reaching_outputs(model)
    for target in targets:
        prefix = f"mdc.linear.{target.fqn}"
        quant = next(node for node in model.graph.node if node.name == f"{prefix}.quant")
        matmul = next(node for node in model.graph.node if node.name == f"{prefix}.matmul")
        dequant = next(node for node in model.graph.node if node.name == f"{prefix}.dequant")
        assert quant.op_type == "NPUAscendQuantV2"
        assert matmul.op_type == "MatMul"
        assert dequant.op_type == "AscendDequant"
        assert all(node.output[0] in reachable for node in (quant, matmul, dequant))
        assert not any(
            node.op_type in {"Gemm", "MatMul"}
            and len(node.input) >= 2
            and node.input[1] == f"graph.{target.fqn}.weight"
            for node in model.graph.node
        )
    properties = {item.key: item.value for item in model.metadata_props}
    assert properties["mdc.linear.target_count"] == str(len(targets))


def test_validator_rejects_isolated_mdc_quantization_node(tmp_path: Path) -> None:
    model = onnx_export(
        _quantized_graph(),
        tmp_path / "linear.onnx",
        mask_mode="masked",
    )
    quant = next(
        node for node in model.graph.node if node.op_type == "NPUAscendQuantV2"
    )
    model.graph.node.append(
        helper.make_node(
            "NPUAscendQuantV2",
            list(quant.input),
            ["mdc.linear.orphan.output"],
            name="mdc.linear.orphan",
            axis=-1,
            dtype=2,
        )
    )
    model.graph.value_info.append(
        helper.make_tensor_value_info(
            "mdc.linear.orphan.output",
            TensorProto.INT8,
            (1, 8, 64),
        )
    )

    with pytest.raises(OnnxExportError, match="do not reach graph outputs"):
        validate_mdc_model(model)


def test_validator_rejects_isolated_core_custom_node(
    exported_pair: tuple[onnx.ModelProto, onnx.ModelProto],
) -> None:
    masked, _ = exported_pair
    rope = next(
        node for node in masked.graph.node if node.op_type == "ApplyRotaryPosEmb"
    )
    masked.graph.node.append(
        helper.make_node(
            "ApplyRotaryPosEmb",
            list(rope.input),
            ["mdc.rope.orphan.query", "mdc.rope.orphan.key"],
            name="mdc.rope.orphan",
            layout=1,
            rotary_mode="half",
        )
    )
    for name, shape in zip(
        ("mdc.rope.orphan.query", "mdc.rope.orphan.key"),
        ((1, 8, 4, 16), (1, 8, 2, 16)),
        strict=True,
    ):
        masked.graph.value_info.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)
        )

    with pytest.raises(OnnxExportError, match="do not reach graph outputs"):
        validate_mdc_model(masked)


def test_linear_fqn_mapping_failure_writes_no_partial_file(tmp_path: Path) -> None:
    graph = _quantized_graph()
    value = metadata(graph)
    missing = replace(value.quantized_targets[0], fqn="missing.linear")
    set_metadata(
        graph,
        replace(
            value,
            quantized_targets=(missing, *value.quantized_targets[1:]),
        ),
    )
    target = tmp_path / "partial.onnx"
    target.write_bytes(b"keep-existing")

    with pytest.raises(OnnxExportError, match="Cannot locate ONNX weight"):
        onnx_export(graph, target, mask_mode="masked", overwrite=True)

    assert target.read_bytes() == b"keep-existing"
    assert tuple(tmp_path.iterdir()) == (target,)
