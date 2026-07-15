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
from mdc_llm_deploy.onnx_export.api import _prune_unreachable
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

    _prune_unreachable(model)
    first = model.SerializeToString()
    _prune_unreachable(model)

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

    _prune_unreachable(model)

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
        match="GPTQ and W4 graphs do not support ONNX export",
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
    config = TinyConfig(
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        moe_intermediate_size=128,
    )
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    graph = export(TinyQwen3Moe(config).eval(), inputs)
    oneshot(graph, "configs/minmax-moe-w8a8.json", [inputs])

    model = onnx_export(graph, tmp_path / "moe.onnx", mask_mode="masked")

    moe = next(node for node in model.graph.node if node.op_type == "MoeExpert")
    reachable = _values_reaching_outputs(model)
    initializers = {item.name: item for item in model.graph.initializer}
    properties = {item.key: item.value for item in model.metadata_props}
    assert moe.output[0] in reachable
    assert tuple(initializers[moe.input[3]].dims) == (5 * 3 * 256 * 128,)
    assert tuple(initializers[moe.input[4]].dims) == (21,)
    assert tuple(initializers[moe.input[5]].dims) == (21,)
    assert properties["mdc.moe.hidden_size"] == "256"
    assert properties["mdc.moe.intermediate_size"] == "128"
    assert len(properties["mdc.moe.weight_offsets"].split(",")) == 15
    assert len(properties["mdc.moe.weight_lengths"].split(",")) == 15
    assert not any("experts." in name for name in initializers)
    assert not any("shared_expert" in name for name in initializers)


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
