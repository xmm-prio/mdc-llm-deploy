"""Tests for standard-intermediate and MDC-dialect ONNX export."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import onnx
import pytest
import torch
from onnx import TensorProto, helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.export import export
from mdc_llm_deploy.graph import GraphStage, QuantizedTarget, metadata, set_metadata
from mdc_llm_deploy.models.tiny import TinyQwen3Dense
from mdc_llm_deploy.onnx_export import onnx_export
from mdc_llm_deploy.onnx_export.validator import validate_mdc_model
from mdc_llm_deploy.quantization import oneshot


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
            in {"NPURmsNorm", "ApplyRoPE", "FusedInferAttentionScore"}
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
            "ApplyRoPE",
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
    rope = next(node for node in masked.graph.node if node.op_type == "ApplyRoPE")
    masked.graph.node.append(
        helper.make_node(
            "ApplyRoPE",
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
