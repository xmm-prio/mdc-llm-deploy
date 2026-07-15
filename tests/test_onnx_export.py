"""Tests for standard-intermediate and MDC-dialect ONNX export."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import onnx
import pytest
import torch

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.export import export
from mdc_llm_deploy.graph import GraphStage, QuantizedTarget, metadata, set_metadata
from mdc_llm_deploy.models.tiny import TinyQwen3Dense
from mdc_llm_deploy.onnx_export import onnx_export
from mdc_llm_deploy.onnx_export.validator import validate_mdc_model


def _graph() -> torch.fx.GraphModule:
    inputs = {"input_ids": torch.arange(8).reshape(1, 8)}
    return export(TinyQwen3Dense().eval(), inputs)


@pytest.fixture
def exported_pair(tmp_path: Path) -> tuple[onnx.ModelProto, onnx.ModelProto]:
    graph = _graph()
    masked = onnx_export(graph, tmp_path / "masked.onnx", mask_mode="masked")
    maskless = onnx_export(graph, tmp_path / "maskless.onnx", mask_mode="maskless")
    return masked, maskless


def test_mdc_onnx_has_real_outputs_and_complete_operator_abi(
    exported_pair: tuple[onnx.ModelProto, onnx.ModelProto],
) -> None:
    masked, _ = exported_pair
    initializer_names = {item.name for item in masked.graph.initializer}
    producers = {
        output: node.op_type for node in masked.graph.node for output in node.output
    }

    assert not initializer_names.intersection(item.name for item in masked.graph.output)
    assert all(producers[item.name] not in {"Constant", "ConstantOfShape"} for item in masked.graph.output)
    assert {node.op_type for node in masked.graph.node} >= {
        "NPURmsNorm",
        "ApplyRoPE",
        "FusedInferAttentionScore",
    }
    attention = next(
        node for node in masked.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    assert len(attention.input) == 29
    assert attention.input[4]
    assert {item.domain: item.version for item in masked.opset_import}[""] == 18


def test_maskless_rewrites_standard_spine_to_non_causal(
    exported_pair: tuple[onnx.ModelProto, onnx.ModelProto],
) -> None:
    masked, maskless = exported_pair

    def softmax_inputs(model: onnx.ModelProto) -> tuple[str, ...]:
        return tuple(
            node.input[0] for node in model.graph.node if node.op_type == "Softmax"
        )

    assert softmax_inputs(masked) != softmax_inputs(maskless)
    attention = next(
        node for node in maskless.graph.node if node.op_type == "FusedInferAttentionScore"
    )
    assert not attention.input[4]
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
