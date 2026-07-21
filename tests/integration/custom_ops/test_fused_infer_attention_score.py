from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import torch

from mdc_llm_deploy.custom_ops.fused_infer_attention_score import (
    ONNX_ATTRIBUTE_NAMES,
    PLUGIN_NAME,
    attention_kernel,
    fused_infer_attention_score,
)
from mdc_llm_deploy.custom_ops.registry import create_onnx_export_profile


class _AttentionModule(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=4,
            scale=0.5,
            input_layout="BNSD",
            num_key_value_heads=2,
        )


class _MaskedAttentionModule(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_infer_attention_score(
            query,
            key,
            value,
            atten_mask=mask,
            num_heads=4,
            scale=0.5,
            input_layout="BNSD",
            num_key_value_heads=2,
        )


def _decode_inputs(
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(1, 4, 1, 8, dtype=dtype),
        torch.randn(1, 2, 5, 8, dtype=dtype),
        torch.randn(1, 2, 5, 8, dtype=dtype),
    )


@pytest.mark.integration
def test_registered_operator_runs_through_fullgraph_compile_and_export() -> None:
    module = _AttentionModule()
    inputs = _decode_inputs(torch.float32)
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    expected = module(*inputs)
    actual = compiled(*inputs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

    exported = torch.export.export(module, inputs, strict=True)
    exported_actual = exported.module()(*inputs)
    torch.testing.assert_close(exported_actual[0], expected[0])
    torch.testing.assert_close(exported_actual[1], expected[1])


@pytest.mark.integration
def test_dynamo_onnx_emits_complete_fia_abi_and_constant_lse(
    tmp_path: Path,
) -> None:
    profile = create_onnx_export_profile(PLUGIN_NAME)
    output_path = tmp_path / "attention.onnx"
    torch.onnx.export(
        _AttentionModule(),
        _decode_inputs(),
        output_path,
        opset_version=18,
        dynamo=True,
        verbose=False,
        custom_translation_table=profile.custom_translation_table,
        input_names=["query", "key", "value"],
        output_names=["attention_out", "softmax_lse"],
    )
    model = onnx.load(output_path)
    onnx.checker.check_model(model, full_check=True)
    nodes = [node for node in model.graph.node if node.op_type == "FusedInferAttentionScore"]

    assert len(nodes) == 1
    node = nodes[0]
    assert node.domain == ""
    assert list(node.input) == ["query", "key", "value"]
    assert len(node.output) == 2
    assert any(
        producer.op_type in {"Cast", "CastLike"}
        and producer.input[0] == node.output[0]
        and model.graph.output[0].name in producer.output
        for producer in model.graph.node
    )
    assert {attribute.name for attribute in node.attribute} == ONNX_ATTRIBUTE_NAMES
    lse_output = model.graph.output[1].name
    assert any(
        producer.domain == ""
        and producer.op_type == "Constant"
        and lse_output in producer.output
        for producer in model.graph.node
    )
    assert not any(node.domain == "mdc_llm_deploy" for node in model.graph.node)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("module", "inputs", "message"),
    [
        (
            _AttentionModule(),
            (
                torch.randn(1, 4, 3, 8, dtype=torch.float16),
                torch.randn(1, 2, 5, 8, dtype=torch.float16),
                torch.randn(1, 2, 5, 8, dtype=torch.float16),
            ),
            "query sequence length S=1",
        ),
        (
            _MaskedAttentionModule(),
            (
                *_decode_inputs(),
                torch.zeros(1, 1, 1, 5, dtype=torch.bool),
            ),
            "optional inputs: atten_mask",
        ),
    ],
)
def test_dynamo_onnx_rejects_torch_legal_inputs_outside_decode_contract(
    tmp_path: Path,
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    message: str,
) -> None:
    profile = create_onnx_export_profile(PLUGIN_NAME)
    with pytest.raises(Exception, match=message):
        torch.onnx.export(
            module,
            inputs,
            tmp_path / "invalid.onnx",
            opset_version=18,
            dynamo=True,
            verbose=False,
            custom_translation_table=profile.custom_translation_table,
        )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_kernel_matches_cpu() -> None:
    cpu_inputs = _decode_inputs()
    expected = attention_kernel(
        *cpu_inputs,
        num_heads=4,
        num_key_value_heads=2,
        scale=0.5,
    )
    actual = fused_infer_attention_score(
        *(tensor.cuda() for tensor in cpu_inputs),
        num_heads=4,
        num_key_value_heads=2,
        scale=0.5,
    )
    torch.testing.assert_close(actual[0].cpu(), expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1].cpu(), expected[1])
