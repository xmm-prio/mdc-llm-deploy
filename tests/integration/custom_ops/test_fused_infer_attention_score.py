from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import torch

from mdc_llm_deploy.custom_ops.fused_infer_attention_score import (
    FusedInferAttentionScore,
)
from mdc_llm_deploy.custom_ops.registry import register_custom_op


class _AttentionModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._operator = register_custom_op(FusedInferAttentionScore).definition

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._operator(
            query,
            key,
            value,
            num_heads=4,
            scale=0.5,
            input_layout="BNSD",
            num_key_value_heads=2,
        )


@pytest.mark.integration
def test_registered_operator_runs_through_torch_compile() -> None:
    module = _AttentionModule()
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    query = torch.randn(1, 4, 2, 4, dtype=torch.float16)
    key = torch.randn(1, 2, 3, 4, dtype=torch.float16)
    value = torch.randn(1, 2, 3, 4, dtype=torch.float16)

    actual = compiled(query, key, value)
    expected = module(query, key, value)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.integration
def test_registered_operator_runs_through_torch_export() -> None:
    module = _AttentionModule()
    query = torch.randn(1, 4, 2, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)

    exported = torch.export.export(module, (query, key, value), strict=True)
    actual = exported.module()(query, key, value)
    expected = module(query, key, value)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.integration
def test_legacy_onnx_export_preserves_29_slots_and_all_attributes(
    tmp_path: Path,
) -> None:
    module = _AttentionModule()
    query = torch.randn(1, 4, 2, 4, dtype=torch.float16)
    key = torch.randn(1, 2, 3, 4, dtype=torch.float16)
    value = torch.randn(1, 2, 3, 4, dtype=torch.float16)
    output_path = tmp_path / "attention.onnx"

    torch.onnx.export(
        module,
        (query, key, value),
        output_path,
        opset_version=18,
        dynamo=False,
        input_names=["query", "key", "value"],
        output_names=["attention_out", "softmax_lse"],
    )
    model = onnx.load(output_path)
    nodes = [node for node in model.graph.node if node.op_type == "FusedInferAttentionScore"]

    assert len(nodes) == 1
    node = nodes[0]
    assert node.domain == ""
    assert len(node.input) == 29
    assert node.input[:3] == ["query", "key", "value"]
    assert list(node.input[3:]) == [""] * 26
    assert {attribute.name for attribute in node.attribute} == set(
        FusedInferAttentionScore.onnx_attribute_defaults
    )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_triton_matches_fp32_cpu_reference() -> None:
    torch.manual_seed(11)
    query_cpu = torch.randn(2, 4, 3, 16, dtype=torch.float16)
    key_cpu = torch.randn(2, 2, 5, 16, dtype=torch.float16)
    value_cpu = torch.randn(2, 2, 5, 16, dtype=torch.float16)
    mask_cpu = torch.zeros(2, 1, 3, 5, dtype=torch.bool)
    mask_cpu[:, :, :, -1] = True
    query_lengths_cpu = torch.tensor([3, 2], dtype=torch.int64)
    key_lengths_cpu = torch.tensor([5, 4], dtype=torch.int64)
    kwargs = {
        "num_heads": 4,
        "num_key_value_heads": 2,
        "scale": 0.25,
        "input_layout": "BNSD",
        "softmax_lse_flag": True,
    }

    expected = FusedInferAttentionScore.cpu(
        query_cpu,
        key_cpu,
        value_cpu,
        atten_mask=mask_cpu,
        actual_seq_lengths=query_lengths_cpu,
        actual_seq_lengths_kv=key_lengths_cpu,
        **kwargs,
    )
    actual = FusedInferAttentionScore.cuda(
        query_cpu.cuda(),
        key_cpu.cuda(),
        value_cpu.cuda(),
        atten_mask=mask_cpu.cuda(),
        actual_seq_lengths=query_lengths_cpu.cuda(),
        actual_seq_lengths_kv=key_lengths_cpu.cuda(),
        **kwargs,
    )

    torch.testing.assert_close(actual[0].cpu(), expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1].cpu(), expected[1], atol=2e-2, rtol=2e-2)
