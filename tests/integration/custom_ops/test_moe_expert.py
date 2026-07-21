from __future__ import annotations

from typing import cast

import onnx
import pytest
import torch
from torch.onnx._internal.exporter._errors import ConversionError

from mdc_llm_deploy.custom_ops import create_onnx_export_profile
from mdc_llm_deploy.custom_ops.moe_expert import cpu, moe_expert


def _floating_case(device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    torch.manual_seed(23)
    return (
        torch.randn(4, 8, device=device),
        torch.tensor(
            [[0, 1], [1, 2], [2, 0], [1, 0]], dtype=torch.int64, device=device
        ),
        torch.tensor(
            [[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.9, 0.1]], device=device
        ),
        torch.randn(3, 3 * 8 * 16, device=device),
    )


def _mdc_case() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(31)
    return (
        torch.randint(-8, 8, (1, 256), dtype=torch.int8, generator=generator),
        torch.tensor([[1, 3]], dtype=torch.int16),
        torch.tensor([[0.375, 0.625]], dtype=torch.float16),
        torch.randint(
            -8,
            8,
            (3 * 4 * 128, 256),
            dtype=torch.int8,
            generator=generator,
        ),
        torch.tensor(
            [
                0.025,
                0.020,
                0.018,
                0.0050,
                0.021,
                0.017,
                0.023,
                0.0045,
                0.019,
                0.022,
                0.016,
                0.0055,
                0.024,
                0.019,
                0.021,
                0.0040,
                0.018,
            ],
            dtype=torch.float32,
        ),
    )


class _MoeExpertModel(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
        quant_scales: torch.Tensor | None = None,
        quant_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            moe_expert(
                x,
                topk_ids,
                topk_weight,
                expert_weights,
                quant_scales,
                quant_offsets,
            ),
        )


@pytest.mark.integration
def test_registered_operator_runs_fake_compile_and_torch_export() -> None:
    model = _MoeExpertModel()
    inputs = _floating_case()
    expected = cpu(*inputs)

    eager = model(*inputs)
    compiled = torch.compile(model, backend="eager", fullgraph=True)(*inputs)
    exported = torch.export.export(model, inputs).module()(*inputs)

    torch.testing.assert_close(eager, expected)
    torch.testing.assert_close(compiled, expected)
    torch.testing.assert_close(exported, expected)


@pytest.mark.integration
def test_dynamo_onnx_emits_five_input_default_domain_abi() -> None:
    profile = create_onnx_export_profile("moe_expert")
    inputs = _mdc_case()

    program = torch.onnx.export(
        _MoeExpertModel(),
        inputs,
        dynamo=True,
        verbose=False,
        opset_version=18,
        input_names=[
            "x",
            "topk_ids",
            "topk_weight",
            "expert_weights",
            "quant_scales",
        ],
        output_names=["out"],
        custom_translation_table=dict(profile.custom_translation_table),
    )
    assert program is not None
    model = program.model_proto
    onnx.checker.check_model(model, full_check=True)

    nodes = [node for node in model.graph.node if node.op_type == "MoeExpert"]
    assert len(nodes) == 1
    node = nodes[0]
    assert node.domain == ""
    assert list(node.input) == [
        "x",
        "topk_ids",
        "topk_weight",
        "expert_weights",
        "quant_scales",
    ]
    assert list(node.output) == ["out"]
    assert not node.attribute
    assert [(item.domain, item.version) for item in model.opset_import] == [("", 18)]


@pytest.mark.integration
def test_dynamo_onnx_rejects_torch_legal_floating_contract() -> None:
    profile = create_onnx_export_profile("moe_expert")

    with pytest.raises(ConversionError, match="x must be INT8"):
        torch.onnx.export(
            _MoeExpertModel(),
            _floating_case(),
            dynamo=True,
            verbose=False,
            opset_version=18,
            custom_translation_table=dict(profile.custom_translation_table),
        )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_fp32() -> None:
    cpu_inputs = _floating_case()
    cuda_inputs = tuple(tensor.cuda() for tensor in cpu_inputs)

    actual = _MoeExpertModel()(*cuda_inputs).cpu()
    expected = cpu(*cpu_inputs)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
