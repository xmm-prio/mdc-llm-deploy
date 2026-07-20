from __future__ import annotations

import io

import onnx
import pytest
import torch

from mdc_llm_deploy.custom_ops.registry import register_custom_op
from mdc_llm_deploy.custom_ops.rms_norm import RmsNorm

_RMS_NORM = register_custom_op(RmsNorm).definition


class _RmsNormModule(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _RMS_NORM(x, gamma, 1e-5)


def _reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    rstd = torch.rsqrt(x.float().square().mean(dim=-1) + epsilon)
    y = (x.float() * rstd.unsqueeze(-1) * gamma.float()).to(x.dtype)
    return y, rstd


@pytest.mark.integration
def test_registered_operator_supports_compile_and_export() -> None:
    module = _RmsNormModule()
    x = torch.randn(2, 3, 16)
    gamma = torch.randn(16)
    expected = _reference(x, gamma)

    compiled = torch.compile(module, backend="eager")
    compiled_output = compiled(x, gamma)
    exported = torch.export.export(module, (x, gamma))
    exported_output = exported.module()(x, gamma)

    torch.testing.assert_close(compiled_output, expected)
    torch.testing.assert_close(exported_output, expected)


@pytest.mark.integration
def test_legacy_onnx_export_emits_npu_rms_norm_abi() -> None:
    module = _RmsNormModule()
    x = torch.randn(2, 3, 16)
    gamma = torch.randn(16)
    buffer = io.BytesIO()

    torch.onnx.export(
        module,
        (x, gamma),
        buffer,
        input_names=["x", "gamma"],
        output_names=["y", "rstd"],
        opset_version=18,
        dynamo=False,
    )
    model = onnx.load_model_from_string(buffer.getvalue())

    nodes = [node for node in model.graph.node if node.op_type == "NPURmsNorm"]
    assert len(nodes) == 1
    node = nodes[0]
    assert node.domain == ""
    assert list(node.input) == ["x", "gamma"]
    assert list(node.output) == ["y", "rstd"]
    epsilon = next(attribute for attribute in node.attribute if attribute.name == "epsilon")
    assert epsilon.f == pytest.approx(1e-5)
    assert model.opset_import[0].version == 18


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_triton_matches_fp32_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    x = torch.randn(4, 3, 256, device="cuda", dtype=dtype)
    gamma = torch.randn(256, device="cuda", dtype=dtype)
    expected_y, expected_rstd = _reference(x, gamma)

    actual_y, actual_rstd = RmsNorm.cuda(x, gamma, 1e-5)

    tolerance = 2e-2 if dtype in {torch.float16, torch.bfloat16} else 2e-5
    torch.testing.assert_close(actual_y, expected_y, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_rstd, expected_rstd, atol=2e-5, rtol=2e-5)
