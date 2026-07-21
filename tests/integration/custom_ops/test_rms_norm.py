from __future__ import annotations

import onnx
import pytest
import torch

from mdc_llm_deploy.custom_ops import create_onnx_export_profile
from mdc_llm_deploy.custom_ops.rms_norm import cpu, cuda, rms_norm


class _RmsNormModule(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rms_norm(x, gamma, 1e-5)


def _reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimensions = tuple(range(x.ndim - gamma.ndim, x.ndim))
    rstd = torch.rsqrt(x.float().square().mean(dim=dimensions) + epsilon)
    shape = (*rstd.shape, *((1,) * gamma.ndim))
    y = (x.float() * rstd.reshape(shape) * gamma.float()).to(x.dtype)
    return y, rstd


@pytest.mark.integration
def test_registered_operator_supports_fullgraph_compile_and_export() -> None:
    module = _RmsNormModule()
    x = torch.randn(2, 3, 4)
    gamma = torch.randn(3, 4)
    expected = _reference(x, gamma)

    compiled_output = torch.compile(module, backend="eager", fullgraph=True)(x, gamma)
    exported_output = torch.export.export(module, (x, gamma), strict=True).module()(x, gamma)

    torch.testing.assert_close(compiled_output, expected)
    torch.testing.assert_close(exported_output, expected)


@pytest.mark.integration
def test_dynamo_onnx_export_emits_exact_npu_rms_norm_abi() -> None:
    profile = create_onnx_export_profile("rms_norm")
    program = torch.onnx.export(
        _RmsNormModule().eval(),
        (torch.randn(2, 3, 16), torch.randn(16)),
        f=None,
        input_names=["x", "gamma"],
        output_names=["y", "rstd"],
        opset_version=18,
        dynamo=True,
        verbose=False,
        external_data=False,
        optimize=False,
        custom_translation_table=dict(profile.custom_translation_table),
    )
    assert program is not None
    model = program.model_proto

    nodes = [node for node in model.graph.node if node.op_type == "NPURmsNorm"]
    assert len(nodes) == 1
    node = nodes[0]
    assert node.domain == ""
    assert list(node.input) == ["x", "gamma"]
    assert list(node.output) == ["y", "rstd"]
    assert [(attribute.name, attribute.f) for attribute in node.attribute] == [
        ("epsilon", pytest.approx(1e-5))
    ]
    assert [(item.domain, item.version) for item in model.opset_import] == [("", 18)]
    onnx.checker.check_model(model, full_check=True)


@pytest.mark.integration
def test_dynamo_onnx_export_rejects_torch_legal_dynamic_shape() -> None:
    profile = create_onnx_export_profile("rms_norm")
    sequence = torch.export.Dim("sequence", min=1)

    with pytest.raises(Exception, match="static input shapes"):
        torch.onnx.export(
            _RmsNormModule().eval(),
            (torch.randn(2, 3, 16), torch.randn(16)),
            f=None,
            opset_version=18,
            dynamo=True,
            verbose=False,
            dynamic_shapes=({1: sequence}, None),
            custom_translation_table=dict(profile.custom_translation_table),
        )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_triton_matches_fp32_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    x = torch.randn(4, 3, 256, device="cuda", dtype=dtype)
    gamma = torch.randn(256, device="cuda", dtype=dtype)
    expected_y, expected_rstd = _reference(x, gamma)

    actual_y, actual_rstd = cuda(x, gamma, 1e-5)
    direct_y, direct_rstd = cpu(x.cpu(), gamma.cpu(), 1e-5)

    tolerance = 2e-2 if dtype in {torch.float16, torch.bfloat16} else 2e-5
    torch.testing.assert_close(actual_y, expected_y, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_rstd, expected_rstd, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(direct_y, expected_y.cpu())
    torch.testing.assert_close(direct_rstd, expected_rstd.cpu())
