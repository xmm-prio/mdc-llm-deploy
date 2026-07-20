from __future__ import annotations

import io

import onnx
import pytest
import torch

from mdc_llm_deploy.custom_ops.moe_expert import MoeExpert
from mdc_llm_deploy.custom_ops.registry import register_custom_op


def _case(device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    torch.manual_seed(23)
    x = torch.randn(4, 8, device=device)
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 0], [1, 0]], dtype=torch.int64, device=device
    )
    topk_weight = torch.tensor(
        [[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.9, 0.1]], device=device
    )
    weights = torch.randn(3, 3 * 8 * 16, device=device)
    return x, topk_ids, topk_weight, weights


@pytest.mark.integration
def test_registered_operator_runs_and_compiles() -> None:
    definition = register_custom_op(MoeExpert).definition
    inputs = _case()
    expected = MoeExpert.cpu(*inputs)

    eager = definition(*inputs)
    compiled = torch.compile(lambda *args: definition(*args), fullgraph=True)(*inputs)

    torch.testing.assert_close(eager, expected)
    torch.testing.assert_close(compiled, expected)


@pytest.mark.integration
def test_registered_operator_exports_with_torch_export() -> None:
    definition = register_custom_op(MoeExpert).definition

    class Model(torch.nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            topk_ids: torch.Tensor,
            topk_weight: torch.Tensor,
            expert_weights: torch.Tensor,
        ) -> torch.Tensor:
            return definition(x, topk_ids, topk_weight, expert_weights)

    inputs = _case()
    exported = torch.export.export(Model(), inputs)

    torch.testing.assert_close(exported.module()(*inputs), MoeExpert.cpu(*inputs))


@pytest.mark.integration
def test_registered_operator_exports_with_optional_onnx_slots() -> None:
    definition = register_custom_op(MoeExpert).definition

    class Model(torch.nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            topk_ids: torch.Tensor,
            topk_weight: torch.Tensor,
            expert_weights: torch.Tensor,
        ) -> torch.Tensor:
            return definition(x, topk_ids, topk_weight, expert_weights)

    buffer = io.BytesIO()
    torch.onnx.export(
        Model(),
        _case(),
        buffer,
        opset_version=18,
        dynamo=False,
        input_names=["x", "topk_ids", "topk_weight", "expert_weights"],
        output_names=["out"],
    )
    model = onnx.load_model_from_string(buffer.getvalue())
    node = next(node for node in model.graph.node if node.op_type == "MoeExpert")

    assert node.domain == ""
    assert list(node.input[:4]) == ["x", "topk_ids", "topk_weight", "expert_weights"]
    assert list(node.input[4:]) == ["", ""]


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_triton_matches_cpu_fp32() -> None:
    cpu_inputs = _case()
    cuda_inputs = tuple(tensor.cuda() for tensor in cpu_inputs)

    actual = MoeExpert.cuda(*cuda_inputs).cpu()
    expected = MoeExpert.cpu(*cpu_inputs)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
