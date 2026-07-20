from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import torch

from mdc_llm_deploy.custom_ops.apply_rotary_pos_emb import ApplyRotaryPosEmb
from mdc_llm_deploy.custom_ops.registry import register_custom_op


class RotaryModule(torch.nn.Module):
    def __init__(self, layout: int = 1, rotary_mode: str = "half") -> None:
        super().__init__()
        self.layout = layout
        self.rotary_mode = rotary_mode
        self.definition = register_custom_op(ApplyRotaryPosEmb).definition

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.definition(
            query,
            key,
            cos,
            sin,
            self.layout,
            self.rotary_mode,
        )


def _inputs(
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    query = torch.randn(2, 3, 4, 8, generator=generator, device=device)
    key = torch.randn(2, 3, 2, 8, generator=generator, device=device)
    angles = torch.randn(1, 3, 1, 8, generator=generator, device=device)
    return query, key, angles.cos(), angles.sin()


@pytest.mark.integration
def test_custom_op_runs_through_compile_and_export() -> None:
    module = RotaryModule(rotary_mode="interleave")
    inputs = _inputs()
    expected = module(*inputs)

    compiled = torch.compile(module, backend="eager")
    actual = compiled(*inputs)
    exported = torch.export.export(module, inputs)
    exported_actual = exported.module()(*inputs)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(exported_actual[0], expected[0])
    torch.testing.assert_close(exported_actual[1], expected[1])


@pytest.mark.integration
def test_legacy_onnx_export_emits_expected_opset18_node(tmp_path: Path) -> None:
    module = RotaryModule(layout=1, rotary_mode="quarter")
    output_path = tmp_path / "apply_rotary_pos_emb.onnx"

    torch.onnx.export(
        module,
        _inputs(),
        output_path,
        opset_version=18,
        dynamo=False,
        input_names=["query", "key", "cos", "sin"],
        output_names=["query_out", "key_out"],
    )

    model = onnx.load(output_path)
    nodes = [node for node in model.graph.node if node.op_type == "ApplyRotaryPosEmb"]
    assert len(nodes) == 1
    assert nodes[0].domain == ""
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in nodes[0].attribute
    }
    assert attributes == {"layout": 1, "rotary_mode": b"quarter"}
    assert model.opset_import[0].version == 18


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("mode", ["half", "interleave", "quarter"])
def test_triton_cuda_matches_fp32_cpu(mode: str) -> None:
    cpu_inputs = _inputs()
    cuda_inputs = tuple(tensor.cuda() for tensor in cpu_inputs)

    expected = ApplyRotaryPosEmb.cpu(*cpu_inputs, 1, mode)
    actual = ApplyRotaryPosEmb.cuda(*cuda_inputs, 1, mode)

    torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=1e-5, atol=1e-5)


@pytest.mark.integration
def test_cuda_never_silently_falls_back_without_cuda() -> None:
    inputs = _inputs()

    with pytest.raises((RuntimeError, AssertionError), match=r"CUDA|cuda"):
        ApplyRotaryPosEmb.cuda(*inputs)
