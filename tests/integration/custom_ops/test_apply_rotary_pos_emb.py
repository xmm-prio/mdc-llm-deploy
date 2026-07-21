from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import torch

from mdc_llm_deploy.custom_ops.apply_rotary_pos_emb import (
    apply_rotary_pos_emb,
    cpu,
    cuda,
)
from mdc_llm_deploy.custom_ops.registry import create_onnx_export_profile


class RotaryModule(torch.nn.Module):
    def __init__(self, layout: int = 1, rotary_mode: str = "half") -> None:
        super().__init__()
        self.layout = layout
        self.rotary_mode = rotary_mode

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_pos_emb(
            query,
            key,
            cos,
            sin,
            self.layout,
            self.rotary_mode,
        )


def _inputs(
    *,
    head_dim: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    query = torch.randn(2, 3, 4, head_dim, generator=generator)
    key = torch.randn(2, 3, 2, head_dim, generator=generator)
    angles = torch.randn(1, 3, 1, head_dim, generator=generator)
    return query, key, angles.cos(), angles.sin()


@pytest.mark.integration
def test_custom_op_runs_through_fullgraph_compile_and_torch_export() -> None:
    module = RotaryModule(rotary_mode="interleave")
    inputs = _inputs()
    expected = module(*inputs)

    compiled = torch.compile(module, backend="eager", fullgraph=True)
    actual = compiled(*inputs)
    exported_actual = torch.export.export(module, inputs).module()(*inputs)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(exported_actual[0], expected[0])
    torch.testing.assert_close(exported_actual[1], expected[1])


@pytest.mark.integration
def test_dynamo_onnx_export_emits_exact_default_domain_abi(tmp_path: Path) -> None:
    module = RotaryModule(layout=1, rotary_mode="quarter").eval()
    profile = create_onnx_export_profile("apply_rotary_pos_emb")
    output_path = tmp_path / "apply_rotary_pos_emb.onnx"

    torch.onnx.export(
        module,
        _inputs(),
        output_path,
        opset_version=18,
        dynamo=True,
        verbose=False,
        custom_translation_table=dict(profile.custom_translation_table),
        input_names=["query", "key", "cos", "sin"],
        output_names=["query_out", "key_out"],
        optimize=False,
    )

    model = onnx.load(output_path)
    onnx.checker.check_model(model, full_check=True)
    nodes = [node for node in model.graph.node if node.op_type == "ApplyRotaryPosEmb"]
    assert len(nodes) == 1
    node = nodes[0]
    assert node.domain == ""
    assert list(node.input) == ["query", "key", "cos", "sin"]
    assert list(node.output) == ["query_out", "key_out"]
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }
    assert attributes == {"layout": 1, "rotary_mode": b"quarter"}
    assert [(opset.domain, opset.version) for opset in model.opset_import] == [("", 18)]


@pytest.mark.integration
def test_torch_valid_large_head_dimension_is_rejected_only_by_onnx(
    tmp_path: Path,
) -> None:
    module = RotaryModule().eval()
    inputs = _inputs(head_dim=1026)
    module(*inputs)
    torch.export.export(module, inputs)
    profile = create_onnx_export_profile("apply_rotary_pos_emb")

    with pytest.raises(Exception, match="head dimension <= 1024"):
        torch.onnx.export(
            module,
            inputs,
            tmp_path / "unsupported.onnx",
            opset_version=18,
            dynamo=True,
            verbose=False,
            custom_translation_table=dict(profile.custom_translation_table),
            optimize=False,
        )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("mode", ["half", "interleave", "quarter"])
def test_triton_cuda_matches_fp32_cpu(mode: str) -> None:
    cpu_inputs = _inputs()
    cuda_inputs = tuple(tensor.cuda() for tensor in cpu_inputs)

    expected = cpu(*cpu_inputs, 1, mode)
    actual = cuda(*cuda_inputs, 1, mode)

    torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=1e-5, atol=1e-5)


@pytest.mark.integration
def test_cuda_never_silently_falls_back_without_cuda() -> None:
    with pytest.raises((RuntimeError, AssertionError), match=r"CUDA|cuda"):
        cuda(*_inputs())
