from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import onnx
import pytest
import torch
from torch import Tensor, nn
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import PretrainedConfig, PreTrainedModel
from transformers.exporters import OnnxConfig, OnnxExporter

from mdc_llm_deploy.quantization.qdq import (
    qdq,
    register_qdq_operator,
    require_supported_torch_version,
)

_REPOSITORY_ROOT = Path(__file__).parents[4]


class _LinearQDQ(nn.Module):
    def __init__(self, *, asymmetric: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=False)
        self.register_buffer("scale", torch.tensor(0.125))
        if asymmetric:
            self.register_buffer("zero_point", torch.tensor(-3, dtype=torch.int8))
        else:
            self.zero_point: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(qdq(inputs, self.scale, self.zero_point))


class _TransformersLinearQDQ(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self) -> None:
        super().__init__(PretrainedConfig())
        self.linear = nn.Linear(3, 2, bias=False)
        self.register_buffer("scale", torch.tensor(0.125))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(qdq(inputs, self.scale))


def _run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _nodes(model: onnx.ModelProto, op_type: str) -> list[onnx.NodeProto]:
    return [node for node in model.graph.node if node.op_type == op_type]


def _assert_standard_symmetric_qdq(model: onnx.ModelProto) -> None:
    quantize = _nodes(model, "QuantizeLinear")
    dequantize = _nodes(model, "DequantizeLinear")
    assert len(quantize) == 1
    assert len(dequantize) == 1
    assert len(quantize[0].input) == 2 or quantize[0].input[2] == ""
    assert len(dequantize[0].input) == 2 or dequantize[0].input[2] == ""
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in quantize[0].attribute
    }
    assert attributes["output_dtype"] == onnx.TensorProto.INT8
    assert model.opset_import[0].version == 21
    onnx.checker.check_model(model)


def test_import_has_no_operator_registration_side_effect() -> None:
    result = _run_isolated(
        "import torch\n"
        "import mdc_llm_deploy.quantization.qdq\n"
        "names = torch._C._dispatch_get_all_op_names()\n"
        "assert 'mdc_llm_deploy::qdq' not in names\n"
    )

    assert result.returncode == 0, result.stderr


def test_eager_symmetric_qdq_matches_int8_formula() -> None:
    inputs = torch.tensor([[-20.0, -0.18, 0.31, 20.0]])
    scale = torch.tensor(0.1)

    output = qdq(inputs, scale)

    expected = torch.round(inputs / scale).clamp(-128, 127) * scale
    torch.testing.assert_close(output, expected)


def test_eager_asymmetric_per_channel_qdq_broadcasts_axis() -> None:
    inputs = torch.tensor([[[-1.0, 0.2], [0.6, 1.4], [2.0, -0.2]]])
    scale = torch.tensor([0.1, 0.2, 0.5])
    zero_point = torch.tensor([-3, 2, 1], dtype=torch.int8)

    output = qdq(inputs, scale, zero_point, axis=1)

    shaped_scale = scale.reshape(1, 3, 1)
    shaped_zero_point = zero_point.to(torch.float32).reshape(1, 3, 1)
    quantized = torch.round(inputs / shaped_scale + shaped_zero_point).clamp(-128, 127)
    expected = (quantized - shaped_zero_point) * shaped_scale
    torch.testing.assert_close(output, expected)


def test_fake_and_meta_kernels_preserve_tensor_metadata() -> None:
    inputs = torch.randn(2, 3)
    scale = torch.tensor(0.1)
    mode = FakeTensorMode()

    fake_output = qdq(mode.from_tensor(inputs), mode.from_tensor(scale))
    meta_output = qdq(
        torch.empty((2, 3), dtype=torch.float16, device="meta"),
        torch.empty((), dtype=torch.float16, device="meta"),
    )

    assert mode.is_our_fake(fake_output)
    assert fake_output.shape == inputs.shape
    assert fake_output.dtype == inputs.dtype
    assert meta_output.shape == inputs.shape
    assert meta_output.dtype is torch.float16
    assert meta_output.device.type == "meta"


def test_version_gate_uses_distribution_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_version(distribution_name: str) -> str:
        calls.append(distribution_name)
        return "2.12.0+build"

    monkeypatch.setattr(
        "mdc_llm_deploy.quantization.qdq._registration.metadata.version",
        fake_version,
    )

    with pytest.raises(RuntimeError, match=r"requires torch==2\.12\.0"):
        require_supported_torch_version()

    assert calls == ["torch"]


def test_lazy_registration_is_thread_safe_and_idempotent() -> None:
    result = _run_isolated(
        "from concurrent.futures import ThreadPoolExecutor\n"
        "from mdc_llm_deploy.quantization.qdq import register_qdq_operator\n"
        "with ThreadPoolExecutor(max_workers=16) as executor:\n"
        "    operators = list(executor.map(lambda _: register_qdq_operator(), range(64)))\n"
        "assert all(operator is operators[0] for operator in operators)\n"
    )

    assert result.returncode == 0, result.stderr


def test_existing_operator_name_conflict_fails_strictly() -> None:
    result = _run_isolated(
        "import torch\n"
        "@torch.library.custom_op('mdc_llm_deploy::qdq', mutates_args=())\n"
        "def occupied(inputs: torch.Tensor, scale: torch.Tensor, "
        "zero_point: torch.Tensor | None, axis: int | None) -> torch.Tensor:\n"
        "    return inputs\n"
        "from mdc_llm_deploy.quantization.qdq import register_qdq_operator\n"
        "register_qdq_operator()\n"
    )

    assert result.returncode != 0
    assert "already" in result.stderr.lower()


@pytest.mark.parametrize("asymmetric", [False, True])
def test_dynamo_export_emits_opset21_standard_qdq(asymmetric: bool) -> None:
    model = _LinearQDQ(asymmetric=asymmetric).eval()

    program = torch.onnx.export(
        model,
        (torch.randn(1, 3),),
        dynamo=True,
        opset_version=21,
        external_data=False,
        optimize=False,
    )

    assert program is not None
    if asymmetric:
        quantize = _nodes(program.model_proto, "QuantizeLinear")
        dequantize = _nodes(program.model_proto, "DequantizeLinear")
        assert len(quantize) == 1
        assert len(dequantize) == 1
        assert len(quantize[0].input) == 3
        assert len(dequantize[0].input) == 3
        zero_point = next(
            initializer
            for initializer in program.model_proto.graph.initializer
            if initializer.name == quantize[0].input[2]
        )
        assert zero_point.data_type == onnx.TensorProto.INT8
        onnx.checker.check_model(program.model_proto)
    else:
        _assert_standard_symmetric_qdq(program.model_proto)


def test_transformers_exporter_emits_opset21_standard_qdq() -> None:
    model = _TransformersLinearQDQ().eval()

    program = OnnxExporter().export(
        model,
        {"inputs": torch.randn(1, 3)},
        OnnxConfig(
            opset_version=21,
            external_data=False,
            optimize=False,
        ),
    )

    _assert_standard_symmetric_qdq(program.model_proto)


def test_register_returns_same_operator_in_current_process() -> None:
    with ThreadPoolExecutor(max_workers=8) as executor:
        operators = list(executor.map(lambda _: register_qdq_operator(), range(32)))

    assert all(operator is operators[0] for operator in operators)
