from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from mdc_llm_deploy.custom_ops.moe_expert import (
    PLUGIN,
    cpu,
    fake,
    moe_expert,
)
from mdc_llm_deploy.custom_ops.moe_expert.onnx import (
    translate,
    validate_onnx_contract,
)


def _pack(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [gate.flatten(start_dim=1), up.flatten(start_dim=1), down.flatten(start_dim=1)],
        dim=1,
    )


def _reference(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        for route in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, route])
            hidden = torch.nn.functional.silu(gate[expert] @ x[token].float())
            hidden *= up[expert] @ x[token].float()
            output[token] += topk_weight[token, route].float() * (
                down[expert] @ hidden
            )
    return output.to(x.dtype)


def _floating_case() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    x = torch.randn(3, 4)
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int64)
    topk_weight = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.1, 0.9]])
    gate = torch.randn(3, 5, 4)
    up = torch.randn(3, 5, 4)
    down = torch.randn(3, 4, 5)
    return x, topk_ids, topk_weight, gate, up, down


def _mdc_case() -> tuple[torch.Tensor, ...]:
    return (
        torch.ones((1, 256), dtype=torch.int8),
        torch.tensor([[0]], dtype=torch.int16),
        torch.ones((1, 1), dtype=torch.float16),
        torch.ones((3 * 128, 256), dtype=torch.int8),
        torch.tensor([0.01, 0.02, 0.02, 0.001, 0.02], dtype=torch.float32),
    )


def test_cpu_float_matches_reference() -> None:
    x, topk_ids, topk_weight, gate, up, down = _floating_case()
    actual = cpu(x, topk_ids, topk_weight, _pack(gate, up, down))
    torch.testing.assert_close(
        actual,
        _reference(x, topk_ids, topk_weight, gate, up, down),
    )


@pytest.mark.parametrize("with_offsets", [False, True])
def test_cpu_conventional_int8_weight_contract(with_offsets: bool) -> None:
    torch.manual_seed(11)
    expert_count, hidden_size, intermediate_size = 2, 3, 4
    x = torch.randn(2, hidden_size)
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    routing = torch.tensor([[0.3, 0.7], [0.8, 0.2]])
    packed = torch.randint(
        -20,
        20,
        (expert_count, 3 * hidden_size * intermediate_size),
        dtype=torch.int8,
    )
    scales = torch.rand(expert_count, 2 * intermediate_size + hidden_size) + 0.01
    offsets = torch.randn_like(scales) if with_offsets else None

    output = cpu(x, ids, routing, packed, scales, offsets)

    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_mdc_cpu_and_fake_return_fp16() -> None:
    inputs = _mdc_case()
    output = cpu(*inputs)
    fake_output = fake(*(tensor.to("meta") for tensor in inputs))

    assert output.shape == (1, 256)
    assert output.dtype == torch.float16
    assert fake_output.shape == (1, 256)
    assert fake_output.dtype == torch.float16


def test_fake_preserves_floating_metadata() -> None:
    with FakeTensorMode() as mode:
        output = fake(
            mode.from_tensor(torch.empty(3, 4)),
            mode.from_tensor(torch.empty(3, 1, dtype=torch.int32)),
            mode.from_tensor(torch.empty(3, 1)),
            mode.from_tensor(torch.empty(2, 36)),
        )

    assert isinstance(output, FakeTensor)
    assert output.shape == (3, 4)
    assert output.dtype == torch.float32


def test_torch_schema_remains_broad_and_opcheck_passes() -> None:
    x, ids, routing, gate, up, down = _floating_case()
    inputs = (x, ids, routing, _pack(gate, up, down))

    torch.library.opcheck(
        cast(Any, moe_expert),
        inputs,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert "quant_offsets" in PLUGIN.torch.schema


class _OnnxValue:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype


def _onnx_inputs(x_dtype: str = "INT8") -> tuple[_OnnxValue, ...]:
    return (
        _OnnxValue((1, 256), x_dtype),
        _OnnxValue((1, 2), "INT16"),
        _OnnxValue((1, 2), "FLOAT16"),
        _OnnxValue((3 * 4 * 128, 256), "INT8"),
        _OnnxValue((17,), "FLOAT"),
    )


def test_onnx_contract_accepts_only_five_mdc_inputs() -> None:
    validate_onnx_contract(*_onnx_inputs())
    assert len(PLUGIN.onnx.schema.inputs) == 5
    assert [parameter.name for parameter in PLUGIN.onnx.schema.inputs] == [
        "x",
        "topk_ids",
        "topk_weight",
        "expert_weights",
        "quant_scales",
    ]

    with pytest.raises(TypeError):
        cast(Any, translate)(*_onnx_inputs(), None)


def test_onnx_contract_rejects_torch_legal_float_case() -> None:
    with pytest.raises(ValueError, match="x must be INT8"):
        validate_onnx_contract(*_onnx_inputs("FLOAT"))


def test_mdc_torch_contract_rejects_quant_offsets() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        cast(Any, cpu)(*_mdc_case(), torch.zeros(1, dtype=torch.float32))


@pytest.mark.parametrize(
    ("ids", "weights", "message"),
    [
        (torch.tensor([[0, 2]]), torch.tensor([[0.5, 0.5]]), "out-of-range"),
        (torch.tensor([[0, 0]]), torch.tensor([[0.5, 0.5]]), "must not repeat"),
        (torch.tensor([[0, 1]]), torch.tensor([[0.4, 0.4]]), "sum to one"),
    ],
)
def test_cpu_rejects_invalid_routing(
    ids: torch.Tensor,
    weights: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        cpu(torch.ones(1, 2), ids, weights, torch.ones(2, 12))
