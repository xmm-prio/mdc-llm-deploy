from __future__ import annotations

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from mdc_llm_deploy.custom_ops.moe_expert import MoeExpert


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
            hidden = hidden * (up[expert] @ x[token].float())
            output[token] += topk_weight[token, route].float() * (down[expert] @ hidden)
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


def test_cpu_float_matches_reference() -> None:
    x, topk_ids, topk_weight, gate, up, down = _floating_case()

    actual = MoeExpert.cpu(x, topk_ids, topk_weight, _pack(gate, up, down))

    expected = _reference(x, topk_ids, topk_weight, gate, up, down)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("with_offsets", [False, True])
def test_cpu_int8_matches_dequantized_reference(with_offsets: bool) -> None:
    torch.manual_seed(11)
    expert_count, hidden_size, intermediate_size = 2, 3, 4
    x = torch.randn(2, hidden_size)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    topk_weight = torch.tensor([[0.3, 0.7], [0.8, 0.2]])
    packed = torch.randint(
        -20,
        20,
        (expert_count, 3 * hidden_size * intermediate_size),
        dtype=torch.int8,
    )
    scales = torch.rand(expert_count, 2 * intermediate_size + hidden_size) * 0.05 + 0.01
    offsets = (
        torch.randint(-3, 4, scales.shape, dtype=torch.int32).float() if with_offsets else None
    )
    effective_offsets = torch.zeros_like(scales) if offsets is None else offsets
    gate_scale, up_scale, down_scale = torch.split(
        scales, [intermediate_size, intermediate_size, hidden_size], dim=1
    )
    gate_offset, up_offset, down_offset = torch.split(
        effective_offsets, [intermediate_size, intermediate_size, hidden_size], dim=1
    )
    gate_end = hidden_size * intermediate_size
    gate = packed[:, :gate_end].float().reshape(expert_count, intermediate_size, hidden_size)
    up = packed[:, gate_end : 2 * gate_end].float().reshape(
        expert_count, intermediate_size, hidden_size
    )
    down = packed[:, 2 * gate_end :].float().reshape(
        expert_count, hidden_size, intermediate_size
    )
    gate = (gate - gate_offset.unsqueeze(-1)) * gate_scale.unsqueeze(-1)
    up = (up - up_offset.unsqueeze(-1)) * up_scale.unsqueeze(-1)
    down = (down - down_offset.unsqueeze(-1)) * down_scale.unsqueeze(-1)

    actual = MoeExpert.cpu(x, topk_ids, topk_weight, packed, scales, offsets)

    torch.testing.assert_close(
        actual,
        _reference(x, topk_ids, topk_weight, gate, up, down),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("ids", "weights", "message"),
    [
        (
            torch.tensor([[0, 2]], dtype=torch.int64),
            torch.tensor([[0.5, 0.5]]),
            "out-of-range",
        ),
        (
            torch.tensor([[0, 0]], dtype=torch.int64),
            torch.tensor([[0.5, 0.5]]),
            "must not repeat",
        ),
        (
            torch.tensor([[0, 1]], dtype=torch.int64),
            torch.tensor([[0.4, 0.4]]),
            "sum to one",
        ),
        (
            torch.tensor([[0, 1]], dtype=torch.int64),
            torch.tensor([[-0.1, 1.1]]),
            "non-negative",
        ),
    ],
)
def test_cpu_rejects_invalid_routing(
    ids: torch.Tensor,
    weights: torch.Tensor,
    message: str,
) -> None:
    x = torch.ones(1, 2)
    packed = torch.ones(2, 12)

    with pytest.raises(ValueError, match=message):
        MoeExpert.cpu(x, ids, weights, packed)


def test_rejects_invalid_packed_and_quantization_contracts() -> None:
    x = torch.ones(1, 2)
    ids = torch.zeros((1, 1), dtype=torch.int64)
    routing = torch.ones(1, 1)

    with pytest.raises(ValueError, match="packed width"):
        MoeExpert.cpu(x, ids, routing, torch.ones(1, 13))
    with pytest.raises(ValueError, match="require quant_scales"):
        MoeExpert.cpu(x, ids, routing, torch.ones(1, 12, dtype=torch.int8))
    with pytest.raises(ValueError, match="quant_scales"):
        MoeExpert.cpu(
            x,
            ids,
            routing,
            torch.ones(1, 12, dtype=torch.int8),
            torch.ones(1, 4),
        )
    with pytest.raises(ValueError, match="must not use"):
        MoeExpert.cpu(x, ids, routing, torch.ones(1, 12), torch.ones(1, 4))


def test_fake_and_meta_preserve_x_metadata() -> None:
    meta_output = MoeExpert.meta(
        torch.empty(2, 4, device="meta"),
        torch.empty(2, 2, dtype=torch.int64, device="meta"),
        torch.empty(2, 2, device="meta"),
        torch.empty(3, 60, device="meta"),
    )

    with FakeTensorMode() as mode:
        fake_output = MoeExpert.fake(
            mode.from_tensor(torch.empty(3, 4)),
            mode.from_tensor(torch.empty(3, 1, dtype=torch.int32)),
            mode.from_tensor(torch.empty(3, 1)),
            mode.from_tensor(torch.empty(2, 36)),
        )

    assert meta_output.shape == (2, 4)
    assert meta_output.device.type == "meta"
    assert isinstance(fake_output, FakeTensor)
    assert fake_output.shape == (3, 4)
    assert fake_output.dtype == torch.float32


class _GraphRecorder:
    def __init__(self) -> None:
        self.call: tuple[str, tuple[Any, ...]] | None = None

    def op(self, name: str, *inputs: Any) -> object:
        self.call = (name, inputs)
        return object()


class _SymbolicType:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self._shape = shape
        self._dtype = dtype

    def sizes(self) -> tuple[int, ...]:
        return self._shape

    def scalarType(self) -> str:  # noqa: N802
        return self._dtype


class _SymbolicValue:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self._type = _SymbolicType(shape, dtype)

    def type(self) -> _SymbolicType:
        return self._type


def _mdc_symbolic_inputs(x_dtype: str = "Char") -> list[_SymbolicValue]:
    return [
        _SymbolicValue((1, 256), x_dtype),
        _SymbolicValue((1, 2), "Short"),
        _SymbolicValue((1, 2), "Half"),
        _SymbolicValue((3 * 4 * 256, 256), "Char"),
        _SymbolicValue((17,), "Float"),
    ]


def test_onnx_emits_real_mdc_six_slot_abi() -> None:
    graph = _GraphRecorder()
    inputs = _mdc_symbolic_inputs()

    output = MoeExpert.onnx(graph, *inputs)

    assert output is not None
    assert graph.call == ("MoeExpert", (*inputs, None))


def test_onnx_rejects_floating_torch_contract() -> None:
    graph = _GraphRecorder()

    with pytest.raises(RuntimeError, match="x must be INT8"):
        MoeExpert.onnx(graph, *_mdc_symbolic_inputs("Float"))


def test_mdc_cpu_and_fake_use_fp16_output_contract() -> None:
    x = torch.ones((1, 256), dtype=torch.int8)
    ids = torch.tensor([[0]], dtype=torch.int16)
    routing = torch.ones((1, 1), dtype=torch.float16)
    weights = torch.ones((3 * 128, 256), dtype=torch.int8)
    scales = torch.tensor([0.01, 0.02, 0.02, 0.001, 0.02], dtype=torch.float32)

    output = MoeExpert.cpu(x, ids, routing, weights, scales)
    meta_output = MoeExpert.fake(
        x.to(device="meta"),
        ids.to(device="meta"),
        routing.to(device="meta"),
        weights.to(device="meta"),
        scales.to(device="meta"),
    )

    assert output.shape == (1, 256)
    assert output.dtype == torch.float16
    assert meta_output.shape == (1, 256)
    assert meta_output.dtype == torch.float16


def test_mdc_contract_rejects_quant_offsets() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        MoeExpert.cpu(
            torch.ones((1, 256), dtype=torch.int8),
            torch.zeros((1, 1), dtype=torch.int16),
            torch.ones((1, 1), dtype=torch.float16),
            torch.ones((3 * 128, 256), dtype=torch.int8),
            torch.ones(5, dtype=torch.float32),
            torch.zeros(1, dtype=torch.int32),
        )
