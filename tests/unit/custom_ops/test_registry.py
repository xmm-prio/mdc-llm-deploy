from __future__ import annotations

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from mdc_llm_deploy.custom_ops import (
    CustomOp,
    get_custom_op,
    register_custom_op,
    registered_custom_ops,
)


class IdentityOp(CustomOp):
    qualified_name = "mdc_test::identity"
    schema = "(Tensor input) -> Tensor"

    @staticmethod
    def cpu(input: torch.Tensor) -> torch.Tensor:
        return input.clone()

    @staticmethod
    def cuda(input: torch.Tensor) -> torch.Tensor:
        return input.clone()

    @staticmethod
    def fake(input: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(input)

    @staticmethod
    def onnx(*args: Any, **kwargs: Any) -> Any:
        return None


class ScaleOp(CustomOp):
    qualified_name = "mdc_test::scale"
    schema = "(Tensor input) -> Tensor"

    @staticmethod
    def cpu(input: torch.Tensor) -> torch.Tensor:
        return input * 2

    @staticmethod
    def cuda(input: torch.Tensor) -> torch.Tensor:
        return input * 2

    @staticmethod
    def fake(input: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(input)

    @staticmethod
    def onnx(*args: Any, **kwargs: Any) -> Any:
        return None


def test_registration_is_incremental_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    onnx_calls: list[tuple[str, Any, int]] = []
    monkeypatch.setattr(
        torch.onnx,
        "register_custom_op_symbolic",
        lambda name, symbolic, opset: onnx_calls.append((name, symbolic, opset)),
    )

    first = register_custom_op(IdentityOp)
    second = register_custom_op(IdentityOp)

    assert first is second
    assert get_custom_op(IdentityOp.qualified_name) is first
    assert first in registered_custom_ops()
    assert onnx_calls == [(IdentityOp.qualified_name, IdentityOp.onnx, 18)]
    assert torch.equal(first.definition(torch.tensor([1.0])), torch.tensor([1.0]))


def test_fake_interface_covers_fake_and_meta_tensors() -> None:
    entry = register_custom_op(ScaleOp)
    meta_output = entry.definition(torch.empty(2, 3, device="meta"))

    with FakeTensorMode() as mode:
        fake_input = mode.from_tensor(torch.empty(4, 5))
        fake_output = entry.definition(fake_input)

    assert meta_output.device.type == "meta"
    assert meta_output.shape == (2, 3)
    assert isinstance(fake_output, FakeTensor)
    assert fake_output.shape == (4, 5)


def test_autograd_is_explicitly_rejected() -> None:
    entry = register_custom_op(IdentityOp)
    output = entry.definition(torch.ones(2, requires_grad=True))

    with pytest.raises(RuntimeError, match="inference-only"):
        output.sum().backward()
