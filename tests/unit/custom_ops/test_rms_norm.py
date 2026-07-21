from __future__ import annotations

import pytest
import torch
from onnxscript import ir

from mdc_llm_deploy.custom_ops.rms_norm import (
    cpu,
    fake,
    rms_norm,
    validate_onnx_inputs,
)


def _reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dims = tuple(range(x.ndim - gamma.ndim, x.ndim))
    rstd = torch.rsqrt(x.float().square().mean(dim=dims) + epsilon)
    y = x.float() * rstd.reshape(*rstd.shape, *((1,) * gamma.ndim)) * gamma.float()
    return y.to(x.dtype), rstd


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gamma_shape", [(4,), (3, 4)])
def test_registered_operator_keeps_broad_torch_contract(
    dtype: torch.dtype,
    gamma_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 3, 4, dtype=dtype)
    gamma = torch.randn(gamma_shape, dtype=dtype)

    actual_y, actual_rstd = rms_norm(x, gamma, 1e-5)
    expected_y, expected_rstd = _reference(x, gamma, 1e-5)

    torch.testing.assert_close(actual_y, expected_y)
    torch.testing.assert_close(actual_rstd, expected_rstd)
    assert actual_y.dtype == dtype
    assert actual_rstd.dtype == torch.float32
    assert actual_rstd.shape == x.shape[: x.ndim - len(gamma_shape)]


@pytest.mark.parametrize(
    ("x", "gamma", "epsilon", "error_type", "message"),
    [
        (torch.ones(2, 3), torch.ones(2), 1e-6, ValueError, "trailing"),
        (
            torch.ones(2, 3),
            torch.ones(3, dtype=torch.float16),
            1e-6,
            TypeError,
            "same dtype",
        ),
        (
            torch.ones(2, 3, dtype=torch.int32),
            torch.ones(3, dtype=torch.int32),
            1e-6,
            TypeError,
            "float16",
        ),
        (torch.ones(2, 3), torch.ones(3), 0.0, ValueError, "positive"),
        (
            torch.tensor([[float("nan"), 1.0]]),
            torch.ones(2),
            1e-6,
            ValueError,
            "finite values",
        ),
    ],
)
def test_cpu_rejects_invalid_torch_contract(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        cpu(x, gamma, epsilon)


def test_fake_returns_documented_metadata() -> None:
    x = torch.empty(2, 3, 4, dtype=torch.float16, device="meta")
    gamma = torch.empty(3, 4, dtype=torch.float16, device="meta")

    y, rstd = fake(x, gamma)
    registered_y, registered_rstd = rms_norm(x, gamma)

    for actual_y, actual_rstd in ((y, rstd), (registered_y, registered_rstd)):
        assert actual_y.shape == (2, 3, 4)
        assert actual_y.dtype == torch.float16
        assert actual_y.device.type == "meta"
        assert actual_rstd.shape == (2,)
        assert actual_rstd.dtype == torch.float32
        assert actual_rstd.device.type == "meta"


def test_registered_operator_passes_opcheck() -> None:
    x = torch.randn(2, 3, 4)
    gamma = torch.randn(3, 4)

    result = torch.library.opcheck(rms_norm, (x, gamma, 1e-5))

    assert set(result.values()) == {"SUCCESS"}


def _ir_value(shape: list[int | str], dtype: ir.DataType = ir.DataType.FLOAT) -> ir.Value:
    return ir.Value(shape=ir.Shape(shape), type=ir.TensorType(dtype))


def test_onnx_contract_accepts_static_trailing_shape() -> None:
    validate_onnx_inputs(_ir_value([2, 3, 4]), _ir_value([3, 4]), 1e-5)


def test_onnx_contract_rejects_torch_legal_dynamic_shape() -> None:
    x = _ir_value([2, "sequence", 4])
    gamma = _ir_value([4])

    with pytest.raises(RuntimeError, match="static input shapes"):
        validate_onnx_inputs(x, gamma, 1e-5)
