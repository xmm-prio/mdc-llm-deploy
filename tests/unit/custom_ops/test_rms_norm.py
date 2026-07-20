from __future__ import annotations

from typing import Any

import pytest
import torch

from mdc_llm_deploy.custom_ops.rms_norm import RmsNorm


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
def test_cpu_matches_fp32_reference(
    dtype: torch.dtype,
    gamma_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 3, 4, dtype=dtype)
    gamma = torch.randn(gamma_shape, dtype=dtype)

    actual_y, actual_rstd = RmsNorm.cpu(x, gamma, 1e-5)
    expected_y, expected_rstd = _reference(x, gamma, 1e-5)

    torch.testing.assert_close(actual_y, expected_y)
    torch.testing.assert_close(actual_rstd, expected_rstd)
    assert actual_y.dtype == dtype
    assert actual_rstd.dtype == torch.float32
    assert actual_rstd.shape == x.shape[: x.ndim - len(gamma_shape)]


@pytest.mark.parametrize(
    ("x", "gamma", "epsilon", "error_type", "message"),
    [
        (
            torch.ones(2, 3),
            torch.ones(2),
            1e-6,
            ValueError,
            "trailing",
        ),
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
        (
            torch.ones(2, 3),
            torch.ones(3),
            0.0,
            ValueError,
            "positive",
        ),
        (
            torch.tensor([[float("nan"), 1.0]]),
            torch.ones(2),
            1e-6,
            ValueError,
            "finite values",
        ),
        (
            torch.ones(2, 3),
            torch.tensor([1.0, float("inf"), 1.0]),
            1e-6,
            ValueError,
            "finite values",
        ),
    ],
)
def test_cpu_rejects_invalid_contract(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        RmsNorm.cpu(x, gamma, epsilon)


def test_fake_returns_documented_metadata() -> None:
    x = torch.empty(2, 3, 4, dtype=torch.float16, device="meta")
    gamma = torch.empty(3, 4, dtype=torch.float16, device="meta")

    y, rstd = RmsNorm.fake(x, gamma)

    assert y.shape == (2, 3, 4)
    assert y.dtype == torch.float16
    assert y.device.type == "meta"
    assert rstd.shape == (2,)
    assert rstd.dtype == torch.float32
    assert rstd.device.type == "meta"


class _Graph:
    def __init__(self) -> None:
        self.call: tuple[str, tuple[Any, ...], dict[str, Any]] | None = None

    def op(self, name: str, *args: Any, **kwargs: Any) -> tuple[str, str]:
        self.call = (name, args, kwargs)
        return ("y", "rstd")


class _TensorType:
    def __init__(self, shape: tuple[int | None, ...], dtype: str) -> None:
        self._shape = shape
        self._dtype = dtype

    def sizes(self) -> tuple[int | None, ...]:
        return self._shape

    def scalarType(self) -> str:  # noqa: N802
        return self._dtype


class _Value:
    def __init__(self, shape: tuple[int | None, ...], dtype: str = "Float") -> None:
        self._type = _TensorType(shape, dtype)

    def type(self) -> _TensorType:
        return self._type


def test_onnx_symbolic_uses_documented_abi() -> None:
    graph = _Graph()
    x = _Value((2, 3, 4))
    gamma = _Value((3, 4))

    outputs = RmsNorm.onnx(graph, x, gamma, 1e-5)

    assert outputs == ("y", "rstd")
    assert graph.call == (
        "NPURmsNorm",
        (x, gamma),
        {"epsilon_f": 1e-5, "outputs": 2},
    )


def test_onnx_symbolic_rejects_dynamic_shapes() -> None:
    graph = _Graph()
    x = _Value((2, 3, None))
    gamma = _Value((3, 4))

    with pytest.raises(RuntimeError, match="static input shapes"):
        RmsNorm.onnx(graph, x, gamma)
