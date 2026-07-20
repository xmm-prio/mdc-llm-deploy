"""RMS normalization custom operator implementation."""

from __future__ import annotations

import math
from typing import Any, ClassVar, cast

import torch
from torch.onnx import symbolic_helper

from .base import CustomOp

_TRITON_RUNTIME: tuple[Any, Any] | None = None
tl: Any = None


def _load_triton_runtime() -> tuple[Any, Any]:
    """Load and define the Triton kernel only when CUDA execution is requested."""
    global _TRITON_RUNTIME
    if _TRITON_RUNTIME is not None:
        return _TRITON_RUNTIME

    try:
        import triton  # type: ignore[import-not-found]
        import triton.language as triton_language  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("RmsNorm CUDA execution requires Triton") from error
    globals()["tl"] = triton_language

    def rms_norm_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        gamma_ptr,
        y_ptr,
        rstd_ptr,
        epsilon,
        normalized_size,
        block_size,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, block_size)
        mask = columns < normalized_size
        offsets = row * normalized_size + columns
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / normalized_size
        rstd = 1.0 / tl.sqrt(mean_square + epsilon)
        tl.store(y_ptr + offsets, x * rstd * gamma, mask=mask)
        tl.store(rstd_ptr + row, rstd)

    rms_norm_kernel.__annotations__.update(
        {
            "normalized_size": triton_language.constexpr,
            "block_size": triton_language.constexpr,
        }
    )
    compiled_kernel = triton.jit(rms_norm_kernel)
    _TRITON_RUNTIME = (triton, compiled_kernel)
    return _TRITON_RUNTIME


class RmsNorm(CustomOp):
    """Apply RMS normalization over the trailing dimensions described by gamma."""

    qualified_name: ClassVar[str] = "mdc_llm_deploy::rms_norm"
    schema: ClassVar[str] = (
        "(Tensor x, Tensor gamma, float epsilon=1e-6) -> (Tensor y, Tensor rstd)"
    )
    _supported_dtypes: ClassVar[frozenset[torch.dtype]] = frozenset(
        {torch.float16, torch.bfloat16, torch.float32}
    )
    _max_triton_block_size: ClassVar[int] = 65_536

    @classmethod
    def _validate(
        cls,
        x: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float,
        *,
        device_type: str | None = None,
        check_values: bool,
    ) -> None:
        if not isinstance(x, torch.Tensor) or not isinstance(gamma, torch.Tensor):
            raise TypeError("x and gamma must be tensors")
        if not isinstance(epsilon, (float, int)) or isinstance(epsilon, bool):
            raise TypeError("epsilon must be a real number")
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if not 1 <= x.ndim <= 8:
            raise ValueError("x must have between 1 and 8 dimensions")
        if not 1 <= gamma.ndim <= x.ndim:
            raise ValueError("gamma must have between 1 and x.ndim dimensions")
        if tuple(x.shape[-gamma.ndim :]) != tuple(gamma.shape):
            raise ValueError("gamma shape must match one or more trailing x dimensions")
        if any(size == 0 for size in gamma.shape):
            raise ValueError("gamma dimensions must be non-empty")
        if x.dtype not in cls._supported_dtypes or gamma.dtype not in cls._supported_dtypes:
            raise TypeError("x and gamma must use float16, bfloat16, or float32")
        if x.dtype != gamma.dtype:
            raise TypeError("x and gamma must have the same dtype")
        if x.device != gamma.device:
            raise ValueError("x and gamma must be on the same device")
        if device_type is not None and x.device.type != device_type:
            raise ValueError(f"x and gamma must be on {device_type}")
        if check_values and (
            not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(gamma).all())
        ):
            raise ValueError("x and gamma must contain only finite values")

    @staticmethod
    def _output_shape(x: torch.Tensor, gamma: torch.Tensor) -> tuple[Any, ...]:
        return tuple(x.shape[: x.ndim - gamma.ndim])

    @classmethod
    def cpu(
        cls,
        x: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute RMS normalization with FP32 accumulation on CPU."""
        cls._validate(x, gamma, epsilon, device_type="cpu", check_values=True)
        normalized_dims = tuple(range(x.ndim - gamma.ndim, x.ndim))
        x_float = x.float()
        rstd = torch.rsqrt(torch.mean(x_float.square(), dim=normalized_dims) + float(epsilon))
        scale_shape = (*rstd.shape, *((1,) * gamma.ndim))
        y = (x_float * rstd.reshape(scale_shape) * gamma.float()).to(x.dtype)
        return y, rstd

    @classmethod
    def cuda(
        cls,
        x: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute RMS normalization using the Triton CUDA kernel."""
        cls._validate(x, gamma, epsilon, device_type="cuda", check_values=True)
        if not x.is_contiguous() or not gamma.is_contiguous():
            raise ValueError("RmsNorm CUDA inputs must be contiguous")

        normalized_size = gamma.numel()
        if normalized_size > cls._max_triton_block_size:
            raise ValueError(
                f"RmsNorm CUDA normalized size must not exceed {cls._max_triton_block_size}"
            )

        triton, kernel = _load_triton_runtime()
        row_count = x.numel() // normalized_size
        y = torch.empty_like(x)
        rstd = torch.empty(cls._output_shape(x, gamma), dtype=torch.float32, device=x.device)
        if row_count == 0:
            return y, rstd

        block_size = triton.next_power_of_2(normalized_size)
        num_warps = 8 if block_size >= 2_048 else 4
        kernel[(row_count,)](
            x,
            gamma,
            y,
            rstd,
            normalized_size=normalized_size,
            block_size=block_size,
            num_warps=num_warps,
            epsilon=float(epsilon),
        )
        return y, rstd

    @classmethod
    def fake(
        cls,
        x: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer output metadata without reading tensor values."""
        cls._validate(x, gamma, epsilon, check_values=False)
        y = torch.empty_like(x)
        rstd = torch.empty(cls._output_shape(x, gamma), dtype=torch.float32, device=x.device)
        return y, rstd

    @classmethod
    def _validate_onnx(cls, x: Any, gamma: Any, epsilon: float) -> None:
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        try:
            x_type = x.type()
            gamma_type = gamma.type()
            x_shape = x_type.sizes()
            gamma_shape = gamma_type.sizes()
            x_dtype = x_type.scalarType()
            gamma_dtype = gamma_type.scalarType()
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("RmsNorm ONNX export requires tensor type metadata") from error

        if x_shape is None or gamma_shape is None:
            raise RuntimeError("RmsNorm ONNX export requires known input ranks")
        if any(size is None for size in (*x_shape, *gamma_shape)):
            raise RuntimeError("RmsNorm ONNX export requires static input shapes")
        if not 1 <= len(x_shape) <= 8 or not 1 <= len(gamma_shape) <= len(x_shape):
            raise RuntimeError("RmsNorm ONNX export supports x rank 1 to 8 and trailing gamma")
        if tuple(x_shape[-len(gamma_shape) :]) != tuple(gamma_shape):
            raise RuntimeError("RmsNorm ONNX gamma shape must match trailing x dimensions")
        if x_dtype not in {"Float", "Half", "BFloat16"} or gamma_dtype != x_dtype:
            raise RuntimeError("RmsNorm ONNX inputs must have the same supported floating dtype")

    @staticmethod
    @symbolic_helper.parse_args("v", "v", "f")
    def onnx(
        graph: Any,
        x: Any,
        gamma: Any,
        epsilon: float = 1e-6,
    ) -> tuple[Any, Any]:
        """Emit the opset-18 NPURmsNorm node using the documented ABI."""
        RmsNorm._validate_onnx(x, gamma, epsilon)
        return cast(
            tuple[Any, Any],
            graph.op("NPURmsNorm", x, gamma, epsilon_f=float(epsilon), outputs=2),
        )
