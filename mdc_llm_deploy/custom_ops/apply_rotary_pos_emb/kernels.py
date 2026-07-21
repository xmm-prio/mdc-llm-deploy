"""CPU and CUDA kernels for ApplyRotaryPosEmb."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .contract import HEAD_AXIS, validate_torch_inputs

_TRITON_KERNEL: Any | None = None
tl: Any = None


def rotate(input: torch.Tensor, rotary_mode: str) -> torch.Tensor:
    """Rotate the active trailing dimension in the selected pairing mode."""
    if rotary_mode == "half":
        first, second = input.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)
    if rotary_mode == "interleave":
        pairs = input.reshape(*input.shape[:-1], -1, 2)
        return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    first, second, third, fourth = input.chunk(4, dim=-1)
    return torch.cat((-second, first, -fourth, third), dim=-1)


def cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int = 1,
    rotary_mode: str = "half",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute a correctness-first FP32 CPU implementation."""
    rotary_dim = validate_torch_inputs(
        query, key, cos, sin, layout, rotary_mode, check_values=True
    )
    cos_fp32 = cos.float()
    sin_fp32 = sin.float()

    def apply(input: torch.Tensor) -> torch.Tensor:
        rotary = input[..., :rotary_dim].float()
        output = rotary * cos_fp32 + rotate(rotary, rotary_mode) * sin_fp32
        if rotary_dim != input.shape[-1]:
            output = torch.cat((output, input[..., rotary_dim:].float()), dim=-1)
        return output.to(input.dtype)

    return apply(query), apply(key)


def _load_triton_kernel() -> Any:
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    try:
        triton = importlib.import_module("triton")
        triton_language = importlib.import_module("triton.language")
    except ImportError as error:
        raise RuntimeError("Triton is required for ApplyRotaryPosEmb CUDA execution") from error
    globals()["tl"] = triton_language

    def kernel(  # type: ignore[no-untyped-def]
        input_ptr,
        cos_ptr,
        sin_ptr,
        output_ptr,
        rows,
        head_dim,
        rotary_dim,
        dim0,
        dim1,
        dim2,
        cos_dim0,
        cos_dim1,
        cos_dim2,
        cos_stride0,
        cos_stride1,
        cos_stride2,
        cos_stride3,
        rank,
        head_axis,
        mode,
        block_size,
    ):
        row = tl.program_id(0)
        column = tl.program_id(1) * block_size + tl.arange(0, block_size)
        mask = (row < rows) & (column < head_dim)
        input_offset = row * head_dim + column
        value = tl.load(input_ptr + input_offset, mask=mask).to(tl.float32)

        if rank == 4:
            index2 = row % dim2
            quotient = row // dim2
            index1 = quotient % dim1
            index0 = quotient // dim1
            cos_index0 = tl.where(cos_dim0 == 1, 0, index0)
            cos_index1 = tl.where((cos_dim1 == 1) | (head_axis == 1), 0, index1)
            cos_index2 = tl.where((cos_dim2 == 1) | (head_axis == 2), 0, index2)
            cos_base = (
                cos_index0 * cos_stride0
                + cos_index1 * cos_stride1
                + cos_index2 * cos_stride2
            )
        else:
            index1 = row % dim1
            index0 = row // dim1
            cos_index0 = tl.where(cos_dim0 == 1, 0, index0)
            cos_index1 = tl.where((cos_dim1 == 1) | (head_axis == 1), 0, index1)
            cos_base = cos_index0 * cos_stride0 + cos_index1 * cos_stride1

        if mode == 0:
            half = rotary_dim // 2
            source_column = tl.where(column < half, column + half, column - half)
            sign = tl.where(column < half, -1.0, 1.0)
        elif mode == 1:
            source_column = tl.where(column % 2 == 0, column + 1, column - 1)
            sign = tl.where(column % 2 == 0, -1.0, 1.0)
        else:
            quarter = rotary_dim // 4
            segment = column // quarter
            source_column = tl.where(segment % 2 == 0, column + quarter, column - quarter)
            sign = tl.where(segment % 2 == 0, -1.0, 1.0)

        rotary_mask = mask & (column < rotary_dim)
        source = tl.load(
            input_ptr + row * head_dim + source_column, mask=rotary_mask
        ).to(tl.float32)
        cosine = tl.load(
            cos_ptr + cos_base + column * cos_stride3, mask=rotary_mask
        ).to(tl.float32)
        sine = tl.load(
            sin_ptr + cos_base + column * cos_stride3, mask=rotary_mask
        ).to(tl.float32)
        result = tl.where(
            column < rotary_dim, value * cosine + sign * source * sine, value
        )
        tl.store(output_ptr + input_offset, result, mask=mask)

    kernel.__annotations__.update(
        {
            "rank": triton_language.constexpr,
            "head_axis": triton_language.constexpr,
            "mode": triton_language.constexpr,
            "block_size": triton_language.constexpr,
        }
    )
    _TRITON_KERNEL = triton.jit(kernel)
    return _TRITON_KERNEL


def cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int = 1,
    rotary_mode: str = "half",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the declared CUDA range with Triton, without fallback."""
    rotary_dim = validate_torch_inputs(
        query, key, cos, sin, layout, rotary_mode, check_values=True
    )
    if query.device.type != "cuda":
        raise RuntimeError("ApplyRotaryPosEmb.cuda requires CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in (query, key, cos, sin)):
        raise ValueError("ApplyRotaryPosEmb CUDA inputs must be contiguous")

    kernel = _load_triton_kernel()
    triton = importlib.import_module("triton")
    mode = {"half": 0, "interleave": 1, "quarter": 2}[rotary_mode]
    rank = query.ndim
    head_axis = HEAD_AXIS[layout]
    block_size = int(triton.next_power_of_2(query.shape[-1]))

    def launch(input: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(input)
        rows = input.numel() // input.shape[-1]
        dimensions = (*input.shape[:-1], 1, 1)
        cos_dimensions = (*cos.shape[:-1], 1, 1)
        cos_strides = (*cos.stride(), 1)
        grid = (rows, triton.cdiv(input.shape[-1], block_size))
        kernel[grid](
            input,
            cos,
            sin,
            output,
            rows,
            input.shape[-1],
            rotary_dim,
            dimensions[0],
            dimensions[1],
            dimensions[2],
            cos_dimensions[0],
            cos_dimensions[1],
            cos_dimensions[2],
            cos_strides[0],
            cos_strides[1],
            cos_strides[2],
            cos_strides[-1],
            rank=rank,
            head_axis=head_axis,
            mode=mode,
            block_size=block_size,
        )
        return output

    return launch(query), launch(key)
