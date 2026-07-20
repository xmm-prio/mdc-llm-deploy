"""Apply rotary position embeddings to query and key tensors."""

import importlib
from typing import Any, ClassVar

import torch
from torch.onnx import symbolic_helper

from .base import CustomOp


class ApplyRotaryPosEmb(CustomOp):
    """Implement the fused MDC query/key rotary position embedding operator."""

    qualified_name = "mdc_llm_deploy::apply_rotary_pos_emb"
    schema = (
        "(Tensor query, Tensor key, Tensor cos, Tensor sin, "
        "int layout=1, str rotary_mode='half') -> (Tensor, Tensor)"
    )
    onnx_opset = 18

    _LAYOUT_RANK: ClassVar[dict[int, int]] = {1: 4, 2: 4, 3: 4, 4: 3}
    _HEAD_AXIS: ClassVar[dict[int, int]] = {1: 2, 2: 2, 3: 1, 4: 1}
    _ROTARY_MODES: ClassVar[frozenset[str]] = frozenset(
        {"half", "interleave", "quarter"}
    )
    _DTYPES: ClassVar[frozenset[torch.dtype]] = frozenset(
        {torch.float16, torch.bfloat16, torch.float32}
    )
    _triton_kernel: ClassVar[Any | None] = None

    @classmethod
    def _validate(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layout: int,
        rotary_mode: str,
        *,
        check_values: bool,
    ) -> int:
        if layout not in cls._LAYOUT_RANK:
            raise ValueError("layout must be one of 1=BSND, 2=SBND, 3=BNSD, or 4=TND")
        if rotary_mode not in cls._ROTARY_MODES:
            raise ValueError("rotary_mode must be 'half', 'interleave', or 'quarter'")

        tensors = (query, key, cos, sin)
        if any(tensor.dtype not in cls._DTYPES for tensor in tensors):
            raise TypeError("query, key, cos, and sin must have a floating-point RoPE dtype")
        if any(tensor.dtype != query.dtype for tensor in tensors[1:]):
            raise TypeError("query, key, cos, and sin must have the same dtype")
        if any(tensor.device != query.device for tensor in tensors[1:]):
            raise ValueError("query, key, cos, and sin must be on the same device")

        rank = cls._LAYOUT_RANK[layout]
        if any(tensor.ndim != rank for tensor in tensors):
            raise ValueError(f"layout {layout} requires every input to have rank {rank}")
        if cos.shape != sin.shape:
            raise ValueError("cos and sin must have the same shape")

        head_axis = cls._HEAD_AXIS[layout]
        for axis in range(rank):
            if axis != head_axis and query.shape[axis] != key.shape[axis]:
                raise ValueError("query and key may differ only in head count")
        if cos.shape[head_axis] != 1:
            raise ValueError("cos and sin head dimension must be 1")
        for axis in range(rank - 1):
            expected = query.shape[axis]
            if axis == head_axis:
                continue
            if cos.shape[axis] not in (1, expected):
                raise ValueError("cos and sin must broadcast over non-head dimensions")

        head_dim = query.shape[-1]
        if key.shape[-1] != head_dim:
            raise ValueError("query and key must have the same head dimension")
        rotary_dim = cos.shape[-1]
        if rotary_dim <= 0 or rotary_dim > head_dim:
            raise ValueError("rotary dimension must satisfy 0 < R <= D")
        divisor = 4 if rotary_mode == "quarter" else 2
        if rotary_dim % divisor:
            raise ValueError(f"{rotary_mode} rotary dimension must be divisible by {divisor}")

        if check_values:
            for name, tensor in zip(("query", "key", "cos", "sin"), tensors, strict=True):
                if not bool(torch.isfinite(tensor).all().item()):
                    raise ValueError(f"{name} must contain only finite values")
        return rotary_dim

    @staticmethod
    def _rotate(input: torch.Tensor, rotary_mode: str) -> torch.Tensor:
        if rotary_mode == "half":
            first, second = input.chunk(2, dim=-1)
            return torch.cat((-second, first), dim=-1)
        if rotary_mode == "interleave":
            pairs = input.reshape(*input.shape[:-1], -1, 2)
            return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        first, second, third, fourth = input.chunk(4, dim=-1)
        return torch.cat((-second, first, -fourth, third), dim=-1)

    @classmethod
    def _reference(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layout: int,
        rotary_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotary_dim = cls._validate(
            query, key, cos, sin, layout, rotary_mode, check_values=True
        )
        cos_fp32 = cos.float()
        sin_fp32 = sin.float()

        def apply(input: torch.Tensor) -> torch.Tensor:
            rotary = input[..., :rotary_dim].float()
            rotated = rotary * cos_fp32 + cls._rotate(rotary, rotary_mode) * sin_fp32
            if rotary_dim != input.shape[-1]:
                rotated = torch.cat((rotated, input[..., rotary_dim:].float()), dim=-1)
            return rotated.to(input.dtype)

        return apply(query), apply(key)

    @classmethod
    def cpu(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layout: int = 1,
        rotary_mode: str = "half",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute a correctness-first FP32 CPU implementation."""
        return cls._reference(query, key, cos, sin, layout, rotary_mode)

    @classmethod
    def _load_triton_kernel(cls) -> Any:
        if cls._triton_kernel is not None:
            return cls._triton_kernel
        try:
            triton = importlib.import_module("triton")
            tl = importlib.import_module("triton.language")
        except ImportError as error:
            raise RuntimeError("Triton is required for ApplyRotaryPosEmb CUDA execution") from error

        @triton.jit  # type: ignore[misc]
        def kernel(
            input_ptr: Any,
            cos_ptr: Any,
            sin_ptr: Any,
            output_ptr: Any,
            rows: Any,
            head_dim: Any,
            rotary_dim: Any,
            dim0: Any,
            dim1: Any,
            dim2: Any,
            cos_dim0: Any,
            cos_dim1: Any,
            cos_dim2: Any,
            cos_stride0: Any,
            cos_stride1: Any,
            cos_stride2: Any,
            cos_stride3: Any,
            rank: tl.constexpr,  # type: ignore[name-defined]
            head_axis: tl.constexpr,  # type: ignore[name-defined]
            mode: tl.constexpr,  # type: ignore[name-defined]
            block_size: tl.constexpr,  # type: ignore[name-defined]
        ) -> None:
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
                cos_index1 = tl.where(
                    (cos_dim1 == 1) | (head_axis == 1), 0, index1
                )
                cos_index2 = tl.where(
                    (cos_dim2 == 1) | (head_axis == 2), 0, index2
                )
                cos_base = (
                    cos_index0 * cos_stride0
                    + cos_index1 * cos_stride1
                    + cos_index2 * cos_stride2
                )
            else:
                index1 = row % dim1
                index0 = row // dim1
                cos_index0 = tl.where(cos_dim0 == 1, 0, index0)
                cos_index1 = tl.where(
                    (cos_dim1 == 1) | (head_axis == 1), 0, index1
                )
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

        cls._triton_kernel = kernel
        return kernel

    @classmethod
    def cuda(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layout: int = 1,
        rotary_mode: str = "half",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the declared CUDA range with Triton, without fallback."""
        rotary_dim = cls._validate(
            query, key, cos, sin, layout, rotary_mode, check_values=True
        )
        if query.device.type != "cuda":
            raise RuntimeError("ApplyRotaryPosEmb.cuda requires CUDA tensors")
        if any(not tensor.is_contiguous() for tensor in (query, key, cos, sin)):
            raise ValueError("ApplyRotaryPosEmb CUDA inputs must be contiguous")

        kernel = cls._load_triton_kernel()
        triton = importlib.import_module("triton")
        mode = {"half": 0, "interleave": 1, "quarter": 2}[rotary_mode]
        rank = query.ndim
        head_axis = cls._HEAD_AXIS[layout]
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

    @classmethod
    def fake(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layout: int = 1,
        rotary_mode: str = "half",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate metadata and create shape-preserving abstract outputs."""
        cls._validate(query, key, cos, sin, layout, rotary_mode, check_values=False)
        return torch.empty_like(query), torch.empty_like(key)

    @staticmethod
    @symbolic_helper.parse_args("v", "v", "v", "v", "i", "s")
    def onnx(
        graph: Any,
        query: Any,
        key: Any,
        cos: Any,
        sin: Any,
        layout: int = 1,
        rotary_mode: str = "half",
    ) -> Any:
        """Emit the opset-18 MDC node after narrowing attributes."""
        if layout not in ApplyRotaryPosEmb._LAYOUT_RANK:
            raise RuntimeError("ONNX export requires layout in {1, 2, 3, 4}")
        if rotary_mode not in ApplyRotaryPosEmb._ROTARY_MODES:
            raise RuntimeError(
                "ONNX export requires rotary_mode in {'half', 'interleave', 'quarter'}"
            )
        return graph.op(
            "ApplyRotaryPosEmb",
            query,
            key,
            cos,
            sin,
            layout_i=layout,
            rotary_mode_s=rotary_mode,
            outputs=2,
        )
