"""Packed expert-major mixture-of-experts inference operator."""

from __future__ import annotations

import importlib
from typing import Any, ClassVar

import torch

from .base import CustomOp

tl: Any = None


class MoeExpert(CustomOp):
    """Execute routed SwiGLU experts stored in one expert-major packed tensor."""

    qualified_name = "mdc_llm_deploy::moe_expert"
    schema = (
        "(Tensor x, Tensor topk_ids, Tensor topk_weight, Tensor expert_weights, "
        "Tensor? quant_scales=None, Tensor? quant_offsets=None) -> Tensor"
    )
    _triton_kernels: ClassVar[tuple[Any, Any, Any, Any] | None] = None

    @staticmethod
    def _dimensions(
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
        quant_scales: torch.Tensor | None,
        quant_offsets: torch.Tensor | None,
    ) -> tuple[int, int, int, int, int]:
        tensors = [x, topk_ids, topk_weight, expert_weights]
        if quant_scales is not None:
            tensors.append(quant_scales)
        if quant_offsets is not None:
            tensors.append(quant_offsets)
        if any(tensor.device != x.device for tensor in tensors):
            raise ValueError("MoeExpert inputs must be on the same device")

        if x.ndim != 2:
            raise ValueError("x must have shape [token_count, hidden_size]")
        if not x.dtype.is_floating_point:
            raise TypeError("x must have a floating-point dtype")
        if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("topk_ids must be an INT32 or INT64 rank-2 tensor")
        if topk_weight.ndim != 2 or topk_weight.dtype != x.dtype:
            raise TypeError("topk_weight must be rank-2 and have the same dtype as x")
        if topk_ids.shape != topk_weight.shape:
            raise ValueError("topk_ids and topk_weight must have the same shape")
        if topk_ids.shape[0] != x.shape[0]:
            raise ValueError("routing token count must match x")
        if topk_ids.shape[1] <= 0:
            raise ValueError("top_k must be positive")
        if expert_weights.ndim != 2 or expert_weights.shape[0] <= 0:
            raise ValueError("expert_weights must have shape [expert_count, packed_width]")

        token_count, hidden_size = x.shape
        expert_count, packed_width = expert_weights.shape
        divisor = 3 * hidden_size
        if hidden_size <= 0 or packed_width <= 0 or packed_width % divisor != 0:
            raise ValueError("expert_weights packed width must equal 3 * hidden_size * intermediate_size")
        intermediate_size = packed_width // divisor
        top_k = topk_ids.shape[1]

        if expert_weights.dtype == torch.int8:
            if quant_scales is None:
                raise ValueError("INT8 expert_weights require quant_scales")
            expected_shape = (expert_count, 2 * intermediate_size + hidden_size)
            if quant_scales.shape != expected_shape or not quant_scales.dtype.is_floating_point:
                raise ValueError(
                    "quant_scales must be floating-point with shape "
                    "[expert_count, 2 * intermediate_size + hidden_size]"
                )
            if quant_offsets is not None and (
                quant_offsets.shape != expected_shape
                or not quant_offsets.dtype.is_floating_point
            ):
                raise ValueError("quant_offsets must match quant_scales shape and be floating-point")
        elif expert_weights.dtype.is_floating_point:
            if expert_weights.dtype != x.dtype:
                raise TypeError("floating expert_weights must have the same dtype as x")
            if quant_scales is not None or quant_offsets is not None:
                raise ValueError("floating expert_weights must not use quantization parameters")
        else:
            raise TypeError("expert_weights must be floating-point or INT8")

        return (
            int(token_count),
            int(hidden_size),
            int(top_k),
            int(expert_count),
            int(intermediate_size),
        )

    @staticmethod
    def _validate_routing(
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_count: int,
    ) -> None:
        if bool(torch.any((topk_ids < 0) | (topk_ids >= expert_count)).item()):
            raise ValueError("topk_ids contains an out-of-range expert id")
        if topk_ids.shape[1] > 1:
            sorted_ids = torch.sort(topk_ids, dim=1).values
            if bool(torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]).item()):
                raise ValueError("topk_ids must not repeat an expert for one token")
        weights = topk_weight.float()
        if not bool(torch.all(torch.isfinite(weights)).item()):
            raise ValueError("topk_weight must contain only finite values")
        if bool(torch.any(weights < 0).item()):
            raise ValueError("topk_weight must be non-negative")
        if not bool(
            torch.allclose(
                weights.sum(dim=1),
                torch.ones(weights.shape[0], device=weights.device),
                rtol=1e-4,
                atol=1e-5,
            )
        ):
            raise ValueError("each topk_weight row must sum to one")

    @staticmethod
    def _unpack_weights(
        expert_weights: torch.Tensor,
        quant_scales: torch.Tensor | None,
        quant_offsets: torch.Tensor | None,
        hidden_size: int,
        intermediate_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = expert_weights.float()
        if expert_weights.dtype == torch.int8:
            assert quant_scales is not None
            scales = quant_scales.float()
            offsets = torch.zeros_like(scales) if quant_offsets is None else quant_offsets.float()
            gate_scale, up_scale, down_scale = torch.split(
                scales, [intermediate_size, intermediate_size, hidden_size], dim=1
            )
            gate_offset, up_offset, down_offset = torch.split(
                offsets, [intermediate_size, intermediate_size, hidden_size], dim=1
            )
        gate_end = hidden_size * intermediate_size
        up_end = 2 * gate_end
        gate = weights[:, :gate_end].reshape(-1, intermediate_size, hidden_size)
        up = weights[:, gate_end:up_end].reshape(-1, intermediate_size, hidden_size)
        down = weights[:, up_end:].reshape(-1, hidden_size, intermediate_size)
        if expert_weights.dtype == torch.int8:
            gate = (gate - gate_offset.unsqueeze(-1)) * gate_scale.unsqueeze(-1)
            up = (up - up_offset.unsqueeze(-1)) * up_scale.unsqueeze(-1)
            down = (down - down_offset.unsqueeze(-1)) * down_scale.unsqueeze(-1)
        return gate, up, down

    @staticmethod
    def cpu(
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
        quant_scales: torch.Tensor | None = None,
        quant_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run routed experts with FP32 accumulation on CPU."""
        token_count, hidden_size, top_k, expert_count, intermediate_size = (
            MoeExpert._dimensions(
                x, topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets
            )
        )
        if x.device.type != "cpu":
            raise ValueError("MoeExpert.cpu requires CPU tensors")
        MoeExpert._validate_routing(topk_ids, topk_weight, expert_count)
        gate, up, down = MoeExpert._unpack_weights(
            expert_weights, quant_scales, quant_offsets, hidden_size, intermediate_size
        )

        x_fp32 = x.float()
        output = torch.zeros((token_count, hidden_size), dtype=torch.float32, device=x.device)
        for route in range(top_k):
            expert_ids = topk_ids[:, route].long()
            selected_gate = gate.index_select(0, expert_ids)
            selected_up = up.index_select(0, expert_ids)
            selected_down = down.index_select(0, expert_ids)
            gate_output = torch.bmm(selected_gate, x_fp32.unsqueeze(-1)).squeeze(-1)
            up_output = torch.bmm(selected_up, x_fp32.unsqueeze(-1)).squeeze(-1)
            activated = torch.nn.functional.silu(gate_output) * up_output
            expert_output = torch.bmm(selected_down, activated.unsqueeze(-1)).squeeze(-1)
            output.add_(expert_output * topk_weight[:, route].float().unsqueeze(-1))
        return output.to(x.dtype)

    @classmethod
    def _load_triton_kernels(cls) -> tuple[Any, Any, Any, Any]:
        if cls._triton_kernels is not None:
            return cls._triton_kernels
        try:
            triton = importlib.import_module("triton")
            globals()["tl"] = importlib.import_module("triton.language")
        except ImportError as error:
            raise RuntimeError("MoeExpert CUDA execution requires Triton") from error

        def gate_up_kernel(  # type: ignore[no-untyped-def]
            x_ptr,
            ids_ptr,
            weights_ptr,
            scales_ptr,
            offsets_ptr,
            gate_ptr,
            up_ptr,
            hidden_size,
            intermediate_size,
            top_k,
            packed_width,
            scale_width,
            quantized,
            has_offsets,
            block_h,
        ):
            output_index = tl.program_id(0)
            route_index = output_index // intermediate_size
            intermediate_index = output_index % intermediate_size
            token_index = route_index // top_k
            expert_index = tl.load(ids_ptr + route_index)
            hidden_offsets = tl.arange(0, block_h)
            mask = hidden_offsets < hidden_size
            x_values = tl.load(x_ptr + token_index * hidden_size + hidden_offsets, mask=mask, other=0.0)
            expert_base = expert_index * packed_width
            gate_base = expert_base + intermediate_index * hidden_size
            up_base = expert_base + hidden_size * intermediate_size + intermediate_index * hidden_size
            gate_weights = tl.load(weights_ptr + gate_base + hidden_offsets, mask=mask, other=0.0)
            up_weights = tl.load(weights_ptr + up_base + hidden_offsets, mask=mask, other=0.0)
            if quantized:
                scale_base = expert_index * scale_width
                gate_scale = tl.load(scales_ptr + scale_base + intermediate_index)
                up_scale = tl.load(scales_ptr + scale_base + intermediate_size + intermediate_index)
                gate_offset = 0.0
                up_offset = 0.0
                if has_offsets:
                    gate_offset = tl.load(offsets_ptr + scale_base + intermediate_index)
                    up_offset = tl.load(
                        offsets_ptr + scale_base + intermediate_size + intermediate_index
                    )
                gate_weights = (gate_weights.to(tl.float32) - gate_offset) * gate_scale
                up_weights = (up_weights.to(tl.float32) - up_offset) * up_scale
            gate_value = tl.sum(x_values.to(tl.float32) * gate_weights.to(tl.float32), axis=0)
            up_value = tl.sum(x_values.to(tl.float32) * up_weights.to(tl.float32), axis=0)
            tl.store(gate_ptr + output_index, gate_value)
            tl.store(up_ptr + output_index, up_value)

        def swiglu_kernel(  # type: ignore[no-untyped-def]
            gate_ptr,
            up_ptr,
            activated_ptr,
            value_count,
            block,
        ):
            offsets = tl.program_id(0) * block + tl.arange(0, block)
            mask = offsets < value_count
            gate = tl.load(gate_ptr + offsets, mask=mask)
            up = tl.load(up_ptr + offsets, mask=mask)
            silu = gate * tl.sigmoid(gate)
            tl.store(activated_ptr + offsets, silu * up, mask=mask)

        def down_kernel(  # type: ignore[no-untyped-def]
            activated_ptr,
            ids_ptr,
            routing_ptr,
            weights_ptr,
            scales_ptr,
            offsets_ptr,
            output_ptr,
            hidden_size,
            intermediate_size,
            top_k,
            packed_width,
            scale_width,
            quantized,
            has_offsets,
            block_i,
        ):
            output_index = tl.program_id(0)
            token_index = output_index // hidden_size
            hidden_index = output_index % hidden_size
            intermediate_offsets = tl.arange(0, block_i)
            mask = intermediate_offsets < intermediate_size
            accumulator = 0.0
            for route in range(0, top_k):
                route_index = token_index * top_k + route
                expert_index = tl.load(ids_ptr + route_index)
                routing_weight = tl.load(routing_ptr + route_index).to(tl.float32)
                activation_base = route_index * intermediate_size
                activation = tl.load(
                    activated_ptr + activation_base + intermediate_offsets, mask=mask, other=0.0
                )
                expert_base = expert_index * packed_width
                down_base = (
                    expert_base
                    + 2 * hidden_size * intermediate_size
                    + hidden_index * intermediate_size
                )
                down_weights = tl.load(
                    weights_ptr + down_base + intermediate_offsets, mask=mask, other=0.0
                )
                if quantized:
                    scale_base = expert_index * scale_width + 2 * intermediate_size
                    down_scale = tl.load(scales_ptr + scale_base + hidden_index)
                    down_offset = 0.0
                    if has_offsets:
                        down_offset = tl.load(offsets_ptr + scale_base + hidden_index)
                    down_weights = (
                        down_weights.to(tl.float32) - down_offset
                    ) * down_scale
                accumulator += routing_weight * tl.sum(
                    activation.to(tl.float32) * down_weights.to(tl.float32), axis=0
                )
            tl.store(output_ptr + output_index, accumulator)

        gate_up_kernel.__annotations__.update(
            {
                "top_k": tl.constexpr,
                "quantized": tl.constexpr,
                "has_offsets": tl.constexpr,
                "block_h": tl.constexpr,
            }
        )
        swiglu_kernel.__annotations__["block"] = tl.constexpr
        down_kernel.__annotations__.update(
            {
                "top_k": tl.constexpr,
                "quantized": tl.constexpr,
                "has_offsets": tl.constexpr,
                "block_i": tl.constexpr,
            }
        )
        compiled_gate_up = triton.jit(gate_up_kernel)
        compiled_swiglu = triton.jit(swiglu_kernel)
        compiled_down = triton.jit(down_kernel)
        cls._triton_kernels = (
            triton,
            compiled_gate_up,
            compiled_swiglu,
            compiled_down,
        )
        return cls._triton_kernels

    @staticmethod
    def cuda(
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
        quant_scales: torch.Tensor | None = None,
        quant_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run staged Triton routing, SwiGLU, and aggregation kernels."""
        token_count, hidden_size, top_k, expert_count, intermediate_size = (
            MoeExpert._dimensions(
                x, topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets
            )
        )
        if x.device.type != "cuda":
            raise ValueError("MoeExpert.cuda requires CUDA tensors")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("MoeExpert CUDA supports FLOAT16, BFLOAT16, and FLOAT32 x")
        MoeExpert._validate_routing(topk_ids, topk_weight, expert_count)
        triton, gate_up_kernel, swiglu_kernel, down_kernel = MoeExpert._load_triton_kernels()

        x_contiguous = x.contiguous()
        ids_contiguous = topk_ids.contiguous()
        routing_contiguous = topk_weight.contiguous()
        weights_contiguous = expert_weights.contiguous()
        quantized = expert_weights.dtype == torch.int8
        scales = weights_contiguous if quant_scales is None else quant_scales.contiguous()
        offsets = weights_contiguous if quant_offsets is None else quant_offsets.contiguous()
        route_count = token_count * top_k
        value_count = route_count * intermediate_size
        gate = torch.empty(value_count, device=x.device, dtype=torch.float32)
        up = torch.empty_like(gate)
        activated = torch.empty_like(gate)
        output = torch.empty((token_count, hidden_size), device=x.device, dtype=x.dtype)
        block_h = triton.next_power_of_2(hidden_size)
        block_i = triton.next_power_of_2(intermediate_size)
        if block_h > 65536 or block_i > 65536:
            raise ValueError("MoeExpert CUDA dimensions exceed Triton kernel limits")

        gate_up_kernel[(value_count,)](
            x_contiguous,
            ids_contiguous,
            weights_contiguous,
            scales,
            offsets,
            gate,
            up,
            hidden_size,
            intermediate_size,
            top_k,
            expert_weights.shape[1],
            2 * intermediate_size + hidden_size,
            quantized=quantized,
            has_offsets=quant_offsets is not None,
            block_h=block_h,
        )
        swiglu_kernel[(triton.cdiv(value_count, 256),)](
            gate, up, activated, value_count, block=256
        )
        down_kernel[(token_count * hidden_size,)](
            activated,
            ids_contiguous,
            routing_contiguous,
            weights_contiguous,
            scales,
            offsets,
            output,
            hidden_size,
            intermediate_size,
            top_k,
            expert_weights.shape[1],
            2 * intermediate_size + hidden_size,
            quantized=quantized,
            has_offsets=quant_offsets is not None,
            block_i=block_i,
        )
        return output

    @staticmethod
    def fake(
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
        expert_weights: torch.Tensor,
        quant_scales: torch.Tensor | None = None,
        quant_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Validate metadata and return output metadata matching x."""
        MoeExpert._dimensions(
            x, topk_ids, topk_weight, expert_weights, quant_scales, quant_offsets
        )
        return torch.empty_like(x)

    @staticmethod
    def onnx(
        graph: Any,
        x: Any,
        topk_ids: Any,
        topk_weight: Any,
        expert_weights: Any,
        quant_scales: Any = None,
        quant_offsets: Any = None,
    ) -> Any:
        """Emit MoeExpert while retaining both optional quantization input slots."""
        return graph.op(
            "MoeExpert",
            x,
            topk_ids,
            topk_weight,
            expert_weights,
            quant_scales,
            quant_offsets,
        )
