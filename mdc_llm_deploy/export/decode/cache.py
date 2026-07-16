"""KV-cache construction for static decode FX graphs."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.fx import GraphModule, Node

from ...errors import UnsupportedPatternError
from ...graph.fx.inspection import node_target
from ...graph.metadata import GraphMetadata, QuantizedTarget


def cache_target(
    value: GraphMetadata,
    edge: str,
    layer_id: int | None = None,
) -> QuantizedTarget | None:
    """Return the unique quantization target for one cache edge."""
    matches = [
        target
        for target in value.quantized_targets
        if target.target_type == "attention"
        and target.fqn.rsplit(".", 1)[-1] == edge
        and (
            layer_id is None
            or f".layers.{layer_id}." in f".{target.fqn}."
        )
    ]
    if len(matches) > 1:
        raise UnsupportedPatternError(
            f"Decode conversion found multiple attention {edge} targets"
        )
    return matches[0] if matches else None


def _register_tensor(
    candidate: GraphModule,
    name: str,
    value: Tensor,
) -> str:
    candidate.register_buffer(name, value, persistent=True)
    return name


def _qparam_tensor(
    target: QuantizedTarget,
    sequence: int,
    *,
    current: bool,
    zero_point: bool,
    device: torch.device,
) -> Tensor:
    raw = target.zero_point if zero_point else target.scale
    dtype = torch.float32
    if len(raw) == 1:
        return torch.tensor(raw[0], dtype=dtype, device=device)
    if len(raw) != sequence:
        raise UnsupportedPatternError(
            f"Decode {target.fqn} parameters must be scalar "
            f"or have {sequence} positions"
        )
    if current:
        return torch.tensor(raw[-1], dtype=dtype, device=device)
    return torch.tensor(raw, dtype=dtype, device=device).reshape(
        1,
        1,
        sequence,
        1,
    )


def _node_tensor(node: Node, edge: str) -> Tensor:
    value = node.meta.get("val")
    if not isinstance(value, Tensor):
        raise UnsupportedPatternError(
            f"Decode cache cannot infer device for {edge!r}"
        )
    return value


def _set_cache_placeholder_metadata(
    past: Node,
    current: Tensor,
    target: QuantizedTarget | None,
    sequence: int,
) -> None:
    if current.ndim < 3:
        raise UnsupportedPatternError(
            "Decode cache source must have at least three dimensions"
        )
    shape = list(current.shape)
    shape[2] = sequence - 1
    dtype = torch.int8 if target is not None else current.dtype
    past.meta["val"] = current.new_empty(shape, dtype=dtype)


def insert_cache_quantization(
    candidate: GraphModule,
    current: Node,
    past: Node,
    target: QuantizedTarget | None,
    edge: str,
    sequence: int,
) -> tuple[Node, Node]:
    """Append current KV state and return stored and attention values."""
    graph = candidate.graph
    source_value = _node_tensor(current, edge)
    _set_cache_placeholder_metadata(past, source_value, target, sequence)
    if target is None:
        present = graph.call_function(
            torch.ops.aten.cat.default,
            args=([past, current], 2),
        )
        return present, present
    if target.bits != 8:
        raise UnsupportedPatternError(
            "Decode cache supports only float or INT8"
        )
    scale_name = _register_tensor(
        candidate,
        f"_mdc_{edge}_current_scale",
        _qparam_tensor(
            target,
            sequence,
            current=True,
            zero_point=False,
            device=source_value.device,
        ),
    )
    zero_name = _register_tensor(
        candidate,
        f"_mdc_{edge}_current_zero_point",
        _qparam_tensor(
            target,
            sequence,
            current=True,
            zero_point=True,
            device=source_value.device,
        ),
    )
    full_scale_name = _register_tensor(
        candidate,
        f"_mdc_{edge}_cache_scale",
        _qparam_tensor(
            target,
            sequence,
            current=False,
            zero_point=False,
            device=source_value.device,
        ),
    )
    full_zero_name = _register_tensor(
        candidate,
        f"_mdc_{edge}_cache_zero_point",
        _qparam_tensor(
            target,
            sequence,
            current=False,
            zero_point=True,
            device=source_value.device,
        ),
    )
    scale = graph.get_attr(scale_name)
    zero = graph.get_attr(zero_name)
    current_fp32 = graph.call_function(
        torch.ops.aten.to.dtype,
        args=(current, torch.float32),
    )
    divided = graph.call_function(
        torch.ops.aten.div.Tensor,
        args=(current_fp32, scale),
    )
    rounded = graph.call_function(
        torch.ops.aten.round.default,
        args=(divided,),
    )
    shifted = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(rounded, zero),
    )
    clamped = graph.call_function(
        torch.ops.aten.clamp.default,
        args=(shifted, -128, 127),
    )
    quantized = graph.call_function(
        torch.ops.aten.to.dtype,
        args=(clamped, torch.int8),
    )
    present = graph.call_function(
        torch.ops.aten.cat.default,
        args=([past, quantized], 2),
    )
    full_scale = graph.get_attr(full_scale_name)
    full_zero = graph.get_attr(full_zero_name)
    present_fp32 = graph.call_function(
        torch.ops.aten.to.dtype,
        args=(present, torch.float32),
    )
    centered = graph.call_function(
        torch.ops.aten.sub.Tensor,
        args=(present_fp32, full_zero),
    )
    dequantized = graph.call_function(
        torch.ops.aten.mul.Tensor,
        args=(centered, full_scale),
    )
    if source_value.dtype != torch.float32:
        dequantized = graph.call_function(
            torch.ops.aten.to.dtype,
            args=(dequantized, source_value.dtype),
        )
    return present, dequantized


def replace_attention_cache_users(
    current: Node,
    replacement: Node,
    output: Node,
) -> None:
    """Replace attention consumers while preserving graph outputs."""
    for user in tuple(current.users):
        if user is output:
            continue
        if "repeat_interleave" in node_target(user):
            user.replace_input_with(current, replacement)
