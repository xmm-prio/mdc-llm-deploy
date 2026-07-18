"""Central artifact input/output ABI contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...errors import GraphStateError
from .types import FusionBoundary, GraphMetadata, TensorAbi

SAVE_KV_CACHE_PROPERTY = "save_kv_cache"
_LAYER_FQN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True, slots=True)
class ArtifactIoAbi:
    """Ordered public input/output ABI for one exported artifact."""

    inputs: tuple[TensorAbi, ...]
    outputs: tuple[TensorAbi, ...]
    layer_count: int
    save_kv_cache: bool


def resolve_save_kv_cache(properties: Mapping[str, Any]) -> bool:
    """Resolve KV publication policy with legacy metadata compatibility."""
    value = properties.get(SAVE_KV_CACHE_PROPERTY, False)
    if type(value) is not bool:
        raise GraphStateError("save_kv_cache metadata must be a bool")
    return value


def _explicit_layer_id(boundary: FusionBoundary) -> int | None:
    matches = _LAYER_FQN.findall(boundary.fqn)
    if len(matches) > 1:
        raise GraphStateError(
            f"Attention boundary has ambiguous layer FQN: {boundary.fqn!r}"
        )
    return int(matches[0]) if matches else None


def order_attention_boundaries(
    boundaries: tuple[FusionBoundary, ...],
) -> tuple[FusionBoundary, ...]:
    """Order attention boundaries by explicit numeric model layer."""
    attention = tuple(item for item in boundaries if item.kind == "attention")
    if not attention:
        return ()
    layer_ids = tuple(_explicit_layer_id(item) for item in attention)
    if len(attention) == 1 and layer_ids == (None,):
        return attention
    if any(layer_id is None for layer_id in layer_ids):
        raise GraphStateError(
            "Multiple attention boundaries require explicit layers.N FQNs"
        )
    numeric_ids = tuple(layer_id for layer_id in layer_ids if layer_id is not None)
    expected = tuple(range(len(attention)))
    if tuple(sorted(numeric_ids)) != expected:
        raise GraphStateError(
            "Attention layer IDs must be unique and contiguous from zero"
        )
    return tuple(
        boundary
        for _, boundary in sorted(
            zip(numeric_ids, attention, strict=True),
            key=lambda item: item[0],
        )
    )


def boundary_sort_key(boundary: FusionBoundary) -> tuple[str, int, int, str]:
    """Return deterministic boundary order with numeric attention layers."""
    if boundary.kind != "attention":
        return (boundary.kind, 1, 0, boundary.fqn)
    layer_id = _explicit_layer_id(boundary)
    return (boundary.kind, 0, layer_id if layer_id is not None else 0, boundary.fqn)


def _validate_cache_pairs(
    entries: tuple[TensorAbi, ...],
    *,
    prefix: str,
    layer_count: int,
) -> None:
    expected_names = tuple(
        f"{prefix}.{layer_id}.{edge}"
        for layer_id in range(layer_count)
        for edge in ("key", "value")
    )
    actual_names = tuple(item.name for item in entries)
    if actual_names != expected_names:
        raise GraphStateError(
            f"{prefix} KV ABI must contain ordered contiguous key/value pairs"
        )
    for layer_id in range(layer_count):
        key, value = entries[layer_id * 2 : layer_id * 2 + 2]
        if key.shape != value.shape:
            raise GraphStateError(
                f"{prefix} layer {layer_id} key/value shapes must match"
            )


def _generic_artifact_abi(value: GraphMetadata) -> ArtifactIoAbi | None:
    if (
        value.output_abi
        and value.output_abi[0].name != "logits"
        and SAVE_KV_CACHE_PROPERTY not in value.properties
        and not any(item.kind == "attention" for item in value.boundaries)
    ):
        return ArtifactIoAbi(
            inputs=value.input_abi,
            outputs=value.output_abi,
            layer_count=0,
            save_kv_cache=False,
        )
    return None


def derive_artifact_io_abi(value: GraphMetadata) -> ArtifactIoAbi:
    """Derive and validate public artifact ABI from internal graph metadata."""
    generic = _generic_artifact_abi(value)
    if generic is not None:
        return generic
    if not value.output_abi or value.output_abi[0].name != "logits":
        raise GraphStateError("Internal output ABI must start with logits")
    cache_outputs = value.output_abi[1:]
    if len(cache_outputs) % 2:
        raise GraphStateError("Internal KV outputs must contain key/value pairs")
    layer_count = len(cache_outputs) // 2
    _validate_cache_pairs(
        cache_outputs,
        prefix="present",
        layer_count=layer_count,
    )

    attention = order_attention_boundaries(value.boundaries)
    if len(attention) != layer_count:
        raise GraphStateError(
            "Attention boundary count must match present KV layer count"
        )

    if not value.input_abi or value.input_abi[0].name != "input_ids":
        raise GraphStateError("Internal input ABI must start with input_ids")
    cache_inputs = value.input_abi[1:]
    if value.stage.is_prefill:
        if cache_inputs:
            raise GraphStateError("Prefill input ABI must not contain KV cache")
    else:
        if len(cache_inputs) != layer_count * 2:
            raise GraphStateError(
                "Decode past KV layer count must match present KV layer count"
            )
        _validate_cache_pairs(
            cache_inputs,
            prefix="past",
            layer_count=layer_count,
        )

    save_kv_cache = resolve_save_kv_cache(value.properties)
    outputs = value.output_abi if save_kv_cache else value.output_abi[:1]
    return ArtifactIoAbi(
        inputs=value.input_abi,
        outputs=outputs,
        layer_count=layer_count,
        save_kv_cache=save_kv_cache,
    )


__all__ = [
    "SAVE_KV_CACHE_PROPERTY",
    "ArtifactIoAbi",
    "boundary_sort_key",
    "derive_artifact_io_abi",
    "order_attention_boundaries",
    "resolve_save_kv_cache",
]
