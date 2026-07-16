"""Graph input placement contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from .graph_types import GraphMetadata

INPUT_DEVICES_PROPERTY = "input_devices"
_SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda", "npu"})


def capture_input_devices(inputs: Mapping[str, Tensor]) -> dict[str, str]:
    """Capture one stable device string for every graph input."""
    return {name: str(tensor.device) for name, tensor in inputs.items()}


def resolve_input_devices(metadata: GraphMetadata) -> tuple[torch.device, ...]:
    """Resolve and validate input devices in ABI order."""
    raw_contract = metadata.properties.get(INPUT_DEVICES_PROPERTY)
    if not isinstance(raw_contract, Mapping):
        raise ValueError("Input device contract is missing or is not a mapping")
    if not all(isinstance(name, str) for name in raw_contract):
        raise ValueError("Input device contract names must be strings")

    input_names = tuple(item.name for item in metadata.input_abi)
    contract_names = tuple(raw_contract)
    missing = sorted(set(input_names) - set(contract_names))
    unexpected = sorted(set(contract_names) - set(input_names))
    if missing or unexpected or len(raw_contract) != len(input_names):
        raise ValueError(
            "Input device contract does not match graph inputs: "
            f"missing={missing}, unexpected={unexpected}"
        )

    devices: list[torch.device] = []
    for name in input_names:
        raw_device: Any = raw_contract[name]
        if not isinstance(raw_device, str):
            raise ValueError(
                f"Input device for {name!r} must be a string, got "
                f"{type(raw_device).__name__}"
            )
        try:
            device = torch.device(raw_device)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                f"Input device for {name!r} is invalid: {raw_device!r}"
            ) from error
        if device.type not in _SUPPORTED_DEVICE_TYPES:
            raise ValueError(
                f"Input device for {name!r} is unsupported: {device}"
            )
        devices.append(device)
    return tuple(devices)


__all__ = [
    "INPUT_DEVICES_PROPERTY",
    "capture_input_devices",
    "resolve_input_devices",
]
