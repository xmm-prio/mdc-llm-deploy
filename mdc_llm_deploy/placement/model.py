"""Internal tensor placement and alias contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from ..errors import MdcDeployError, UnsupportedPatternError

TensorKind = Literal["parameter", "buffer"]
_SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda", "npu"})


class PlacementError(MdcDeployError):
    """Raised when tensor placement or alias invariants are violated."""


@dataclass(frozen=True, slots=True)
class TensorPlacement:
    """Placement metadata for one parameter or buffer name."""

    fqn: str
    kind: TensorKind
    device: torch.device
    dtype: torch.dtype
    persistent: bool


@dataclass(frozen=True, slots=True)
class PlacementSnapshot:
    """Immutable model placement, device-map, and alias description."""

    tensors: tuple[TensorPlacement, ...]
    alias_groups: tuple[tuple[str, ...], ...]
    hf_device_map: tuple[tuple[str, str], ...]

    @property
    def by_fqn(self) -> dict[str, TensorPlacement]:
        """Return tensor metadata indexed by fully qualified name."""
        return {item.fqn: item for item in self.tensors}


def _owner_and_local_name(model: nn.Module, fqn: str) -> tuple[nn.Module, str]:
    owner_name, separator, local_name = fqn.rpartition(".")
    owner = model.get_submodule(owner_name) if separator else model
    return owner, local_name


def _buffer_is_persistent(model: nn.Module, fqn: str) -> bool:
    owner, local_name = _owner_and_local_name(model, fqn)
    return local_name not in owner._non_persistent_buffers_set


def _format_device_map(value: object) -> str:
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return f"cuda:{value}"
    return str(value)


def _iter_hooks(model: nn.Module) -> Iterable[tuple[str, object]]:
    for fqn, module in model.named_modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is not None:
            yield fqn, hook


def _offload_hook_name(hook: object) -> str | None:
    hook_name = type(hook).__name__
    if bool(getattr(hook, "offload", False)) or hook_name in {
        "CpuOffload",
        "CpuOffloadWithHook",
    }:
        return hook_name
    nested_hooks = getattr(hook, "hooks", ())
    if isinstance(nested_hooks, (tuple, list)):
        for nested_hook in nested_hooks:
            nested_name = _offload_hook_name(nested_hook)
            if nested_name is not None:
                return nested_name
    return None


def reject_dynamic_offload(model: nn.Module) -> None:
    """Reject disk placement and hooks that page weights during execution."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None:
        if not isinstance(device_map, Mapping):
            raise UnsupportedPatternError("hf_device_map must be a mapping")
        disk_names = [
            str(name)
            for name, device in device_map.items()
            if _format_device_map(device).lower() == "disk"
        ]
        if disk_names:
            raise UnsupportedPatternError(
                f"Disk offload is not supported: {sorted(disk_names)}"
            )

    for fqn, hook in _iter_hooks(model):
        hook_name = _offload_hook_name(hook)
        if hook_name is not None:
            location = fqn or "<root>"
            raise UnsupportedPatternError(
                f"Dynamic weight offload is not supported at {location}: "
                f"{hook_name}"
            )


def _validate_resident_device(fqn: str, tensor: Tensor) -> None:
    if tensor.device.type not in _SUPPORTED_DEVICE_TYPES:
        raise UnsupportedPatternError(
            f"Tensor {fqn!r} uses unsupported device {tensor.device}"
        )


def capture_placement(model: nn.Module) -> PlacementSnapshot:
    """Capture duplicate names so tied tensor aliases remain observable."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    reject_dynamic_offload(model)

    placements: list[TensorPlacement] = []
    identities: dict[int, list[str]] = {}
    for fqn, parameter in model.named_parameters(remove_duplicate=False):
        _validate_resident_device(fqn, parameter)
        placements.append(
            TensorPlacement(
                fqn=fqn,
                kind="parameter",
                device=parameter.device,
                dtype=parameter.dtype,
                persistent=True,
            )
        )
        identities.setdefault(id(parameter), []).append(fqn)
    for fqn, buffer in model.named_buffers(remove_duplicate=False):
        _validate_resident_device(fqn, buffer)
        placements.append(
            TensorPlacement(
                fqn=fqn,
                kind="buffer",
                device=buffer.device,
                dtype=buffer.dtype,
                persistent=_buffer_is_persistent(model, fqn),
            )
        )
        identities.setdefault(id(buffer), []).append(fqn)

    aliases = tuple(
        sorted(
            (
                tuple(sorted(names))
                for names in identities.values()
                if len(names) > 1
            ),
            key=lambda names: names[0],
        )
    )
    raw_device_map = getattr(model, "hf_device_map", {})
    device_map = tuple(
        sorted(
            (
                (str(name), _format_device_map(device))
                for name, device in raw_device_map.items()
            ),
            key=lambda item: item[0],
        )
    )
    return PlacementSnapshot(
        tensors=tuple(sorted(placements, key=lambda item: item.fqn)),
        alias_groups=aliases,
        hf_device_map=device_map,
    )


def validate_placement_preserved(
    before: PlacementSnapshot,
    after: PlacementSnapshot,
) -> None:
    """Validate exact tensor metadata, aliases, and device-map preservation."""
    before_by_fqn = before.by_fqn
    after_by_fqn = after.by_fqn
    if not before_by_fqn.keys() <= after_by_fqn.keys():
        missing = sorted(before_by_fqn.keys() - after_by_fqn.keys())
        raise PlacementError(f"Tensor names changed; missing={missing}")
    changed = [
        fqn
        for fqn, placement in before_by_fqn.items()
        if placement != after_by_fqn[fqn]
    ]
    if changed:
        raise PlacementError(f"Tensor placement changed: {sorted(changed)}")
    original_names = before_by_fqn.keys()
    projected_aliases = tuple(
        sorted(
            (
                tuple(name for name in group if name in original_names)
                for group in after.alias_groups
                if sum(name in original_names for name in group) > 1
            ),
            key=lambda names: names[0],
        )
    )
    if before.alias_groups != projected_aliases:
        raise PlacementError(
            "Tensor alias groups changed: "
            f"{before.alias_groups!r} != {projected_aliases!r}"
        )
    if before.hf_device_map != after.hf_device_map:
        raise PlacementError(
            "hf_device_map changed: "
            f"{before.hf_device_map!r} != {after.hf_device_map!r}"
        )


def _module_devices(module: nn.Module) -> set[torch.device]:
    tensors: list[Tensor] = list(module.parameters(recurse=False))
    tensors.extend(module.buffers(recurse=False))
    return {tensor.device for tensor in tensors}


def _one_device(devices: set[torch.device], source: str) -> torch.device | None:
    if len(devices) > 1:
        values = sorted(str(device) for device in devices)
        raise PlacementError(f"Conflicting {source} devices: {values}")
    return next(iter(devices), None)


def inherit_device(
    *,
    sources: Iterable[Tensor] = (),
    replaced_module: nn.Module | None = None,
    parent_modules: Iterable[nn.Module] = (),
) -> torch.device:
    """Resolve source, replaced-module, then nearest-parent device priority."""
    source_device = _one_device(
        {tensor.device for tensor in sources},
        "source tensor",
    )
    if source_device is not None:
        return source_device
    if replaced_module is not None:
        module_device = _one_device(
            _module_devices(replaced_module),
            "replaced module",
        )
        if module_device is not None:
            return module_device
    for parent in parent_modules:
        parent_device = _one_device(
            _module_devices(parent),
            "parent module",
        )
        if parent_device is not None:
            return parent_device
    raise PlacementError("Cannot infer device for new tensor")


__all__ = [
    "PlacementError",
    "PlacementSnapshot",
    "TensorPlacement",
    "capture_placement",
    "inherit_device",
    "reject_dynamic_offload",
    "validate_placement_preserved",
]
