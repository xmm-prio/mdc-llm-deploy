"""Recursive tensor device migration."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, cast

import torch
from torch import Tensor

_ATOMIC_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    torch.device,
    torch.dtype,
)


class _MoveContext:
    def __init__(self, device: torch.device, non_blocking: bool) -> None:
        self.device = device
        self.non_blocking = non_blocking
        self.memo: dict[int, Any] = {}

    def move(self, value: Any, path: str) -> Any:
        if isinstance(value, _ATOMIC_TYPES):
            return value

        identity = id(value)
        if identity in self.memo:
            return self.memo[identity]

        if isinstance(value, Tensor):
            moved = value.to(device=self.device, non_blocking=self.non_blocking)
            self.memo[identity] = moved
            return moved
        if type(value) is list:
            result: list[Any] = []
            self.memo[identity] = result
            result.extend(
                self.move(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            )
            return result
        if type(value) is dict:
            result_dict: dict[Any, Any] = {}
            self.memo[identity] = result_dict
            for key, item in value.items():
                result_dict[key] = self.move(item, f"{path}[{key!r}]")
            return result_dict
        if type(value) is set:
            result_set: set[Any] = set()
            self.memo[identity] = result_set
            for index, item in enumerate(value):
                result_set.add(self.move(item, f"{path}{{{index}}}"))
            return result_set
        if is_dataclass(value) and not isinstance(value, type):
            result_dataclass = object.__new__(type(value))
            self.memo[identity] = result_dataclass
            for field in fields(value):
                object.__setattr__(
                    result_dataclass,
                    field.name,
                    self.move(
                        getattr(value, field.name),
                        f"{path}.{field.name}",
                    ),
                )
            return result_dataclass
        if isinstance(value, tuple) and hasattr(type(value), "_fields"):
            namedtuple_type = cast(Any, type(value))
            field_names = namedtuple_type._fields
            moved_items = [
                self.move(item, f"{path}.{name}")
                for name, item in zip(field_names, value, strict=True)
            ]
            if identity in self.memo:
                return self.memo[identity]
            result_namedtuple = namedtuple_type(*moved_items)
            self.memo[identity] = result_namedtuple
            return result_namedtuple
        if type(value) is tuple:
            tuple_items = tuple(
                self.move(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            )
            if identity in self.memo:
                return self.memo[identity]
            result_tuple = tuple(tuple_items)
            self.memo[identity] = result_tuple
            return result_tuple
        if type(value) is frozenset:
            result_frozenset = frozenset(
                self.move(item, f"{path}{{{index}}}")
                for index, item in enumerate(value)
            )
            self.memo[identity] = result_frozenset
            return result_frozenset
        raise TypeError(
            f"Unsupported value at {path}: {type(value).__qualname__}"
        )


def move_to_device(
    value: Any,
    device: torch.device | str | int,
    *,
    non_blocking: bool = False,
) -> Any:
    """Move every nested tensor to one device while preserving structure."""
    target = torch.device(device)
    return _MoveContext(target, non_blocking).move(value, "$")


__all__ = ["move_to_device"]
