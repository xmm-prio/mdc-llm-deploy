"""Immutable snapshots for JSON-like metadata values."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

from ...errors import GraphStateError


class FrozenJsonMapping(Mapping[str, Any]):
    """Immutable, value-equal mapping for frozen JSON-like data."""

    __slots__ = ("_index", "_items")
    _items: tuple[tuple[str, Any], ...]
    _index: Mapping[str, Any] | None

    def __init__(self, items: tuple[tuple[str, Any], ...]) -> None:
        object.__setattr__(self, "_items", items)
        object.__setattr__(self, "_index", None)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__!s} is immutable")

    def __getitem__(self, key: str) -> Any:
        if type(key) is not str:
            return self._linear_lookup(key)

        index = self._index
        if index is None:
            candidate_index: dict[str, Any] = {}
            for candidate, value in self._items:
                if type(candidate) is not str:
                    return self._linear_lookup(key)
                candidate_index.setdefault(candidate, value)
            index = MappingProxyType(candidate_index)
            object.__setattr__(self, "_index", index)
        return index[key]

    def _linear_lookup(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __getstate__(self) -> tuple[None, dict[str, Any]]:
        return None, {"_items": self._items, "_index": None}

    def __deepcopy__(
        self,
        memo: dict[int, Any],
    ) -> FrozenJsonMapping:
        memo[id(self)] = self
        return self


def freeze_json(value: Any) -> Any:
    """Capture JSON containers as a deeply immutable snapshot."""
    try:
        return _freeze_json(value, set())
    except RecursionError as error:
        raise GraphStateError(
            "properties must not contain circular references"
        ) from error


def validate_json_mapping(properties: Mapping[str, Any]) -> None:
    """Validate a string-keyed mapping as finite JSON-like data."""
    if not isinstance(properties, Mapping) or not all(
        isinstance(key, str) and key for key in properties
    ):
        raise GraphStateError(
            "properties must be a string-keyed dictionary"
        )

    active: set[int] = set()

    def visit(item: Any) -> bool:
        if item is None or isinstance(item, (str, bool, int)):
            return True
        if isinstance(item, float):
            return math.isfinite(item)
        if isinstance(item, tuple):
            identity = id(item)
            if identity in active:
                return False
            active.add(identity)
            try:
                return all(visit(value) for value in item)
            finally:
                active.remove(identity)
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                return False
            active.add(identity)
            try:
                return all(
                    isinstance(key, str) and key and visit(value)
                    for key, value in item.items()
                )
            finally:
                active.remove(identity)
        return False

    try:
        valid = visit(properties)
    except RecursionError:
        valid = False
    if not valid:
        raise GraphStateError(
            "properties must contain finite JSON-compatible values"
        )


def _freeze_json(value: Any, active: set[int]) -> Any:
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise GraphStateError(
                "properties must not contain circular references"
            )
        active.add(identity)
        try:
            return FrozenJsonMapping(
                tuple(
                    (key, _freeze_json(item, active))
                    for key, item in value.items()
                )
            )
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise GraphStateError(
                "properties must not contain circular references"
            )
        active.add(identity)
        try:
            return tuple(_freeze_json(item, active) for item in value)
        finally:
            active.remove(identity)
    return value
