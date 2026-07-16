from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from mdc_llm_deploy.utils import move_to_device


@dataclass
class Node:
    value: Any
    next: Node | None = None


@dataclass(frozen=True, slots=True)
class FrozenPayload:
    tensor: torch.Tensor
    calculated: int = field(init=False, default=7)


def test_move_to_device_handles_supported_nested_values() -> None:
    point_type = namedtuple("Point", ("x", "y"))
    tensor = torch.ones(2, dtype=torch.float16)
    value = {
        "list": [tensor, None, "value"],
        "tuple": (tensor, 4),
        "namedtuple": point_type(tensor, True),
        "set": {1, 2},
        "frozenset": frozenset({3, 4}),
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }

    moved = move_to_device(value, "cpu")

    assert moved is not value
    assert moved["list"][0] is moved["tuple"][0]
    assert moved["list"][0] is moved["namedtuple"].x
    assert moved["list"][0].dtype is torch.float16
    assert moved["namedtuple"].y is True
    assert moved["set"] == {1, 2}
    assert moved["frozenset"] == frozenset({3, 4})
    assert moved["device"] == torch.device("cpu")
    assert moved["dtype"] is torch.float32


def test_move_to_device_rebuilds_frozen_slots_dataclass() -> None:
    payload = FrozenPayload(torch.ones(1))

    moved = move_to_device(payload, "cpu")

    assert isinstance(moved, FrozenPayload)
    assert moved is not payload
    assert moved.tensor.device.type == "cpu"
    assert moved.calculated == 7


def test_move_to_device_preserves_shared_container_references() -> None:
    shared = [torch.ones(1)]
    value = [shared, shared]

    moved = move_to_device(value, "cpu")

    assert moved[0] is moved[1]


def test_move_to_device_preserves_representable_cycles() -> None:
    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    tuple_member: list[Any] = []
    cyclic_tuple = (tuple_member,)
    tuple_member.append(cyclic_tuple)
    node = Node(torch.ones(1))
    node.next = node

    moved_list = move_to_device(cyclic_list, "cpu")
    moved_tuple = move_to_device(cyclic_tuple, "cpu")
    moved_node = move_to_device(node, "cpu")

    assert moved_list[0] is moved_list
    assert moved_tuple[0][0] is moved_tuple
    assert moved_node.next is moved_node


def test_move_to_device_reports_unknown_object_path() -> None:
    class Unsupported:
        pass

    with pytest.raises(
        TypeError,
        match=r"Unsupported value at \$\['payload'\]\[1\]: .*Unsupported",
    ):
        move_to_device({"payload": [torch.ones(1), Unsupported()]}, "cpu")


def test_move_to_device_forwards_non_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.device, bool]] = []

    def fake_to(
        tensor: torch.Tensor,
        *,
        device: torch.device,
        non_blocking: bool,
    ) -> torch.Tensor:
        calls.append((device, non_blocking))
        return tensor

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    tensor = torch.ones(1)

    assert move_to_device(tensor, "cpu", non_blocking=True) is tensor
    assert calls == [(torch.device("cpu"), True)]
