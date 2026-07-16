from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from mdc_llm_deploy.errors import UnsupportedPatternError
from mdc_llm_deploy.placement import (
    PlacementError,
    PlacementSnapshot,
    capture_placement,
    inherit_device,
    reject_dynamic_offload,
    validate_placement_preserved,
)


class TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared_parameter = nn.Parameter(torch.ones(2))
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 2, bias=False)
        self.first.weight = shared_parameter
        self.second.weight = shared_parameter
        shared_buffer = torch.zeros(2, dtype=torch.float16)
        self.register_buffer("persistent", shared_buffer)
        self.register_buffer("transient", shared_buffer, persistent=False)
        self.hf_device_map = {"first": "cpu", "second": torch.device("cpu")}


def test_capture_placement_records_fqns_aliases_and_persistence() -> None:
    snapshot = capture_placement(TiedModel())

    assert tuple(snapshot.by_fqn) == (
        "first.weight",
        "persistent",
        "second.weight",
        "transient",
    )
    assert snapshot.alias_groups == (
        ("first.weight", "second.weight"),
        ("persistent", "transient"),
    )
    assert snapshot.by_fqn["persistent"].persistent is True
    assert snapshot.by_fqn["transient"].persistent is False
    assert snapshot.by_fqn["persistent"].dtype is torch.float16
    assert snapshot.hf_device_map == (
        ("first", "cpu"),
        ("second", "cpu"),
    )


def test_validate_placement_accepts_identical_snapshots() -> None:
    snapshot = capture_placement(TiedModel())

    validate_placement_preserved(snapshot, snapshot)


def test_validate_placement_allows_new_tensor_with_preserved_old_aliases() -> None:
    before = capture_placement(TiedModel())
    new_tensor = replace(before.tensors[0], fqn="new_tensor")
    after = replace(
        before,
        tensors=(*before.tensors, new_tensor),
        alias_groups=(
            ("first.weight", "new_tensor", "second.weight"),
            ("persistent", "transient"),
        ),
    )

    validate_placement_preserved(before, after)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("tensors", (), "Tensor names changed"),
        ("alias_groups", (), "Tensor alias groups changed"),
        ("hf_device_map", (), "hf_device_map changed"),
    ],
)
def test_validate_placement_rejects_contract_changes(
    field_name: str,
    value: tuple[object, ...],
    match: str,
) -> None:
    before = capture_placement(TiedModel())
    after = replace(before, **{field_name: value})

    with pytest.raises(PlacementError, match=match):
        validate_placement_preserved(before, after)


def test_validate_placement_rejects_metadata_change() -> None:
    before = capture_placement(TiedModel())
    changed = replace(before.tensors[0], dtype=torch.float64)
    after = replace(before, tensors=(changed, *before.tensors[1:]))

    with pytest.raises(PlacementError, match="Tensor placement changed"):
        validate_placement_preserved(before, after)


def test_inherit_device_uses_frozen_priority_order() -> None:
    source = torch.ones(1)
    replaced = nn.Linear(1, 1, device="meta")
    nearest_parent = nn.Linear(1, 1, device="meta")
    distant_parent = nn.Linear(1, 1)

    assert inherit_device(
        sources=[source],
        replaced_module=replaced,
        parent_modules=[nearest_parent, distant_parent],
    ) == torch.device("cpu")
    assert inherit_device(
        replaced_module=replaced,
        parent_modules=[distant_parent],
    ) == torch.device("meta")
    assert inherit_device(
        parent_modules=[nearest_parent, distant_parent],
    ) == torch.device("meta")


def test_inherit_device_rejects_source_conflict() -> None:
    with pytest.raises(PlacementError, match="Conflicting source tensor devices"):
        inherit_device(
            sources=[torch.ones(1), torch.ones(1, device="meta")],
        )


def test_inherit_device_rejects_missing_placement_source() -> None:
    with pytest.raises(PlacementError, match="Cannot infer device"):
        inherit_device(replaced_module=nn.ReLU(), parent_modules=[nn.ReLU()])


def test_reject_dynamic_offload_rejects_disk_device_map() -> None:
    model = nn.Linear(1, 1)
    model.hf_device_map = {"": "disk"}

    with pytest.raises(UnsupportedPatternError, match="Disk offload"):
        reject_dynamic_offload(model)


@pytest.mark.parametrize("hook_name", ["AlignDevicesHook", "CpuOffload"])
def test_reject_dynamic_offload_rejects_weight_paging(hook_name: str) -> None:
    hook_type = type(hook_name, (), {})
    hook = hook_type()
    hook.offload = hook_name == "AlignDevicesHook"
    model = nn.Linear(1, 1)
    model._hf_hook = hook

    with pytest.raises(UnsupportedPatternError, match="Dynamic weight offload"):
        reject_dynamic_offload(model)


def test_reject_dynamic_offload_allows_static_accelerate_hook() -> None:
    hook_type = type("AlignDevicesHook", (), {})
    hook = hook_type()
    hook.offload = False
    model = nn.Linear(1, 1)
    model._hf_hook = hook

    reject_dynamic_offload(model)


def test_reject_dynamic_offload_inspects_sequential_hooks() -> None:
    dynamic_hook = type("AlignDevicesHook", (), {"offload": True})()
    sequential_hook = type("SequentialHook", (), {"hooks": (dynamic_hook,)})()
    model = nn.Linear(1, 1)
    model._hf_hook = sequential_hook

    with pytest.raises(UnsupportedPatternError, match="AlignDevicesHook"):
        reject_dynamic_offload(model)


def test_capture_placement_rejects_nonresident_tensor() -> None:
    with pytest.raises(UnsupportedPatternError, match="unsupported device meta"):
        capture_placement(nn.Linear(1, 1, device="meta"))


def test_snapshot_by_fqn_returns_new_mapping() -> None:
    snapshot = PlacementSnapshot((), (), ())

    assert snapshot.by_fqn == {}
    assert snapshot.by_fqn is not snapshot.by_fqn
