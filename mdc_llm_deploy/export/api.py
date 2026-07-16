"""Static ATen FX export."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.fx import GraphModule, Node

from ..errors import UnsupportedPatternError
from ..fx_inspection import flatten_nodes, node_target
from ..graph import (
    infer_model_kind,
    set_metadata,
)
from ..graph_types import GRAPH_SCHEMA_VERSION, GraphMetadata, GraphStage
from ..operator_schema import TORCH_NAMESPACE
from ..placement import (
    PlacementSnapshot,
    capture_placement,
    validate_placement_preserved,
)
from .decode import convert_to_decode as convert_to_decode
from .discovery import discover_metadata


def _validate_aten_graph(graph: GraphModule) -> None:
    forbidden = [
        node.name
        for node in graph.graph.nodes
        if node.op in {"call_module", "call_method"}
        or (
            node.op == "call_function"
            and "aten::" not in node_target(node)
            and f"{TORCH_NAMESPACE}::" not in node_target(node)
            and node.target is not operator.getitem
        )
    ]
    if forbidden:
        raise UnsupportedPatternError(
            f"Export must produce a functional ATen graph; found {forbidden[:4]}"
        )


def _validate_input_ids(example_inputs: Mapping[str, Tensor]) -> Tensor:
    input_ids = example_inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2:
        raise UnsupportedPatternError("Static export requires rank-2 input_ids")
    return input_ids


def _model_kind(model: nn.Module) -> str:
    for attribute in ("model_kind", "_mdc_model_kind"):
        value = getattr(model, attribute, None)
        if value in {"dense", "moe"}:
            return str(value)
    return infer_model_kind(model)


def _state_fqn_to_graph_attribute(
    exported: torch.export.ExportedProgram,
    placement: PlacementSnapshot,
) -> dict[str, str]:
    signature = exported.graph_signature
    mapping = {
        fqn: fqn
        for fqn in (
            *signature.inputs_to_parameters.values(),
            *signature.inputs_to_buffers.values(),
        )
    }
    expected = set(placement.by_fqn)
    missing = sorted(expected - mapping.keys())
    if missing:
        raise UnsupportedPatternError(
            f"Export signature omitted model tensors: {missing}"
        )
    return {fqn: mapping[fqn] for fqn in sorted(expected)}


def _validate_exported_placement(
    graph: GraphModule,
    placement: PlacementSnapshot,
    fqn_to_attribute: Mapping[str, str],
) -> None:
    graph_placement = capture_placement(graph)
    graph_by_fqn = graph_placement.by_fqn
    changed: list[str] = []
    for fqn, expected in placement.by_fqn.items():
        attribute = fqn_to_attribute[fqn]
        actual = graph_by_fqn.get(attribute)
        if (
            actual is None
            or actual.kind != expected.kind
            or actual.device != expected.device
            or actual.dtype != expected.dtype
            or actual.persistent != expected.persistent
        ):
            changed.append(fqn)
    if changed:
        raise UnsupportedPatternError(
            f"Export changed tensor placement: {sorted(changed)}"
        )

    attributes = cast(
        dict[str, Tensor],
        dict(graph.named_parameters(remove_duplicate=False)),
    )
    attributes.update(graph.named_buffers(remove_duplicate=False))
    expected_aliases = {
        tuple(sorted(fqn_to_attribute[name] for name in group))
        for group in placement.alias_groups
    }
    actual_identities: dict[int, list[str]] = {}
    for attribute in fqn_to_attribute.values():
        actual_identities.setdefault(id(attributes[attribute]), []).append(attribute)
    actual_aliases = {
        tuple(sorted(names))
        for names in actual_identities.values()
        if len(names) > 1
    }
    if actual_aliases != expected_aliases:
        raise UnsupportedPatternError(
            "Export changed tensor alias groups: "
            f"{sorted(expected_aliases)!r} != {sorted(actual_aliases)!r}"
        )


def _tensor_devices(value: Any) -> set[torch.device]:
    if isinstance(value, Tensor):
        return {value.device}
    if isinstance(value, (tuple, list)):
        devices: set[torch.device] = set()
        for item in value:
            devices.update(_tensor_devices(item))
        return devices
    if isinstance(value, Mapping):
        devices = set()
        for item in value.values():
            devices.update(_tensor_devices(item))
        return devices
    return set()


def _node_devices(node: Node) -> set[torch.device]:
    return _tensor_devices(node.meta.get("val"))


def _is_explicit_transfer(node: Node) -> bool:
    return node.op == "call_function" and node_target(node) in {
        "aten::to",
        "aten::_to_copy",
        "aten::copy",
    }


def _validate_explicit_device_transfers(
    graph: GraphModule,
    placement: PlacementSnapshot,
) -> None:
    resident_devices = {item.device for item in placement.tensors}
    if len(resident_devices) <= 1:
        return
    transfers: set[str] = set()
    implicit_edges: list[str] = []
    for node in graph.graph.nodes:
        destination_devices = _node_devices(node)
        for source in flatten_nodes((node.args, node.kwargs)):
            source_devices = _node_devices(source)
            if (
                source_devices
                and destination_devices
                and source_devices.isdisjoint(destination_devices)
            ):
                if _is_explicit_transfer(node):
                    transfers.add(node.name)
                else:
                    implicit_edges.append(f"{source.name}->{node.name}")
    if implicit_edges:
        raise UnsupportedPatternError(
            "Export captured implicit cross-device edges: "
            f"{sorted(implicit_edges)[:8]}"
        )
    if not transfers:
        raise UnsupportedPatternError(
            "Multi-device export must capture an explicit device transfer"
        )


def export(
    model: nn.Module,
    example_inputs: Mapping[str, Tensor],
) -> GraphModule:
    """Export an eval-mode model to a static, functional ATen FX graph."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    if model.training:
        raise ValueError("model must be in eval mode")
    if not example_inputs:
        raise ValueError("example_inputs must not be empty")
    if not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in example_inputs.items()
    ):
        raise TypeError("example_inputs must map strings to tensors")
    placement = capture_placement(model)
    input_ids = _validate_input_ids(example_inputs)
    try:
        exported = torch.export.export(
            model,
            args=(),
            kwargs=dict(example_inputs),
            strict=False,
        )
        graph = cast(GraphModule, exported.module())
    except Exception as error:
        raise UnsupportedPatternError(f"ATen export failed: {error}") from error
    after_export = capture_placement(model)
    validate_placement_preserved(placement, after_export)
    fqn_to_attribute = _state_fqn_to_graph_attribute(exported, placement)
    _validate_exported_placement(graph, placement, fqn_to_attribute)
    _validate_explicit_device_transfers(graph, placement)
    _validate_aten_graph(graph)
    model_kind = _model_kind(model)
    object.__setattr__(
        graph,
        "_mdc_model_kind",
        model_kind,
    )
    discovered = discover_metadata(model, graph, example_inputs)
    properties = dict(discovered.properties)
    properties["state_fqn_to_graph_attribute"] = fqn_to_attribute
    value = GraphMetadata(
        schema_version=GRAPH_SCHEMA_VERSION,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind=model_kind,
        input_abi=discovered.input_abi,
        output_abi=discovered.output_abi,
        boundaries=discovered.boundaries,
        sequence_length=int(input_ids.shape[1]),
        properties=properties,
    )
    set_metadata(graph, value)
    return graph
