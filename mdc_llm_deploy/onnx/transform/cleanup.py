"""Structural cleanup utilities for lowered ONNX graphs."""

from __future__ import annotations

import onnx

from ...errors import OnnxExportError
from ..model_inspection import (
    optional_static_shape as static_shape,
)
from ..validation.topology import CUSTOM_OPS, STANDARD_DOMAINS


def producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    """Map each produced value to its node."""
    return {output: node for node in model.graph.node for output in node.output}


def replace_nodes(
    model: onnx.ModelProto,
    removed: set[int],
    replacement_by_index: dict[int, list[onnx.NodeProto]],
) -> None:
    """Replace selected nodes while retaining graph order."""
    nodes = list(model.graph.node)
    result: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        result.extend(replacement_by_index.get(index, ()))
        if index not in removed:
            result.append(node)
    del model.graph.node[:]
    model.graph.node.extend(result)


def remove_redundant_identities(model: onnx.ModelProto) -> None:
    """Remove strict Identity nodes without crossing an MDC custom operator."""
    while True:
        nodes = list(model.graph.node)
        graph_outputs = {item.name for item in model.graph.output}
        producers = producer_map(model)
        consumer_counts: dict[str, int] = {}
        for node in nodes:
            for name in node.input:
                if name:
                    consumer_counts[name] = consumer_counts.get(name, 0) + 1
        removable: tuple[onnx.NodeProto, onnx.NodeProto, str, str] | None = None
        for identity in nodes:
            if (
                identity.op_type != "Identity"
                or identity.domain not in STANDARD_DOMAINS
                or identity.attribute
                or len(identity.input) != 1
                or len(identity.output) != 1
            ):
                continue
            source = identity.input[0]
            output = identity.output[0]
            producer = producers.get(source)
            if (
                not source
                or not output
                or source in graph_outputs
                or consumer_counts.get(source) != 1
                or producer is None
                or producer.op_type in CUSTOM_OPS
            ):
                continue
            removable = identity, producer, source, output
            break
        if removable is None:
            return
        identity, producer, source, output = removable
        producer.output[:] = [
            output if name == source else name for name in producer.output
        ]
        typed_values = {item.name for item in model.graph.input}
        typed_values.update(item.name for item in model.graph.output)
        typed_values.update(item.name for item in model.graph.value_info)
        if output not in typed_values:
            source_info = next(
                (item for item in model.graph.value_info if item.name == source),
                None,
            )
            if source_info is not None:
                source_info.name = output
        model.graph.node.remove(identity)


def prune_unreachable(model: onnx.ModelProto) -> None:
    """Remove standard lowering remnants that cannot affect graph outputs."""
    remove_redundant_identities(model)
    producers = producer_map(model)
    required_values = [item.name for item in model.graph.output]
    required_outputs: set[str] = set()
    while required_values:
        value_name = required_values.pop()
        producer = producers.get(value_name)
        if producer is None or any(
            output in required_outputs for output in producer.output
        ):
            continue
        required_outputs.update(producer.output)
        required_values.extend(name for name in producer.input if name)
    retained = [
        node
        for node in model.graph.node
        if any(output in required_outputs for output in node.output)
    ]
    del model.graph.node[:]
    model.graph.node.extend(retained)
    used_initializers = {
        name for node in model.graph.node for name in node.input if name
    }
    used_initializers.update(item.name for item in model.graph.output)
    retained_initializers = [
        item for item in model.graph.initializer if item.name in used_initializers
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)
    valid_values = {item.name for item in model.graph.input}
    valid_values.update(item.name for item in model.graph.output)
    valid_values.update(item.name for item in model.graph.initializer)
    valid_values.update(output for node in model.graph.node for output in node.output)
    retained_value_info = [
        item for item in model.graph.value_info if item.name in valid_values
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)


def topologically_sort(model: onnx.ModelProto) -> None:
    """Restore topological order after inserting fused MDC nodes."""
    known = {item.name for item in model.graph.input}
    known.update(item.name for item in model.graph.initializer)
    pending = list(model.graph.node)
    ordered: list[onnx.NodeProto] = []
    while pending:
        ready = next(
            (
                node
                for node in pending
                if all(not name or name in known for name in node.input)
            ),
            None,
        )
        if ready is None:
            blocked = {
                node.name or node.op_type: [
                    name for name in node.input if name and name not in known
                ]
                for node in pending
            }
            raise OnnxExportError(
                f"Lowered ONNX graph cannot be topologically sorted: {blocked}"
            )
        pending.remove(ready)
        ordered.append(ready)
        known.update(ready.output)
    del model.graph.node[:]
    model.graph.node.extend(ordered)


def remove_dynamic_value_info(model: onnx.ModelProto) -> None:
    """Retain only fully static intermediate value metadata."""
    static_values = [
        item for item in model.graph.value_info if static_shape(item) is not None
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(static_values)
