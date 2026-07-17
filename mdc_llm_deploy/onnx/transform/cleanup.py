"""Structural cleanup utilities for lowered ONNX graphs."""

from __future__ import annotations

import heapq

import onnx

from ...errors import OnnxExportError
from ..inspection import (
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


class _IdentityCleanupContext:
    """Maintain live graph indexes while removing redundant identities."""

    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.nodes = list(model.graph.node)
        self.alive = [True] * len(self.nodes)
        self.index_by_node_id = {
            id(node): index for index, node in enumerate(self.nodes)
        }
        self.consumer_count_by_value: dict[str, int] = {}
        self.consumer_indices_by_value: dict[str, list[int]] = {}
        self.producer_indices_by_value: dict[str, list[int]] = {}
        for index, node in enumerate(self.nodes):
            for input_name in node.input:
                if not input_name:
                    continue
                self.consumer_count_by_value[input_name] = (
                    self.consumer_count_by_value.get(input_name, 0) + 1
                )
                self.consumer_indices_by_value.setdefault(input_name, []).append(index)
            for output_name in node.output:
                if output_name:
                    heapq.heappush(
                        self.producer_indices_by_value.setdefault(output_name, []),
                        -index,
                    )

        self.graph_outputs = {item.name for item in model.graph.output}
        self.typed_name_counts: dict[str, int] = {}
        for items in (model.graph.input, model.graph.output, model.graph.value_info):
            for item in items:
                self.typed_name_counts[item.name] = (
                    self.typed_name_counts.get(item.name, 0) + 1
                )
        self.value_info_indices_by_name: dict[str, list[int]] = {}
        self.value_info_names: list[str] = []
        for index, item in enumerate(model.graph.value_info):
            self.value_info_names.append(item.name)
            heapq.heappush(
                self.value_info_indices_by_name.setdefault(item.name, []),
                index,
            )

        self.candidate_heap: list[int] = []
        self.queued = [False] * len(self.nodes)
        for index in range(len(self.nodes)):
            self._enqueue(index)

    @staticmethod
    def _is_structurally_legal(identity: onnx.NodeProto) -> bool:
        return (
            identity.op_type == "Identity"
            and identity.domain in STANDARD_DOMAINS
            and not identity.attribute
            and len(identity.input) == 1
            and len(identity.output) == 1
        )

    def _enqueue(self, index: int) -> None:
        if (
            self.alive[index]
            and not self.queued[index]
            and self._is_structurally_legal(self.nodes[index])
        ):
            heapq.heappush(self.candidate_heap, index)
            self.queued[index] = True

    def _producer_index(self, value_name: str) -> int | None:
        indices = self.producer_indices_by_value.get(value_name)
        if indices is None:
            return None
        while indices:
            index = -indices[0]
            if self.alive[index] and value_name in self.nodes[index].output:
                return index
            heapq.heappop(indices)
        return None

    def _value_info_index(self, value_name: str) -> int | None:
        indices = self.value_info_indices_by_name.get(value_name)
        if indices is None:
            return None
        while indices and self.value_info_names[indices[0]] != value_name:
            heapq.heappop(indices)
        return indices[0] if indices else None

    def _removable(self, index: int) -> tuple[int, str, str] | None:
        identity = self.nodes[index]
        source = identity.input[0]
        output = identity.output[0]
        producer_index = self._producer_index(source)
        if (
            not source
            or not output
            or source in self.graph_outputs
            or self.consumer_count_by_value.get(source) != 1
            or producer_index is None
            or self.nodes[producer_index].op_type in CUSTOM_OPS
        ):
            return None
        return producer_index, source, output

    def _enqueue_consumers(self, value_name: str) -> None:
        for index in self.consumer_indices_by_value.get(value_name, ()):
            self._enqueue(index)

    def _remove_one(
        self,
        identity_index: int,
        producer_index: int,
        source: str,
        output: str,
    ) -> None:
        identity = self.nodes[identity_index]
        producer = self.nodes[producer_index]
        replaced_output_count = sum(name == source for name in producer.output)

        producer.output[:] = [
            output if name == source else name for name in producer.output
        ]
        value_info_index = None
        if self.typed_name_counts.get(output, 0) == 0:
            value_info_index = self._value_info_index(source)
            if value_info_index is not None:
                self.model.graph.value_info[value_info_index].name = output
        self.model.graph.node.remove(identity)

        self.alive[identity_index] = False
        for _ in range(replaced_output_count):
            heapq.heappush(
                self.producer_indices_by_value.setdefault(output, []),
                -producer_index,
            )
        if value_info_index is not None:
            self.typed_name_counts[source] -= 1
            self.typed_name_counts[output] = (
                self.typed_name_counts.get(output, 0) + 1
            )
            self.value_info_names[value_info_index] = output
            heapq.heappush(
                self.value_info_indices_by_name.setdefault(output, []),
                value_info_index,
            )
        self.consumer_count_by_value[source] -= 1
        self._enqueue_consumers(source)
        if output != source:
            self._enqueue_consumers(output)

    def run(self) -> None:
        """Remove candidates in current graph order until convergence."""
        while self.candidate_heap:
            index = heapq.heappop(self.candidate_heap)
            self.queued[index] = False
            if not self.alive[index]:
                continue
            removable = self._removable(index)
            if removable is None:
                continue
            self._remove_one(index, *removable)


def remove_redundant_identities(model: onnx.ModelProto) -> None:
    """Remove strict Identity nodes without crossing an MDC custom operator."""
    _IdentityCleanupContext(model).run()


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
    nodes = list(model.graph.node)
    external_values = {item.name for item in model.graph.input}
    external_values.update(item.name for item in model.graph.initializer)
    producer_index_by_value: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output in node.output:
            if not output:
                continue
            previous_index = producer_index_by_value.get(output)
            if output in external_values or previous_index is not None:
                previous = (
                    "graph input or initializer"
                    if output in external_values
                    else f"node {previous_index}"
                )
                raise OnnxExportError(
                    "Lowered ONNX graph cannot be topologically sorted: "
                    f"value {output!r} has duplicate producers "
                    f"({previous} and node {index})"
                )
            producer_index_by_value[output] = index

    consumers_by_producer: list[list[int]] = [[] for _ in nodes]
    remaining_dependencies: list[int] = []
    for consumer_index, node in enumerate(nodes):
        producer_indices: set[int] = set()
        missing_values: set[str] = set()
        for input_name in node.input:
            if not input_name or input_name in external_values:
                continue
            producer_index = producer_index_by_value.get(input_name)
            if producer_index is None:
                missing_values.add(input_name)
            else:
                producer_indices.add(producer_index)
        for producer_index in producer_indices:
            consumers_by_producer[producer_index].append(consumer_index)
        remaining_dependencies.append(len(producer_indices) + len(missing_values))

    ready = [
        (index, index)
        for index, dependency_count in enumerate(remaining_dependencies)
        if dependency_count == 0
    ]
    heapq.heapify(ready)
    ordered: list[onnx.NodeProto] = []
    scheduled = [False] * len(nodes)
    known = set(external_values)
    while ready:
        _, index = heapq.heappop(ready)
        node = nodes[index]
        ordered.append(node)
        scheduled[index] = True
        known.update(output for output in node.output if output)
        for consumer_index in consumers_by_producer[index]:
            remaining_dependencies[consumer_index] -= 1
            if remaining_dependencies[consumer_index] == 0:
                heapq.heappush(ready, (consumer_index, consumer_index))

    if len(ordered) != len(nodes):
        blocked = {
            node.name or node.op_type: [
                name for name in node.input if name and name not in known
            ]
            for index, node in enumerate(nodes)
            if not scheduled[index]
        }
        raise OnnxExportError(
            f"Lowered ONNX graph cannot be topologically sorted: {blocked}"
        )
    del model.graph.node[:]
    model.graph.node.extend(ordered)


def remove_dynamic_value_info(model: onnx.ModelProto) -> None:
    """Retain only fully static intermediate value metadata."""
    static_values = [
        item for item in model.graph.value_info if static_shape(item) is not None
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(static_values)
