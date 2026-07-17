"""Graph lifecycle operations and compatibility contract exports."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode

from ..errors import GraphStateError
from .contract import (
    require_boundaries as require_boundaries,
)
from .contract import (
    validate_capability_request as validate_capability_request,
)
from .contract import (
    validate_metadata as validate_metadata,
)
from .metadata import (
    GRAPH_METADATA_KEY as GRAPH_METADATA_KEY,
)
from .metadata import (
    GRAPH_SCHEMA_VERSION as GRAPH_SCHEMA_VERSION,
)
from .metadata import (
    FusionBoundary as FusionBoundary,
)
from .metadata import (
    GraphMetadata as GraphMetadata,
)
from .metadata import (
    GraphStage as GraphStage,
)
from .metadata import (
    QuantizedTarget as QuantizedTarget,
)
from .metadata import (
    TensorAbi as TensorAbi,
)

if TYPE_CHECKING:
    from torch.fx import GraphModule


def metadata(graph: GraphModule) -> GraphMetadata:
    """Return and validate graph metadata."""
    value = graph.meta.get(GRAPH_METADATA_KEY)
    if not isinstance(value, GraphMetadata):
        raise GraphStateError("Graph does not carry MDC metadata")
    validate_metadata(value)
    return value


def set_metadata(graph: GraphModule, value: GraphMetadata) -> None:
    """Attach validated graph metadata."""
    validate_metadata(value)
    graph.meta[GRAPH_METADATA_KEY] = value


def validate_graph(graph: GraphModule) -> GraphMetadata:
    """Validate graph structure and all attached cross-module metadata."""
    if not hasattr(graph, "graph") or not hasattr(graph, "meta"):
        raise TypeError("graph must be a torch.fx.GraphModule")
    graph.graph.lint()  # type: ignore[no-untyped-call]
    value = metadata(graph)
    nodes = tuple(graph.graph.nodes)
    names = [node.name for node in nodes]
    if len(names) != len(set(names)):
        raise GraphStateError("FX node names must be unique")
    placeholders = tuple(node for node in nodes if node.op == "placeholder")
    if len(placeholders) != len(value.input_abi):
        raise GraphStateError("FX placeholders do not match input ABI cardinality")
    output_nodes = tuple(node for node in nodes if node.op == "output")
    if len(output_nodes) != 1:
        raise GraphStateError("FX graph must contain exactly one output node")
    raw_output = output_nodes[0].args[0]
    output_count = len(raw_output) if isinstance(raw_output, (tuple, list)) else 1
    if value.output_abi and output_count != len(value.output_abi):
        raise GraphStateError("FX outputs do not match output ABI cardinality")
    return value


T = TypeVar("T", bound="GraphModule")


def _collect_tensors(value: object, tensors: dict[int, Tensor]) -> None:
    if isinstance(value, Tensor):
        tensors[id(value)] = value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _collect_tensors(key, tensors)
            _collect_tensors(item, tensors)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _collect_tensors(item, tensors)


def _graph_tensors(graph: GraphModule) -> dict[int, Tensor]:
    tensors = {
        id(tensor): tensor
        for tensor in (
            *graph.parameters(),
            *graph.buffers(),
        )
    }
    for module in graph.modules():
        _collect_tensors(module.__dict__, tensors)
    _collect_tensors(graph.meta, tensors)
    for node in graph.graph.nodes:
        _collect_tensors(node.meta, tensors)
    return tensors


def _written_tensors(
    function: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[Tensor, ...]:
    schema = getattr(function, "_schema", None)
    arguments = getattr(schema, "arguments", ())
    written: list[Tensor] = []
    for index, argument in enumerate(arguments):
        alias_info = getattr(argument, "alias_info", None)
        if alias_info is None or not alias_info.is_write:
            continue
        value = args[index] if index < len(args) else kwargs.get(argument.name)
        values: dict[int, Tensor] = {}
        _collect_tensors(value, values)
        written.extend(values.values())
    return tuple(written)


class _TensorMutationJournal(TorchDispatchMode, AbstractContextManager[None]):
    def __init__(self, shared_tensors: dict[int, Tensor]) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self._shared_tensors = shared_tensors
        self._snapshots: dict[int, Tensor] = {}

    def __torch_dispatch__(
        self,
        function: object,
        types: tuple[type[Any], ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        call_kwargs = kwargs or {}
        for tensor in _written_tensors(function, args, call_kwargs):
            identity = id(tensor)
            if identity in self._shared_tensors and identity not in self._snapshots:
                self._snapshots[identity] = tensor.detach().clone(
                    memory_format=torch.preserve_format
                )
        return function(*args, **call_kwargs)  # type: ignore[operator]

    def restore(self) -> None:
        with torch.no_grad():
            for identity, snapshot in self._snapshots.items():
                original = self._shared_tensors[identity]
                if (
                    original.shape == snapshot.shape
                    and original.stride() == snapshot.stride()
                ):
                    original.copy_(snapshot)
                else:
                    original.data = snapshot

    def detach_candidate(self, candidate: GraphModule) -> None:
        replacements: dict[int, Tensor] = {}
        for identity, original in self._shared_tensors.items():
            if identity not in self._snapshots:
                continue
            value = original.detach().clone(memory_format=torch.preserve_format)
            replacements[identity] = (
                nn.Parameter(value, requires_grad=original.requires_grad)
                if isinstance(original, nn.Parameter)
                else value
            )
        self.restore()
        for module in candidate.modules():
            for name, parameter in module._parameters.items():
                if parameter is not None and id(parameter) in replacements:
                    module._parameters[name] = cast(
                        nn.Parameter,
                        replacements[id(parameter)],
                    )
            for name, buffer in module._buffers.items():
                if buffer is not None and id(buffer) in replacements:
                    module._buffers[name] = replacements[id(buffer)]


def transactional_update(graph: T, mutator: Callable[[T], None]) -> T:
    """Validate and atomically commit a copy-on-write graph candidate."""
    validate_graph(graph)
    tensors = _graph_tensors(graph)
    candidate = copy.deepcopy(graph, memo=tensors.copy())
    journal = _TensorMutationJournal(tensors)
    try:
        with journal:
            mutator(candidate)
        journal.detach_candidate(candidate)
    except BaseException as error:
        try:
            journal.restore()
        except BaseException as rollback_error:
            error.add_note(
                "Graph tensor rollback failed; original graph state may be incomplete "
                f"or unknown: {type(rollback_error).__name__}: {rollback_error}"
            )
        raise
    candidate.graph.lint()  # type: ignore[no-untyped-call]
    candidate.recompile()
    validate_graph(candidate)
    object.__setattr__(graph, "__dict__", candidate.__dict__)
    graph.recompile()
    return graph


def infer_model_kind(module: nn.Module) -> str:
    """Infer model kind from module metadata."""
    kind = getattr(module, "_mdc_model_kind", None)
    if kind in {"dense", "moe"}:
        return str(kind)
    module_names = tuple(name.lower() for name, _ in module.named_modules())
    return "moe" if any("expert" in name or "moe" in name for name in module_names) else "dense"
