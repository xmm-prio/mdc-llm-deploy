"""Graph lifecycle operations and compatibility contract exports."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from .errors import GraphStateError
from .graph_contract import (
    require_boundaries as require_boundaries,
)
from .graph_contract import (
    validate_capability_request as validate_capability_request,
)
from .graph_contract import (
    validate_metadata as validate_metadata,
)
from .graph_types import (
    GRAPH_METADATA_KEY as GRAPH_METADATA_KEY,
)
from .graph_types import (
    GRAPH_SCHEMA_VERSION as GRAPH_SCHEMA_VERSION,
)
from .graph_types import (
    FusionBoundary as FusionBoundary,
)
from .graph_types import (
    GraphMetadata as GraphMetadata,
)
from .graph_types import (
    GraphStage as GraphStage,
)
from .graph_types import (
    QuantizedTarget as QuantizedTarget,
)
from .graph_types import (
    TensorAbi as TensorAbi,
)

if TYPE_CHECKING:
    from torch import nn
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


def transactional_update(graph: T, mutator: Callable[[T], None]) -> T:
    """Validate a candidate then atomically replace state, preserving identity."""
    validate_graph(graph)
    candidate = copy.deepcopy(graph)
    mutator(candidate)
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
