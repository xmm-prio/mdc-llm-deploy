"""Static ATen FX export."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor, nn
from torch.fx import GraphModule

from ..errors import UnsupportedPatternError
from ..fx_inspection import node_target
from ..graph import (
    infer_model_kind,
    set_metadata,
)
from ..graph_types import GRAPH_SCHEMA_VERSION, GraphMetadata, GraphStage
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
    devices = {value.device for value in example_inputs.values()}
    devices.update(parameter.device for parameter in model.parameters())
    devices.update(buffer.device for buffer in model.buffers())
    if len(devices) > 1:
        raise ValueError(
            "model parameters and example inputs must use one device"
        )
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
    _validate_aten_graph(graph)
    model_kind = _model_kind(model)
    object.__setattr__(
        graph,
        "_mdc_model_kind",
        model_kind,
    )
    discovered = discover_metadata(model, graph, example_inputs)
    value = GraphMetadata(
        schema_version=GRAPH_SCHEMA_VERSION,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind=model_kind,
        input_abi=discovered.input_abi,
        output_abi=discovered.output_abi,
        boundaries=discovered.boundaries,
        sequence_length=int(input_ids.shape[1]),
        properties=discovered.properties,
    )
    set_metadata(graph, value)
    return graph
