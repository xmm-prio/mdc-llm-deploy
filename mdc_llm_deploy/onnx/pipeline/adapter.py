"""Atomic MDC ONNX graph adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import onnx
from onnx import GraphProto, NodeProto

from ...core.observability import get_logger, log_stage, progress_task
from ..fusion import (
    APPLY_ROTARY_POS_EMB_FUSION_PASS,
    FUSED_INFER_ATTENTION_SCORE_FUSION_PASS,
    RMS_NORM_FUSION_PASS,
    FusionPass,
    run_fusion_passes,
)
from ..graph import clone_model
from ..schema import ALL_SCHEMA_NAMES, register_schemas
from .compatibility import lower_opset_compatibility_core
from .normalization import normalize_graph_core
from .opset import downgrade_opset_core
from .qdq import lower_qdq_core

_CUSTOM_SCHEMA_NAMES = frozenset(ALL_SCHEMA_NAMES)
_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class AdapterConfig:
    """Configure the MDC ONNX graph adapter."""

    fuse_rms_norm: bool = True
    fuse_apply_rotary_pos_emb: bool = True
    fuse_fused_infer_attention_score: bool = True
    show_progress: bool = True


_FUSION_SELECTION: tuple[tuple[str, FusionPass], ...] = (
    ("fuse_rms_norm", RMS_NORM_FUSION_PASS),
    ("fuse_apply_rotary_pos_emb", APPLY_ROTARY_POS_EMB_FUSION_PASS),
    ("fuse_fused_infer_attention_score", FUSED_INFER_ATTENTION_SCORE_FUSION_PASS),
)


def _nodes(graph: GraphProto) -> Iterator[NodeProto]:
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _nodes(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _nodes(subgraph)


def _register_required_schemas(model: onnx.ModelProto) -> None:
    required = tuple(
        dict.fromkeys(
            node.op_type
            for node in _nodes(model.graph)
            if node.domain in ("", "ai.onnx") and node.op_type in _CUSTOM_SCHEMA_NAMES
        )
    )
    if required:
        _logger.info("Registering required ONNX schemas: count=%d", len(required))
        _logger.debug("Required ONNX schemas: %s", required)
        register_schemas(*required)


def _validate_final_graph(model: onnx.ModelProto) -> None:
    residual = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.domain in ("", "ai.onnx")
            and node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        }
    )
    if residual:
        _logger.warning(
            "ONNX final validation found residual QDQ operators: count=%d operators=%s",
            len(residual),
            residual,
        )
        raise ValueError(f"main graph still contains residual QDQ operators: {residual}")
    onnx.checker.check_model(model)


def _default_opset(model: onnx.ModelProto) -> int | None:
    return next(
        (opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")),
        None,
    )


def _run_stage(
    model: onnx.ModelProto,
    name: str,
    operation: Callable[[onnx.ModelProto], object],
) -> None:
    before = sum(1 for _ in _nodes(model.graph))
    with log_stage(_logger, f"ONNX {name}", details=f"nodes={before}"):
        operation(model)
    after = sum(1 for _ in _nodes(model.graph))
    _logger.info(
        "ONNX %s node change: before=%d after=%d delta=%+d",
        name,
        before,
        after,
        after - before,
    )


class OnnxAdapter:
    """Apply the atomic MDC pipeline according to immutable configuration."""

    def __init__(self, config: AdapterConfig) -> None:
        self._config = config

    @property
    def config(self) -> AdapterConfig:
        """Return adapter configuration."""
        return self._config

    def __call__(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Adapt one model in place after all stages succeed."""
        if not isinstance(model, onnx.ModelProto):
            raise TypeError("model must be an onnx.ModelProto")
        working = clone_model(model)
        source_nodes = sum(1 for _ in _nodes(working.graph))
        source_opset = _default_opset(working)
        fusion_passes = self._selected_fusion_passes()
        stages: Sequence[tuple[str, Callable[[onnx.ModelProto], object]]] = (
            ("QDQ lowering", lower_qdq_core),
            ("schema registration before lowering", _register_required_schemas),
            ("compatibility lowering", lower_opset_compatibility_core),
            ("opset downgrade", downgrade_opset_core),
            ("graph normalization", normalize_graph_core),
            ("fusion", lambda graph: run_fusion_passes(graph, passes=fusion_passes)),
            ("schema registration after fusion", _register_required_schemas),
            ("final validation", _validate_final_graph),
        )
        _logger.info(
            "ONNX adapter started: nodes=%d source_opset=%s fusion_passes=%d show_progress=%s",
            source_nodes,
            source_opset,
            len(fusion_passes),
            self._config.show_progress,
        )
        _logger.debug(
            "ONNX adapter configuration: fuse_rms_norm=%s "
            "fuse_apply_rotary_pos_emb=%s fuse_fused_infer_attention_score=%s",
            self._config.fuse_rms_norm,
            self._config.fuse_apply_rotary_pos_emb,
            self._config.fuse_fused_infer_attention_score,
        )
        with progress_task(
            "Processing ONNX pipeline",
            total=len(stages),
            show_progress=self._config.show_progress,
        ) as advance:
            for name, operation in stages:
                _run_stage(working, name, operation)
                advance()
        model.CopyFrom(working)
        final_nodes = sum(1 for _ in _nodes(model.graph))
        _logger.info(
            "ONNX adapter completed: nodes=%d node_delta=%+d target_opset=%s",
            final_nodes,
            final_nodes - source_nodes,
            _default_opset(model),
        )
        return model

    def _selected_fusion_passes(self) -> tuple[FusionPass, ...]:
        return tuple(
            fusion_pass
            for config_field, fusion_pass in _FUSION_SELECTION
            if getattr(self._config, config_field)
        )


__all__ = ["AdapterConfig", "OnnxAdapter"]
