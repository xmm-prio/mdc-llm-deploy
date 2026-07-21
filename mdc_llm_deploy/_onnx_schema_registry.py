"""Compatibility layer for the central ONNX schema registry."""

from __future__ import annotations

from collections.abc import Iterable

from onnx.defs import OpSchema

from .onnx.schemas import OnnxSchemaConflictError, register_schema_objects


def ensure_onnx_schemas(schemas: Iterable[OpSchema]) -> None:
    """Register declarative schemas through the central registry."""
    register_schema_objects(schemas)


__all__ = ["OnnxSchemaConflictError", "ensure_onnx_schemas"]
