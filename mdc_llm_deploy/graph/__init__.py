"""Torch-independent graph contracts and metadata."""

from .contract import (
    require_boundaries,
    validate_capability_request,
    validate_metadata,
)
from .metadata import (
    GRAPH_METADATA_KEY,
    GRAPH_SCHEMA_VERSION,
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    QuantizedTarget,
    TensorAbi,
)

__all__ = [
    "GRAPH_METADATA_KEY",
    "GRAPH_SCHEMA_VERSION",
    "FusionBoundary",
    "GraphMetadata",
    "GraphStage",
    "QuantizedTarget",
    "TensorAbi",
    "require_boundaries",
    "validate_capability_request",
    "validate_metadata",
]
