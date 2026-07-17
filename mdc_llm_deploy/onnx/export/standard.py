"""Build a validated standard ONNX intermediate model."""

from pathlib import Path

import onnx
from torch.fx import GraphModule

from ...errors import OnnxExportError
from ...graph.metadata import GraphMetadata
from .legacy import export_legacy_onnx
from .normalization import normalize_standard_onnx


def build_standard_onnx(
    graph: GraphModule,
    metadata: GraphMetadata,
    directory: Path,
) -> onnx.ModelProto:
    """Build and validate the standard ONNX intermediate model."""
    try:
        raw = export_legacy_onnx(graph, metadata, directory)
        return normalize_standard_onnx(raw, graph, metadata)
    except OnnxExportError:
        raise
    except Exception as error:
        raise OnnxExportError(
            f"Standard ONNX validation failed: {error}"
        ) from error
