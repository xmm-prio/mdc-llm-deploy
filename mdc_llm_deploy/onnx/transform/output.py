"""Finalize the public runtime output contract."""

from __future__ import annotations

import onnx

from ...errors import GraphStateError, OnnxExportError
from ...graph.metadata import GraphMetadata, derive_artifact_io_abi


def finalize_artifact_outputs(
    model: onnx.ModelProto,
    metadata: GraphMetadata,
) -> None:
    """Expose graph outputs in centralized artifact ABI order."""
    try:
        expected = derive_artifact_io_abi(metadata).outputs
    except GraphStateError as error:
        raise OnnxExportError(
            f"Invalid graph artifact ABI: {error}"
        ) from error
    outputs_by_name: dict[str, list[onnx.ValueInfoProto]] = {}
    for output in model.graph.output:
        outputs_by_name.setdefault(output.name, []).append(output)
    missing_or_duplicate = {
        entry.name: len(outputs_by_name.get(entry.name, ()))
        for entry in expected
        if len(outputs_by_name.get(entry.name, ())) != 1
    }
    if missing_or_duplicate:
        raise OnnxExportError(
            "ONNX output finalization requires one source for every artifact "
            f"output: {missing_or_duplicate}"
        )
    del model.graph.output[:]
    model.graph.output.extend(
        outputs_by_name[entry.name][0] for entry in expected
    )


__all__ = ["finalize_artifact_outputs"]
