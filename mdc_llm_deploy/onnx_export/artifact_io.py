"""Validated atomic persistence for ONNX and external tensor data."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import onnx

from ..errors import OnnxExportError
from .validator import validate_serialized_model


def commit_validated_onnx(
    model: onnx.ModelProto,
    target: Path,
    *,
    external_data: bool,
) -> onnx.ModelProto:
    """Validate temporary artifacts and atomically replace final paths."""
    data_target = target.with_name(f"{target.name}.data")
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{target.stem}.",
            dir=target.parent,
            ignore_cleanup_errors=True,
        ) as directory:
            temporary_model = Path(directory) / target.name
            temporary_data = Path(directory) / data_target.name
            if external_data:
                onnx.save_model(
                    model,
                    temporary_model,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=data_target.name,
                    size_threshold=0,
                    convert_attribute=False,
                )
            else:
                onnx.save_model(model, temporary_model)
            validate_serialized_model(str(temporary_model))
            if external_data and temporary_data.is_file():
                os.replace(temporary_data, data_target)
            elif data_target.exists():
                data_target.unlink()
            os.replace(temporary_model, target)
        return onnx.load(target, load_external_data=True)
    except OnnxExportError:
        raise
    except Exception as error:
        raise OnnxExportError(f"ONNX export failed: {error}") from error


__all__ = ["commit_validated_onnx"]
