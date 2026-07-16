"""Validated and atomic persistence for MDC ONNX artifacts."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path

import onnx

from ..errors import OnnxExportError
from .validator import validate_serialized_model


def commit_validated_onnx(
    model: onnx.ModelProto,
    target: Path,
    *,
    overwrite: bool,
) -> onnx.ModelProto:
    """Serialize, validate, and atomically replace an ONNX artifact."""
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.stem}.",
            suffix=".onnx.tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        os.close(descriptor)
        descriptor = -1
        onnx.save_model(model, temporary)
        validated = validate_serialized_model(str(temporary))
        if overwrite:
            os.replace(temporary, target)
        elif os.name == "nt":
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
        return validated
    except (FileExistsError, OnnxExportError):
        raise
    except Exception as error:
        raise OnnxExportError(f"ONNX export failed: {error}") from error
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
