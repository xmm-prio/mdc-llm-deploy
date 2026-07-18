"""Validated atomic persistence for ONNX and external tensor data."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import onnx

from ...errors import OnnxExportError
from ..validation.model import load_validated_mdc_artifact
from .normalization import validate_serialized_normalized_onnx


@dataclass(frozen=True)
class _PublicationMember:
    target: Path
    staged: Path | None


def _snapshot_file(source: Path, backup: Path) -> None:
    follow_symlinks = os.name != "nt"
    try:
        os.link(source, backup, follow_symlinks=follow_symlinks)
    except OSError as link_error:
        try:
            shutil.copy2(source, backup, follow_symlinks=follow_symlinks)
        except OSError as copy_error:
            raise OnnxExportError(
                "ONNX snapshot failed: "
                f"link failed: {link_error}; copy failed: {copy_error}"
            ) from copy_error


def _snapshot_publication(
    members: tuple[_PublicationMember, ...],
    directory: Path,
) -> dict[Path, Path | None]:
    snapshots: dict[Path, Path | None] = {}
    for index, member in enumerate(members):
        if member.target.exists():
            backup = directory / f".backup-{index}"
            _snapshot_file(member.target, backup)
            snapshots[member.target] = backup
        else:
            snapshots[member.target] = None
    return snapshots


def _restore_publication_member(
    member: _PublicationMember,
    backup: Path | None,
) -> None:
    if backup is None:
        member.target.unlink(missing_ok=True)
    else:
        os.replace(backup, member.target)


def _commit_publication(
    members: tuple[_PublicationMember, ...],
    directory: Path,
) -> None:
    snapshots = _snapshot_publication(members, directory)
    attempted: list[_PublicationMember] = []
    try:
        for member in members:
            attempted.append(member)
            if member.staged is None:
                member.target.unlink(missing_ok=True)
            else:
                os.replace(member.staged, member.target)
    except Exception as publication_error:
        rollback_errors: list[Exception] = []
        for member in reversed(attempted):
            try:
                _restore_publication_member(member, snapshots[member.target])
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            summary = "; ".join(str(error) for error in rollback_errors)
            raise OnnxExportError(
                "ONNX publication failed: "
                f"{publication_error}; rollback failed: {summary}"
            ) from publication_error
        raise


def _commit_validated_onnx(
    model: onnx.ModelProto,
    target: Path,
    *,
    external_data: bool,
    validate_serialized: Callable[[str], object],
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
            validate_serialized(str(temporary_model))
            if external_data:
                members = (
                    _PublicationMember(
                        data_target,
                        temporary_data if temporary_data.is_file() else None,
                    ),
                    _PublicationMember(target, temporary_model),
                )
            else:
                members = (
                    _PublicationMember(target, temporary_model),
                    _PublicationMember(data_target, None),
                )
            _commit_publication(members, Path(directory))
        return onnx.load(target, load_external_data=True)
    except OnnxExportError:
        raise
    except Exception as error:
        raise OnnxExportError(f"ONNX export failed: {error}") from error


def commit_standard_onnx(
    model: onnx.ModelProto,
    target: Path,
    *,
    external_data: bool,
) -> onnx.ModelProto:
    """Atomically publish a validated standard ONNX artifact."""
    return _commit_validated_onnx(
        model,
        target,
        external_data=external_data,
        validate_serialized=validate_serialized_normalized_onnx,
    )


def commit_mdc_onnx(
    model: onnx.ModelProto,
    target: Path,
    *,
    external_data: bool,
) -> onnx.ModelProto:
    """Atomically publish a validated MDC ONNX artifact."""
    return _commit_validated_onnx(
        model,
        target,
        external_data=external_data,
        validate_serialized=load_validated_mdc_artifact,
    )


__all__ = ["commit_mdc_onnx", "commit_standard_onnx"]
