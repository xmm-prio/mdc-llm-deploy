"""Checkpoint location resolution."""

from __future__ import annotations

from pathlib import Path


def resolve_checkpoint(
    source: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local checkpoint directory or Hugging Face repository."""
    candidate = Path(source)
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.exists():
        raise ValueError(
            f"Local checkpoint source must be a directory: {candidate}"
        )
    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=str(source),
        revision=revision,
        allow_patterns=["*.json", "*.safetensors"],
        local_files_only=local_files_only,
    )
    return Path(downloaded)


__all__ = ["resolve_checkpoint"]
