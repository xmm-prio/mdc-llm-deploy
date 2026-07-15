"""Deterministic release fixture helpers."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

FIXTURE_SEED = 20260714
FIXTURE_SEQUENCE_LENGTH = 3072
FIXTURE_VOCAB_SIZE = 128


def release_input_ids() -> np.ndarray:
    """Create canonical little-endian int64 release input IDs."""
    generator = np.random.Generator(np.random.PCG64(FIXTURE_SEED))
    values = generator.integers(
        0,
        FIXTURE_VOCAB_SIZE,
        size=(1, FIXTURE_SEQUENCE_LENGTH),
        dtype=np.int64,
    )
    return values.astype("<i8", copy=False)


def fixture_bytes() -> bytes:
    """Return canonical fixture bytes."""
    return release_input_ids().tobytes(order="C")


def fixture_sha256() -> str:
    """Return canonical fixture SHA-256."""
    return hashlib.sha256(fixture_bytes()).hexdigest()


def write_fixture(path: str | Path, *, overwrite: bool = False) -> str:
    """Atomically write canonical fixture and return its SHA-256."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(fixture_bytes())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return fixture_sha256()
