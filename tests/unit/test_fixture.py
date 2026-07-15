from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest

from mdc_llm_deploy.utils import (
    FIXTURE_SEED,
    FIXTURE_SEQUENCE_LENGTH,
    FIXTURE_VOCAB_SIZE,
    fixture_bytes,
    fixture_sha256,
    release_input_ids,
    write_fixture,
)

fixture_module = importlib.import_module("mdc_llm_deploy.utils.fixture")
EXPECTED_FIXTURE_SHA256 = "58483859943471c7c5b7c6d0d1282a83219e843af289a537ba95bb30a33f5c48"


def test_release_fixture_is_little_endian_and_deterministic() -> None:
    first = release_input_ids()
    second = release_input_ids()
    decoded = np.frombuffer(fixture_bytes(), dtype="<i8").reshape(1, -1)

    assert FIXTURE_SEED == 20260714
    assert FIXTURE_SEQUENCE_LENGTH == 3072
    assert FIXTURE_VOCAB_SIZE == 128
    assert first.shape == (1, 3072)
    assert first.dtype.kind == "i"
    assert first.dtype.itemsize == 8
    assert first.flags.c_contiguous
    assert np.array_equal(first, second)
    assert np.array_equal(first, decoded)
    assert int(first.min()) >= 0
    assert int(first.max()) < 128
    assert first[0, :8].tolist() == [16, 122, 57, 123, 90, 27, 51, 17]
    assert len(fixture_bytes()) == 3072 * 8
    assert fixture_sha256() == hashlib.sha256(fixture_bytes()).hexdigest()
    assert fixture_sha256() == EXPECTED_FIXTURE_SHA256


def test_write_fixture_success_and_existing_file_invariance(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "input_ids.bin"
    result = write_fixture(target)

    assert result == fixture_sha256()
    assert target.read_bytes() == fixture_bytes()
    with pytest.raises(FileExistsError):
        write_fixture(target)
    assert target.read_bytes() == fixture_bytes()


def test_write_fixture_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "input_ids.bin"
    original = b"original"
    target.write_bytes(original)

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(fixture_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        write_fixture(target, overwrite=True)

    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]
