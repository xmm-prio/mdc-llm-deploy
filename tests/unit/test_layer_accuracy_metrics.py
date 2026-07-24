from __future__ import annotations

import numpy as np
import pytest

from examples.qwen3_8b_layer_accuracy.metrics import compare_arrays, load_array


def test_compare_arrays_reports_exact_match() -> None:
    values = np.arange(12, dtype=np.float16).reshape(1, 3, 4)

    metrics = compare_arrays(values, values.copy())

    assert metrics.finite
    assert metrics.cosine == pytest.approx(1.0)
    assert metrics.max_absolute_error == 0.0
    assert metrics.mean_absolute_error == 0.0
    assert metrics.mean_relative_error == 0.0


def test_compare_arrays_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_arrays(np.zeros((2, 2)), np.zeros((4,)))


def test_load_array_requires_binary_shape(tmp_path) -> None:
    path = tmp_path / "output.bin"
    np.arange(4, dtype=np.float16).tofile(path)

    with pytest.raises(ValueError, match="shape is required"):
        load_array(path)

    loaded = load_array(path, shape=(2, 2))
    np.testing.assert_array_equal(loaded, np.arange(4, dtype=np.float16).reshape(2, 2))
