from __future__ import annotations

import pytest

from mdc_llm_deploy.models import ExportModelConfig


def test_export_config_enables_kv_cache_by_default() -> None:
    assert ExportModelConfig(sequence_length=4).save_kv_cache is True
    assert (
        ExportModelConfig(sequence_length=4, save_kv_cache=False).save_kv_cache
        is False
    )


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_export_config_requires_exact_bool_for_save_kv_cache(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="save_kv_cache must be a bool"):
        ExportModelConfig(sequence_length=4, save_kv_cache=value)  # type: ignore[arg-type]
