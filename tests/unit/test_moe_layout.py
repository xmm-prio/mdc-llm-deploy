from __future__ import annotations

from collections.abc import Callable

import pytest

from mdc_llm_deploy.moe_layout import MoeExpertLayout, infer_moe_layout


def test_expert_major_layout_has_per_row_projection_offsets() -> None:
    layout = MoeExpertLayout(3)
    segments = layout.weight_segments(8, 4)

    assert layout.expert_order() == ("0", "1", "2")
    assert layout.quant_parameter_count == 9
    assert layout.quant_parameter_order() == (
        "expert.0.gate",
        "expert.0.up",
        "expert.0.down",
        "expert.1.gate",
        "expert.1.up",
        "expert.1.down",
        "expert.2.gate",
        "expert.2.up",
        "expert.2.down",
    )
    assert [(item.expert_id, item.offset) for item in segments] == [
        (0, 0),
        (0, 32),
        (0, 64),
        (1, 0),
        (1, 32),
        (1, 64),
        (2, 0),
        (2, 32),
        (2, 64),
    ]
    assert layout.packed_weight_length(8, 4) == 96
    assert layout.routing_shape(7, 2) == (7, 2)


@pytest.mark.parametrize("expert_count", [1, 2, 17])
def test_layout_infers_arbitrary_expert_count(expert_count: int) -> None:
    assert infer_moe_layout((expert_count, 96)).expert_count == expert_count


@pytest.mark.parametrize(
    "operation",
    [
        lambda: MoeExpertLayout(0),
        lambda: MoeExpertLayout(2).routing_shape(1, 3),
        lambda: infer_moe_layout((3,)),
        lambda: infer_moe_layout((3, 0)),
    ],
)
def test_layout_rejects_invalid_values(operation: Callable[[], object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        operation()
