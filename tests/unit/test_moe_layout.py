from __future__ import annotations

from collections.abc import Callable

import pytest

from mdc_llm_deploy.moe_layout import DEFAULT_MOE_LAYOUT, TinyMoeAbiLayout


def test_default_moe_layout_freezes_release_abi() -> None:
    layout = DEFAULT_MOE_LAYOUT

    assert layout.expert_count == 5
    assert layout.shared_expert_id == 4
    assert layout.route_width == 3
    assert layout.quant_parameter_count == 21
    assert layout.packed_projection_count == 15
    assert layout.input_activation_projections == (
        "gate_proj",
        "up_proj",
    )
    assert layout.output_projection == "down_proj"
    assert (
        layout.projection_for_fqn(
            "layers.0.experts.2.gate_proj"
        )
        == "gate_proj"
    )
    assert layout.projection_for_fqn("not_gate_proj") is None
    assert layout.expert_order() == ("0", "1", "2", "3", "4(shared)")
    assert layout.quant_parameter_order() == (
        "input",
        "expert.0.gate",
        "expert.0.up",
        "expert.0.intermediate",
        "expert.0.down",
        "expert.1.gate",
        "expert.1.up",
        "expert.1.intermediate",
        "expert.1.down",
        "expert.2.gate",
        "expert.2.up",
        "expert.2.intermediate",
        "expert.2.down",
        "expert.3.gate",
        "expert.3.up",
        "expert.3.intermediate",
        "expert.3.down",
        "expert.4.gate",
        "expert.4.up",
        "expert.4.intermediate",
        "expert.4.down",
    )


def test_default_moe_layout_calculates_quantization_indices() -> None:
    layout = DEFAULT_MOE_LAYOUT

    assert [
        layout.scale_index(0, slot)
        for slot in layout.quant_slots
    ] == [1, 2, 3, 4]
    assert [
        layout.scale_index(4, slot)
        for slot in layout.quant_slots
    ] == [17, 18, 19, 20]
    assert [
        layout.quant_slot_for_projection(projection)
        for projection in layout.projections
    ] == ["gate", "up", "down"]


def test_default_moe_layout_calculates_packed_weight_segments() -> None:
    layout = DEFAULT_MOE_LAYOUT

    segments = layout.weight_segments(256, 128)

    assert [
        (
            item.expert_id,
            item.projection,
            item.offset,
            item.length,
            item.rows,
            item.columns,
        )
        for item in segments
    ] == [
        (0, "gate_proj", 0, 32768, 128, 256),
        (0, "up_proj", 32768, 32768, 128, 256),
        (0, "down_proj", 65536, 32768, 256, 128),
        (1, "gate_proj", 98304, 32768, 128, 256),
        (1, "up_proj", 131072, 32768, 128, 256),
        (1, "down_proj", 163840, 32768, 256, 128),
        (2, "gate_proj", 196608, 32768, 128, 256),
        (2, "up_proj", 229376, 32768, 128, 256),
        (2, "down_proj", 262144, 32768, 256, 128),
        (3, "gate_proj", 294912, 32768, 128, 256),
        (3, "up_proj", 327680, 32768, 128, 256),
        (3, "down_proj", 360448, 32768, 256, 128),
        (4, "gate_proj", 393216, 32768, 128, 256),
        (4, "up_proj", 425984, 32768, 128, 256),
        (4, "down_proj", 458752, 32768, 256, 128),
    ]
    assert layout.packed_weight_length(256, 128) == 491520
    assert layout.routing_shape(3072) == (3072, 3)


@pytest.mark.parametrize(
    ("operation", "exception", "message"),
    [
        (
            lambda: TinyMoeAbiLayout(routed_expert_count=8),
            ValueError,
            "exactly four routed experts",
        ),
        (
            lambda: TinyMoeAbiLayout(shared_expert_count=2),
            ValueError,
            "exactly one shared expert",
        ),
        (
            lambda: TinyMoeAbiLayout(routed_top_k=4),
            ValueError,
            "top-k 2",
        ),
        (
            lambda: TinyMoeAbiLayout(routed_expert_count=4.0),
            TypeError,
            "must be an integer",
        ),
        (
            lambda: DEFAULT_MOE_LAYOUT.scale_index(5, "gate"),
            ValueError,
            "expert_id must be",
        ),
        (
            lambda: DEFAULT_MOE_LAYOUT.scale_index(True, "gate"),
            TypeError,
            "must be an integer",
        ),
        (
            lambda: DEFAULT_MOE_LAYOUT.weight_index(0, "unknown"),
            ValueError,
            "Unsupported projection",
        ),
        (
            lambda: DEFAULT_MOE_LAYOUT.quant_slot_for_projection(
                "unknown"
            ),
            ValueError,
            "Unsupported projection",
        ),
        (
            lambda: DEFAULT_MOE_LAYOUT.weight_segments(0, 128),
            ValueError,
            "must be positive",
        ),
        (
            lambda: DEFAULT_MOE_LAYOUT.packed_weight_length(256.5, 128),
            TypeError,
            "must be an integer",
        ),
        (
            lambda: DEFAULT_MOE_LAYOUT.routing_shape(0),
            ValueError,
            "must be positive",
        ),
        (
            lambda: DEFAULT_MOE_LAYOUT.routing_shape(True),
            TypeError,
            "must be an integer",
        ),
    ],
)
def test_moe_layout_rejects_values_outside_release_abi(
    operation: Callable[[], object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        operation()
