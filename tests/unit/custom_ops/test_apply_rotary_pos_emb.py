from __future__ import annotations

import pytest
import torch

from mdc_llm_deploy.custom_ops.apply_rotary_pos_emb import ApplyRotaryPosEmb


def _inputs(
    layout: int,
    *,
    rotary_dim: int = 8,
    head_dim: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shapes = {
        1: ((2, 3, 4, head_dim), (2, 3, 2, head_dim), (1, 3, 1, rotary_dim)),
        2: ((3, 2, 4, head_dim), (3, 2, 2, head_dim), (3, 1, 1, rotary_dim)),
        3: ((2, 4, 3, head_dim), (2, 2, 3, head_dim), (1, 1, 3, rotary_dim)),
        4: ((6, 4, head_dim), (6, 2, head_dim), (6, 1, rotary_dim)),
    }
    query_shape, key_shape, rope_shape = shapes[layout]
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(query_shape, generator=generator)
    key = torch.randn(key_shape, generator=generator)
    angles = torch.randn(rope_shape, generator=generator)
    return query, key, angles.cos(), angles.sin()


def _manual_rotate(input: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "half":
        first, second = input.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)
    if mode == "interleave":
        pairs = input.reshape(*input.shape[:-1], -1, 2)
        return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    first, second, third, fourth = input.chunk(4, dim=-1)
    return torch.cat((-second, first, -fourth, third), dim=-1)


@pytest.mark.parametrize("layout", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["half", "interleave", "quarter"])
def test_cpu_matches_formula_for_every_layout_and_mode(layout: int, mode: str) -> None:
    query, key, cos, sin = _inputs(layout)

    query_out, key_out = ApplyRotaryPosEmb.cpu(query, key, cos, sin, layout, mode)

    expected_query = query.float() * cos.float() + _manual_rotate(query.float(), mode) * sin.float()
    expected_key = key.float() * cos.float() + _manual_rotate(key.float(), mode) * sin.float()
    torch.testing.assert_close(query_out, expected_query)
    torch.testing.assert_close(key_out, expected_key)


def test_cpu_uses_fp32_and_preserves_partial_tail_and_dtype() -> None:
    query, key, cos, sin = _inputs(1, rotary_dim=4, head_dim=8)
    query = query.half()
    key = key.half()
    cos = cos.half()
    sin = sin.half()

    query_out, key_out = ApplyRotaryPosEmb.cpu(query, key, cos, sin, 1, "quarter")

    expected_rotary = (
        query[..., :4].float() * cos.float()
        + _manual_rotate(query[..., :4].float(), "quarter") * sin.float()
    ).half()
    torch.testing.assert_close(query_out[..., :4], expected_rotary)
    assert torch.equal(query_out[..., 4:], query[..., 4:])
    assert torch.equal(key_out[..., 4:], key[..., 4:])
    assert query_out.dtype == query.dtype
    assert key_out.dtype == key.dtype


def test_fake_returns_query_and_key_metadata() -> None:
    query, key, cos, sin = _inputs(3)
    query = query.to(device="meta")
    key = key.to(device="meta")
    cos = cos.to(device="meta")
    sin = sin.to(device="meta")

    query_out, key_out = ApplyRotaryPosEmb.fake(query, key, cos, sin, 3, "half")

    assert query_out.shape == query.shape
    assert key_out.shape == key.shape
    assert query_out.device.type == "meta"
    assert key_out.device.type == "meta"


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (lambda values: (*values[:4], 0, "half"), ValueError, "layout"),
        (lambda values: (*values[:4], 1, "bad"), ValueError, "rotary_mode"),
        (
            lambda values: (values[0], values[1], values[2].half(), values[3], 1, "half"),
            TypeError,
            "same dtype",
        ),
        (
            lambda values: (values[0], values[1], values[2], values[3][..., :4], 1, "half"),
            ValueError,
            "same shape",
        ),
        (
            lambda values: (values[0], values[1], values[2][..., :3], values[3][..., :3], 1, "half"),
            ValueError,
            "divisible by 2",
        ),
        (
            lambda values: (values[0], values[1], values[2][..., :6], values[3][..., :6], 1, "quarter"),
            ValueError,
            "divisible by 4",
        ),
    ],
)
def test_invalid_contract_is_rejected(
    mutate: object,
    error: type[Exception],
    message: str,
) -> None:
    values = _inputs(1)
    arguments = mutate(values)  # type: ignore[operator]

    with pytest.raises(error, match=message):
        ApplyRotaryPosEmb.cpu(*arguments)


def test_nonfinite_input_is_rejected() -> None:
    query, key, cos, sin = _inputs(1)
    query[0, 0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="query must contain only finite values"):
        ApplyRotaryPosEmb.cpu(query, key, cos, sin)
