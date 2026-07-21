from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from mdc_llm_deploy.custom_ops.apply_rotary_pos_emb import (
    apply_rotary_pos_emb,
    cpu,
    fake,
)


def _inputs(
    layout: int,
    *,
    rotary_dim: int = 8,
    head_dim: int = 8,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shapes = {
        1: ((2, 3, 4, head_dim), (2, 3, 2, head_dim), (1, 3, 1, rotary_dim)),
        2: ((3, 2, 4, head_dim), (3, 2, 2, head_dim), (3, 1, 1, rotary_dim)),
        3: ((2, 4, 3, head_dim), (2, 2, 3, head_dim), (1, 1, 3, rotary_dim)),
        4: ((6, 4, head_dim), (6, 2, head_dim), (6, 1, rotary_dim)),
    }
    query_shape, key_shape, rope_shape = shapes[layout]
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(query_shape, generator=generator, dtype=dtype)
    key = torch.randn(key_shape, generator=generator, dtype=dtype)
    angles = torch.randn(rope_shape, generator=generator, dtype=dtype)
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

    query_out, key_out = cpu(query, key, cos, sin, layout, mode)

    expected_query = query.float() * cos.float() + _manual_rotate(query.float(), mode) * sin.float()
    expected_key = key.float() * cos.float() + _manual_rotate(key.float(), mode) * sin.float()
    torch.testing.assert_close(query_out, expected_query)
    torch.testing.assert_close(key_out, expected_key)


def test_torch_contract_preserves_partial_rotation_and_large_head_dimension() -> None:
    query, key, cos, sin = _inputs(1, rotary_dim=4, head_dim=1028, dtype=torch.float16)

    query_out, key_out = apply_rotary_pos_emb(query, key, cos, sin, 1, "quarter")

    assert torch.equal(query_out[..., 4:], query[..., 4:])
    assert torch.equal(key_out[..., 4:], key[..., 4:])
    assert query_out.dtype == query.dtype
    assert key_out.dtype == key.dtype


def test_fake_and_compile_preserve_metadata_and_results() -> None:
    inputs = _inputs(3)
    expected = apply_rotary_pos_emb(*inputs, 3, "interleave")

    with FakeTensorMode() as mode:
        fake_inputs = tuple(mode.from_tensor(value) for value in inputs)
        fake_outputs = apply_rotary_pos_emb(*fake_inputs, 3, "interleave")
    compiled = torch.compile(
        lambda *values: apply_rotary_pos_emb(*values, 3, "interleave"),
        backend="eager",
        fullgraph=True,
    )
    actual = compiled(*inputs)

    assert fake_outputs[0].shape == inputs[0].shape
    assert fake_outputs[1].shape == inputs[1].shape
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_opcheck_covers_registered_torch_operator() -> None:
    torch.library.opcheck(
        apply_rotary_pos_emb,
        (*_inputs(1), 1, "half"),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )


def test_import_registers_torch_operator_without_onnx_schema() -> None:
    code = """
import onnx
from mdc_llm_deploy.custom_ops.apply_rotary_pos_emb import apply_rotary_pos_emb
try:
    onnx.defs.get_schema("ApplyRotaryPosEmb", 18, "")
except onnx.defs.SchemaError:
    print(f"schema-absent:{apply_rotary_pos_emb._name}")
else:
    raise AssertionError("operator import unexpectedly registered ONNX schema")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "schema-absent:apply_rotary_pos_emb"


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
            lambda values: (
                values[0],
                values[1],
                values[2][..., :3],
                values[3][..., :3],
                1,
                "half",
            ),
            ValueError,
            "divisible by 2",
        ),
    ],
)
def test_invalid_torch_contract_is_rejected(
    mutate: object,
    error: type[Exception],
    message: str,
) -> None:
    arguments = mutate(_inputs(1))  # type: ignore[operator]

    with pytest.raises(error, match=message):
        cpu(*arguments)


def test_fake_kernel_does_not_read_values() -> None:
    query, key, cos, sin = _inputs(1)
    query[0, 0, 0, 0] = torch.nan

    outputs = fake(query, key, cos, sin)

    assert outputs[0].shape == query.shape
    assert outputs[1].shape == key.shape
