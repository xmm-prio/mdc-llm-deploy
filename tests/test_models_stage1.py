from __future__ import annotations

import hashlib

import pytest
import torch

from mdc_llm_deploy.models import (
    INITIALIZATION_SEED,
    PREFILL_SEQUENCE_LENGTH,
    TinyQwen3Dense,
    TinyQwen3Moe,
)
from mdc_llm_deploy.utils import release_input_ids

EXPECTED_OUTPUT_HASHES = {
    TinyQwen3Dense: "e4a3d62e196d4cc794c1e4eff522694b24ed1490fde04b0c569fd576c8ce53a7",
    TinyQwen3Moe: "41752b64f4aa0484f8e8eec5d3bb506ec07b85fe573f73d9c5648c62bacaafc3",
}


def _tensor_hash(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_hash(model: torch.nn.Module) -> str:
    return _tensor_hash(*(value for _, value in sorted(model.state_dict().items())))


@pytest.mark.parametrize("model_type", [TinyQwen3Dense, TinyQwen3Moe])
def test_tiny_models_are_seeded_without_global_rng_side_effect(model_type: type[torch.nn.Module]) -> None:
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    first = model_type()
    after = torch.random.get_rng_state()
    second = model_type()

    assert INITIALIZATION_SEED == 20260714
    assert torch.equal(before, after)
    assert _state_hash(first) == _state_hash(second)
    assert not first.training
    assert all(parameter.dtype == torch.float16 for parameter in first.parameters())


@pytest.mark.parametrize("model_type", [TinyQwen3Dense, TinyQwen3Moe])
def test_tiny_models_have_deterministic_forward(model_type: type[torch.nn.Module]) -> None:
    input_ids = torch.from_numpy(release_input_ids()[:, :16].copy())
    first = model_type()
    second = model_type()

    with torch.inference_mode():
        first_output = first(input_ids)
        second_output = second(input_ids)

    assert _tensor_hash(*first_output) == _tensor_hash(*second_output)
    assert torch.equal(first_output.logits, second_output.logits)
    assert torch.equal(first_output.key_cache, second_output.key_cache)
    assert torch.equal(first_output.value_cache, second_output.value_cache)


@pytest.mark.parametrize("model_type", [TinyQwen3Dense, TinyQwen3Moe])
def test_prefill_abi_shape_dtype_and_hash(model_type: type[torch.nn.Module]) -> None:
    model = model_type()
    input_ids = torch.from_numpy(release_input_ids().copy())

    with torch.inference_mode():
        output = model(input_ids)

    assert input_ids.shape == (1, PREFILL_SEQUENCE_LENGTH)
    assert input_ids.dtype == torch.int64
    assert output.logits.shape == (1, 3072, 128)
    assert output.key_cache.shape == (1, 2, 3072, 16)
    assert output.value_cache.shape == (1, 2, 3072, 16)
    assert output.logits.dtype == torch.float16
    assert output.key_cache.dtype == torch.float16
    assert output.value_cache.dtype == torch.float16
    output_hash = _tensor_hash(*output)
    assert len(output_hash) == 64
    assert output_hash == EXPECTED_OUTPUT_HASHES[model_type]


def test_tiny_architecture_is_frozen() -> None:
    dense = TinyQwen3Dense()
    moe = TinyQwen3Moe()

    assert dense.config.vocab_size == 128
    assert dense.config.hidden_size == 64
    assert dense.config.intermediate_size == 128
    assert dense.config.num_hidden_layers == 1
    assert dense.config.num_attention_heads == 4
    assert dense.config.num_key_value_heads == 2
    assert dense.config.head_dim == 16
    assert dense.config.max_position_embeddings == 3072
    assert dense.config._attn_implementation == "eager"
    assert dense.config.attention_dropout == 0.0
    assert dense.config.embedding_dropout == 0.0
    assert moe.config.num_experts == 4
    assert moe.config.num_experts_per_tok == 2
    assert moe.config.moe_intermediate_size == 64
    assert moe.config.num_shared_experts == 1
    assert len(moe.moe.experts) == 4


@pytest.mark.parametrize(
    ("input_ids", "error_type"),
    [
        (torch.zeros(3072, dtype=torch.int64), ValueError),
        (torch.zeros((2, 1), dtype=torch.int64), ValueError),
        (torch.zeros((1, 3073), dtype=torch.int64), ValueError),
        (torch.zeros((1, 1), dtype=torch.int32), TypeError),
    ],
)
def test_input_abi_rejects_invalid_values(
    input_ids: torch.Tensor,
    error_type: type[Exception],
) -> None:
    model = TinyQwen3Dense()
    with pytest.raises(error_type):
        model(input_ids)


def test_model_rejects_non_floating_parameter_dtype() -> None:
    with pytest.raises(TypeError):
        TinyQwen3Dense(dtype=torch.int8)
