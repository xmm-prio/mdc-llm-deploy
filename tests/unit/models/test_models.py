from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mdc_llm_deploy.models import (
    AutoExportModel,
    ExportModelConfig,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)
from mdc_llm_deploy.models.checkpoint import (
    load_safetensors,
    resolve_checkpoint,
)
from tests.support.models.qwen3 import (
    dense_config,
    dense_model,
    moe_config,
    moe_model,
)


def test_dense_model_returns_logits_and_every_layer_cache() -> None:
    model = dense_model(8, layers=2)
    output = model(torch.arange(8).reshape(1, 8))

    assert len(output) == 5
    assert output[0].shape == (1, 8, 128)
    assert all(item.shape == (1, 2, 8, 16) for item in output[1:])
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_top_level_transformers_device_and_dtype_properties() -> None:
    model = dense_model(4)

    assert model.device == next(model.parameters()).device
    assert model.dtype is torch.float32

    model.to(dtype=torch.bfloat16)

    assert model.dtype is torch.bfloat16
    with pytest.raises(AttributeError):
        model.device = torch.device("cpu")  # type: ignore[misc]
    with pytest.raises(AttributeError):
        model.dtype = torch.float16  # type: ignore[misc]


def test_tied_embeddings_are_one_parameter_and_storage() -> None:
    config = dense_config(tie_word_embeddings=True)
    model = Qwen3ForCausalLM(
        config,
        ExportModelConfig(4),
        dtype=torch.float32,
    )

    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert (
        model.lm_head.weight.untyped_storage().data_ptr()
        == model.model.embed_tokens.weight.untyped_storage().data_ptr()
    )
    unique_names = dict(model.named_parameters())
    all_names = dict(model.named_parameters(remove_duplicate=False))
    assert "model.embed_tokens.weight" in unique_names
    assert "lm_head.weight" not in unique_names
    assert all_names["lm_head.weight"] is all_names["model.embed_tokens.weight"]

    model.to(dtype=torch.bfloat16)

    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert model.dtype is torch.bfloat16


def test_dense_model_supports_attention_width_different_from_hidden_size() -> None:
    config = replace(dense_config(), hidden_size=32)
    model = Qwen3ForCausalLM(
        config,
        ExportModelConfig(sequence_length=8),
        dtype=torch.float32,
    )

    output = model(torch.arange(8).reshape(1, 8))

    assert output[0].shape == (1, 8, 128)
    assert model.model.layers[0].self_attn.q_proj.out_features == 64
    assert model.model.layers[0].self_attn.o_proj.in_features == 64


def test_mask_semantics_are_frozen_at_construction() -> None:
    causal = dense_model(8, mask_mode="causal")
    unmasked = dense_model(8, mask_mode="none")

    assert causal.causal_mask is not None
    assert causal.causal_mask.dtype is torch.bool
    assert unmasked.causal_mask is None
    assert causal.cos_cache.shape == (1, 8, 1, 16)
    assert causal.sin_cache.shape == (1, 8, 1, 16)


@pytest.mark.parametrize(
    ("expert_count", "top_k"),
    [(2, 1), (4, 2), (6, 3)],
)
def test_moe_supports_variable_expert_count_and_top_k(
    expert_count: int,
    top_k: int,
) -> None:
    model = moe_model(4, expert_count=expert_count, top_k=top_k)
    output = model(torch.arange(4).reshape(1, 4))
    block = model.model.layers[0].mlp

    assert isinstance(model, Qwen3MoeForCausalLM)
    assert block.expert_weights.shape == (
        expert_count,
        3 * 64 * 32,
    )
    assert output[0].shape == (1, 4, 128)


def test_moe_accepts_int8_expert_major_weights() -> None:
    model = moe_model(4, expert_count=3, top_k=2)
    block = model.model.layers[0].mlp
    weights = block.expert_weights.detach().reshape(3, 3, -1)
    scales = weights.abs().amax(dim=-1).clamp_min(1e-8) / 127
    quantized = torch.round(weights / scales.unsqueeze(-1)).clamp(
        -128, 127
    ).to(torch.int8)

    block.set_packed_weights(
        quantized.reshape_as(block.expert_weights),
        scales=scales,
    )
    output = model(torch.arange(4).reshape(1, 4))

    assert block.expert_weights.dtype is torch.int8
    assert block.quant_scales is not None
    assert output[0].shape == (1, 4, 128)


def test_model_rejects_shape_not_frozen_by_export_config() -> None:
    model = dense_model(8)
    with pytest.raises(ValueError, match="ExportModelConfig"):
        model(torch.arange(4).reshape(1, 4))


def test_auto_model_loads_single_safetensors_checkpoint(tmp_path: Path) -> None:
    source = dense_model(4)
    config = dense_config()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                name: getattr(config, name)
                for name in config.__dataclass_fields__
            }
        ),
        encoding="utf-8",
    )
    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if name not in {"cos_cache", "sin_cache", "causal_mask"}
    }
    save_file(state, tmp_path / "model.safetensors")

    loaded = AutoExportModel.from_pretrained(
        tmp_path,
        ExportModelConfig(4),
        dtype=torch.float32,
    )

    assert isinstance(loaded, Qwen3ForCausalLM)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_auto_model_restores_tied_parameter_after_safetensors_load(
    tmp_path: Path,
) -> None:
    config = dense_config(tie_word_embeddings=True)
    source = Qwen3ForCausalLM(
        config,
        ExportModelConfig(4),
        dtype=torch.float32,
    )
    with torch.no_grad():
        source.model.embed_tokens.weight.fill_(0.25)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                name: getattr(config, name)
                for name in config.__dataclass_fields__
            }
        ),
        encoding="utf-8",
    )
    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if name
        not in {
            "cos_cache",
            "sin_cache",
            "causal_mask",
            "lm_head.weight",
        }
    }
    save_file(state, tmp_path / "model.safetensors")

    loaded = AutoExportModel.from_pretrained(
        tmp_path,
        ExportModelConfig(4),
        dtype=torch.float32,
    )

    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert (
        loaded.lm_head.weight.untyped_storage().data_ptr()
        == loaded.model.embed_tokens.weight.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(
        loaded.model.embed_tokens.weight,
        torch.full_like(loaded.model.embed_tokens.weight, 0.25),
    )


def test_auto_model_tolerates_checkpoint_state_mismatches(tmp_path: Path) -> None:
    source = dense_model(4)
    config = dense_config()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                name: getattr(config, name)
                for name in config.__dataclass_fields__
            }
        ),
        encoding="utf-8",
    )
    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if name
        not in {
            "cos_cache",
            "sin_cache",
            "causal_mask",
            "model.embed_tokens.weight",
        }
    }
    state["model.norm.weight"] = torch.full_like(
        state["model.norm.weight"],
        3,
    )
    state["lm_head.weight"] = torch.ones(1, 1)
    state["unexpected.weight"] = torch.ones(1)
    save_file(state, tmp_path / "model.safetensors")

    loaded = AutoExportModel.from_pretrained(
        tmp_path,
        ExportModelConfig(4),
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        loaded.model.norm.weight,
        torch.full_like(loaded.model.norm.weight, 3),
    )
    assert loaded.model.embed_tokens.weight.shape == (128, 64)
    assert loaded.lm_head.weight.shape == (128, 64)


def test_loader_reads_indexed_safetensors_shards(tmp_path: Path) -> None:
    save_file({"left": torch.ones(2)}, tmp_path / "part-1.safetensors")
    save_file({"right": torch.zeros(3)}, tmp_path / "part-2.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "left": "part-1.safetensors",
                    "right": "part-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    state = load_safetensors(tmp_path)

    assert set(state) == {"left", "right"}


@pytest.mark.parametrize("as_string", [False, True], ids=["path", "str"])
def test_checkpoint_resolver_uses_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    as_string: bool,
) -> None:
    import huggingface_hub

    def fail_snapshot_download(**kwargs: object) -> str:
        pytest.fail(f"unexpected Hub download: {kwargs}")

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        fail_snapshot_download,
    )
    source: str | Path = str(tmp_path) if as_string else tmp_path

    resolved = resolve_checkpoint(source)

    assert resolved == tmp_path.resolve()


@pytest.mark.parametrize(
    "filename",
    ["config.json", "model.safetensors", "notes.txt"],
)
@pytest.mark.parametrize("as_string", [False, True], ids=["path", "str"])
def test_checkpoint_resolver_rejects_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    as_string: bool,
) -> None:
    import huggingface_hub

    def fail_snapshot_download(**kwargs: object) -> str:
        pytest.fail(f"unexpected Hub download: {kwargs}")

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        fail_snapshot_download,
    )
    candidate = tmp_path / filename
    candidate.touch()
    source: str | Path = str(candidate) if as_string else candidate

    with pytest.raises(ValueError) as exc_info:
        resolve_checkpoint(source)

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == (
        f"Local checkpoint source must be a directory: {Path(source)}"
    )


def test_checkpoint_resolver_treats_missing_path_as_hub_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        fake_snapshot_download,
    )
    source = tmp_path / "missing-checkpoint"

    resolved = resolve_checkpoint(
        source,
        revision="missing-revision",
        local_files_only=True,
    )

    assert resolved == tmp_path
    assert calls == [
        {
            "repo_id": str(source),
            "revision": "missing-revision",
            "allow_patterns": ["*.json", "*.safetensors"],
            "local_files_only": True,
        }
    ]


def test_checkpoint_resolver_uses_hub_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        fake_snapshot_download,
    )

    resolved = resolve_checkpoint(
        "org/qwen",
        revision="fixed",
        local_files_only=True,
    )

    assert resolved == tmp_path
    assert calls == [
        {
            "repo_id": "org/qwen",
            "revision": "fixed",
            "allow_patterns": ["*.json", "*.safetensors"],
            "local_files_only": True,
        }
    ]


def test_auto_model_packs_int8_moe_checkpoint(
    tmp_path: Path,
) -> None:
    source = moe_model(4, expert_count=3, top_k=2)
    config = moe_config(expert_count=3, top_k=2)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                name: getattr(config, name)
                for name in config.__dataclass_fields__
            }
        ),
        encoding="utf-8",
    )
    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if name
        not in {
            "cos_cache",
            "sin_cache",
            "causal_mask",
            "model.layers.0.mlp.expert_weights",
        }
    }
    length = config.hidden_size * config.moe_intermediate_size
    for expert_id in range(config.num_experts):
        for projection_id, projection in enumerate(
            ("gate_proj", "up_proj", "down_proj")
        ):
            prefix = (
                f"model.layers.0.mlp.experts.{expert_id}.{projection}"
            )
            state[f"{prefix}.weight"] = torch.full(
                (
                    config.moe_intermediate_size,
                    config.hidden_size,
                )
                if projection != "down_proj"
                else (
                    config.hidden_size,
                    config.moe_intermediate_size,
                ),
                expert_id * 3 + projection_id + 1,
                dtype=torch.int8,
            )
            state[f"{prefix}.weight_scale"] = torch.tensor(
                0.01 * (expert_id * 3 + projection_id + 1)
            )
    save_file(state, tmp_path / "model.safetensors")

    loaded = AutoExportModel.from_pretrained(
        tmp_path,
        ExportModelConfig(4),
        dtype=torch.float32,
    )
    block = loaded.model.layers[0].mlp

    assert isinstance(loaded, Qwen3MoeForCausalLM)
    assert block.expert_weights.dtype is torch.int8
    assert block.expert_weights.shape == (3, 3 * length)
    assert block.quant_scales is not None
    assert block.quant_scales.shape == (3, 3)


def test_auto_model_loads_transformers_packed_moe_checkpoint(
    tmp_path: Path,
) -> None:
    source = moe_model(4, expert_count=3, top_k=2)
    config = moe_config(expert_count=3, top_k=2)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                name: getattr(config, name)
                for name in config.__dataclass_fields__
            }
        ),
        encoding="utf-8",
    )
    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if name
        not in {
            "cos_cache",
            "sin_cache",
            "causal_mask",
            "model.layers.0.mlp.expert_weights",
        }
    }
    packed = source.model.layers[0].mlp.expert_weights.detach()
    projections = packed.reshape(3, 3, 32, 64)
    state["model.layers.0.mlp.experts.gate_up_proj"] = torch.cat(
        (projections[:, 0], projections[:, 1]),
        dim=1,
    )
    state["model.layers.0.mlp.experts.down_proj"] = (
        packed[:, 2 * 32 * 64 :].reshape(3, 64, 32)
    ).contiguous()
    save_file(state, tmp_path / "model.safetensors")

    loaded = AutoExportModel.from_pretrained(
        tmp_path,
        ExportModelConfig(4),
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        loaded.model.layers[0].mlp.expert_weights,
        packed,
    )


def test_dense_logits_and_kv_align_with_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    hf_config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_bias=False,
        tie_word_embeddings=False,
        use_cache=True,
    )
    reference = transformers.Qwen3ForCausalLM(hf_config).eval().float()
    model = dense_model(8)
    own_state = model.state_dict()
    shared = {
        name: value
        for name, value in reference.state_dict().items()
        if name in own_state and own_state[name].shape == value.shape
    }
    model.load_state_dict(shared, strict=False)
    input_ids = torch.arange(8).reshape(1, 8)

    with torch.inference_mode():
        expected = reference(input_ids, use_cache=True)
        actual = model(input_ids)

    torch.testing.assert_close(actual[0], expected.logits, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        actual[1],
        expected.past_key_values.layers[0].keys,
        atol=1e-5,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        actual[2],
        expected.past_key_values.layers[0].values,
        atol=1e-5,
        rtol=1e-4,
    )


def test_moe_logits_and_kv_align_with_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    hf_config = transformers.Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=3,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_bias=False,
        tie_word_embeddings=False,
        use_cache=True,
    )
    hf_config._experts_implementation = "eager"
    reference = transformers.Qwen3MoeForCausalLM(
        hf_config
    ).eval().float()
    model = moe_model(8, expert_count=3, top_k=2)
    reference_state = reference.state_dict()
    own_state = model.state_dict()
    shared = {
        name: value
        for name, value in reference_state.items()
        if name in own_state and own_state[name].shape == value.shape
    }
    model.load_state_dict(shared, strict=False)
    gate_up = reference_state[
        "model.layers.0.mlp.experts.gate_up_proj"
    ]
    down = reference_state["model.layers.0.mlp.experts.down_proj"]
    rows = [
        torch.cat(
            (
                    gate_up[expert_id, :32].reshape(-1),
                    gate_up[expert_id, 32:].reshape(-1),
                down[expert_id].reshape(-1),
            )
        )
        for expert_id in range(3)
    ]
    model.model.layers[0].mlp.set_packed_weights(torch.stack(rows))
    input_ids = torch.arange(8).reshape(1, 8)

    with torch.inference_mode():
        expected = reference(input_ids, use_cache=True)
        actual = model(input_ids)

    torch.testing.assert_close(
        actual[0],
        expected.logits,
        atol=1e-5,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        actual[1],
        expected.past_key_values.layers[0].keys,
        atol=1e-5,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        actual[2],
        expected.past_key_values.layers[0].values,
        atol=1e-5,
        rtol=1e-4,
    )
