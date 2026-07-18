from __future__ import annotations

from dataclasses import replace

import pytest

from mdc_llm_deploy.errors import GraphStateError
from mdc_llm_deploy.graph.metadata import (
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    TensorAbi,
    derive_artifact_io_abi,
    order_attention_boundaries,
)


def _metadata(
    *,
    stage: GraphStage = GraphStage.FLOAT_PREFILL,
    layers: int = 2,
    save_kv_cache: bool | None = True,
) -> GraphMetadata:
    sequence_length = 4 if stage.is_prefill else 1
    properties = (
        {} if save_kv_cache is None else {"save_kv_cache": save_kv_cache}
    )
    cache_shape = (1, 2, 4, 8)
    cache_inputs = (
        ()
        if stage.is_prefill
        else tuple(
            TensorAbi(name, "float16", (1, 2, 3, 8))
            for layer_id in range(layers)
            for name in (
                f"past.{layer_id}.key",
                f"past.{layer_id}.value",
            )
        )
    )
    return GraphMetadata(
        schema_version=1,
        stage=stage,
        model_kind="dense",
        input_abi=(
            TensorAbi("input_ids", "int64", (1, sequence_length)),
            *cache_inputs,
        ),
        output_abi=(
            TensorAbi("logits", "float16", (1, sequence_length, 32)),
            *(
                TensorAbi(name, "float16", cache_shape)
                for layer_id in range(layers)
                for name in (
                    f"present.{layer_id}.key",
                    f"present.{layer_id}.value",
                )
            ),
        ),
        boundaries=tuple(
            FusionBoundary("attention", f"model.layers.{layer_id}.self_attn")
            for layer_id in range(layers)
        ),
        sequence_length=4,
        absolute_position=None if stage.is_prefill else 3,
        properties=properties,
    )


@pytest.mark.parametrize(
    ("stage", "save_kv_cache", "output_names"),
    [
        (
            GraphStage.FLOAT_PREFILL,
            True,
            (
                "logits",
                "present.0.key",
                "present.0.value",
                "present.1.key",
                "present.1.value",
            ),
        ),
        (GraphStage.FLOAT_PREFILL, False, ("logits",)),
        (
            GraphStage.FLOAT_DECODE,
            True,
            (
                "logits",
                "present.0.key",
                "present.0.value",
                "present.1.key",
                "present.1.value",
            ),
        ),
        (GraphStage.FLOAT_DECODE, False, ("logits",)),
    ],
)
def test_artifact_abi_derives_stage_and_public_cache_policy(
    stage: GraphStage,
    save_kv_cache: bool,
    output_names: tuple[str, ...],
) -> None:
    result = derive_artifact_io_abi(
        _metadata(stage=stage, save_kv_cache=save_kv_cache)
    )

    assert tuple(item.name for item in result.outputs) == output_names
    assert result.layer_count == 2
    assert result.save_kv_cache is save_kv_cache
    if stage.is_prefill:
        assert tuple(item.name for item in result.inputs) == ("input_ids",)
    else:
        assert tuple(item.name for item in result.inputs) == (
            "input_ids",
            "past.0.key",
            "past.0.value",
            "past.1.key",
            "past.1.value",
        )


def test_missing_save_kv_cache_metadata_keeps_legacy_logits_only_contract() -> None:
    result = derive_artifact_io_abi(_metadata(save_kv_cache=None))

    assert result.save_kv_cache is False
    assert tuple(item.name for item in result.outputs) == ("logits",)


def test_twelve_attention_layers_use_numeric_not_lexical_order() -> None:
    boundaries = tuple(
        FusionBoundary("attention", f"model.layers.{layer_id}.self_attn")
        for layer_id in (0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9)
    )

    ordered = order_attention_boundaries(boundaries)

    assert tuple(
        int(item.fqn.split(".")[2]) for item in ordered
    ) == tuple(range(12))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: replace(
                value,
                output_abi=(
                    value.output_abi[0],
                    value.output_abi[2],
                    value.output_abi[1],
                    *value.output_abi[3:],
                ),
            ),
            "ordered contiguous key/value pairs",
        ),
        (
            lambda value: replace(value, boundaries=value.boundaries[:1]),
            "boundary count",
        ),
        (
            lambda value: replace(
                value,
                properties={"save_kv_cache": 1},
            ),
            "must be a bool",
        ),
    ],
)
def test_artifact_abi_rejects_invalid_layer_contract(
    mutate: object,
    message: str,
) -> None:
    candidate = mutate(_metadata())  # type: ignore[operator]

    with pytest.raises(GraphStateError, match=message):
        derive_artifact_io_abi(candidate)


def test_attention_layers_must_be_unique_and_contiguous() -> None:
    value = _metadata()
    duplicated = (
        value.boundaries[0],
        replace(value.boundaries[1], fqn="model.layers.0.other_attn"),
    )

    with pytest.raises(GraphStateError, match="unique and contiguous"):
        derive_artifact_io_abi(replace(value, boundaries=duplicated))
