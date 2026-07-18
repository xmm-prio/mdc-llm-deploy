"""Semantic acceptance for serialized local release-matrix artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import onnx

from mdc_llm_deploy.capabilities import (
    Algorithm,
    Capability,
    ModelKind,
    Phase,
    Target,
)
from mdc_llm_deploy.errors import GraphStateError, OnnxExportError
from mdc_llm_deploy.graph.metadata import (
    SAVE_KV_CACHE_PROPERTY,
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    TensorAbi,
    derive_artifact_io_abi,
    resolve_save_kv_cache,
)
from mdc_llm_deploy.onnx.validation.metadata import ValidatedMetadata
from mdc_llm_deploy.onnx.validation.model import (
    load_validated_mdc_artifact,
)


@dataclass(frozen=True, slots=True)
class ReleaseModelContract:
    """Fixed ABI constants shared by the two local release models."""

    layer_count: int = 2
    key_value_heads: int = 2
    head_dim: int = 64
    vocab_size: int = 128

    def cache_dtype(self, capability: Capability) -> int:
        """Return expected decode-cache element type."""
        if (
            capability.algorithm is Algorithm.MINMAX
            and capability.target is Target.ATTENTION
        ):
            return onnx.TensorProto.INT8
        return onnx.TensorProto.FLOAT16


@dataclass(frozen=True, slots=True)
class ReleaseValidationEvidence:
    """Immutable evidence captured by release semantic acceptance."""

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    operator_counts: tuple[tuple[str, int], ...]
    declared_targets: frozenset[str]
    observed_quantized_targets: frozenset[str]


_CONTRACT = ReleaseModelContract()
_STAGE_BY_COMBINATION = {
    (Algorithm.FP16, Phase.PREFILL): GraphStage.FLOAT_PREFILL,
    (Algorithm.FP16, Phase.DECODE): GraphStage.FLOAT_DECODE,
    (Algorithm.MINMAX, Phase.PREFILL): GraphStage.QUANTIZED_PREFILL,
    (Algorithm.MINMAX, Phase.DECODE): GraphStage.QUANTIZED_DECODE,
}


def _identity(capability: Capability) -> str:
    target = capability.target.value if capability.target is not None else "baseline"
    return (
        f"model={capability.model.value}, "
        f"algorithm={capability.algorithm.value}, "
        f"target={target}, phase={capability.phase.value}, "
        f"mask_mode={capability.mask_mode.value}"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OnnxExportError(message)


def _quantized_target(capability: Capability) -> str:
    if capability.target is None:
        raise OnnxExportError("Quantized release capability must declare a target")
    return capability.target.value


def _tensor_signature(value: onnx.ValueInfoProto) -> tuple[int, tuple[int, ...]]:
    tensor_type = value.type.tensor_type
    _require(tensor_type.elem_type != 0, f"Tensor {value.name!r} lacks a dtype")
    dimensions: list[int] = []
    for dimension in tensor_type.shape.dim:
        _require(
            dimension.HasField("dim_value") and dimension.dim_value > 0,
            f"Tensor {value.name!r} must have a positive static shape",
        )
        dimensions.append(dimension.dim_value)
    return tensor_type.elem_type, tuple(dimensions)


def _save_kv_cache(metadata: ValidatedMetadata) -> bool:
    raw_value: object = metadata.properties.get(SAVE_KV_CACHE_PROPERTY, False)
    if isinstance(raw_value, str):
        serialized_values = {"true": True, "false": False}
        if raw_value not in serialized_values:
            raise OnnxExportError(
                "Release save_kv_cache metadata must be 'true' or 'false'"
            )
        raw_value = serialized_values[raw_value]
    try:
        return resolve_save_kv_cache({SAVE_KV_CACHE_PROPERTY: raw_value})
    except GraphStateError as error:
        raise OnnxExportError(f"Invalid release artifact ABI: {error}") from error


def _tensor_abi(value: onnx.ValueInfoProto, *, name: str | None = None) -> TensorAbi:
    _, shape = _tensor_signature(value)
    return TensorAbi(name or value.name, "float16", shape)


def _artifact_abi(
    model: onnx.ModelProto,
    metadata: ValidatedMetadata,
) -> tuple[TensorAbi, ...]:
    save_kv_cache = _save_kv_cache(metadata)
    inputs = tuple(_tensor_abi(item) for item in model.graph.input)
    public_outputs = tuple(_tensor_abi(item) for item in model.graph.output)
    stage = GraphStage(metadata.stage)
    if save_kv_cache or stage.is_prefill:
        internal_outputs = public_outputs
    else:
        internal_outputs = (
            public_outputs[0],
            *(
                _tensor_abi(
                    item,
                    name=f"present.{item.name.removeprefix('past.')}",
                )
                for item in model.graph.input[1:]
            ),
        )
    layer_count = (len(internal_outputs) - 1) // 2
    contract_metadata = GraphMetadata(
        schema_version=1,
        stage=stage,
        model_kind=metadata.properties["mdc.model_kind"],
        input_abi=inputs,
        output_abi=internal_outputs,
        boundaries=tuple(
            FusionBoundary(
                "attention",
                f"model.layers.{layer_id}.self_attn",
            )
            for layer_id in range(layer_count)
        ),
        properties={SAVE_KV_CACHE_PROPERTY: save_kv_cache},
    )
    try:
        return derive_artifact_io_abi(contract_metadata).outputs
    except GraphStateError as error:
        raise OnnxExportError(f"Invalid release artifact ABI: {error}") from error


def _validate_metadata_contract(
    metadata: ValidatedMetadata,
    capability: Capability,
) -> None:
    expected_algorithm = frozenset({capability.algorithm.value})
    expected_targets = frozenset(
        {
            Algorithm.FP16.value
            if capability.algorithm is Algorithm.FP16
            else _quantized_target(capability)
        }
    )
    _require(
        metadata.properties["mdc.model_kind"] == capability.model.value,
        "Release model kind does not match capability",
    )
    _require(
        metadata.mask_mode == capability.mask_mode.value,
        "Release mask mode does not match capability",
    )
    _require(
        metadata.stage == _STAGE_BY_COMBINATION[
            (capability.algorithm, capability.phase)
        ].value,
        "Release graph stage does not match capability",
    )
    _require(
        metadata.algorithms == expected_algorithm,
        "Release algorithm does not match capability",
    )
    _require(
        metadata.targets == expected_targets,
        "Release target does not match capability",
    )


def _validate_io_contract(
    model: onnx.ModelProto,
    metadata: ValidatedMetadata,
    capability: Capability,
    contract: ReleaseModelContract,
) -> None:
    inputs = tuple(model.graph.input)
    outputs = tuple(model.graph.output)
    expected_outputs = _artifact_abi(model, metadata)
    _require(
        tuple(item.name for item in outputs)
        == tuple(item.name for item in expected_outputs),
        "Release ONNX outputs do not match artifact ABI",
    )
    save_kv_cache = _save_kv_cache(metadata)
    if capability.phase is Phase.PREFILL:
        _require(
            tuple(item.name for item in inputs) == ("input_ids",),
            "Prefill inputs must be exactly ('input_ids',)",
        )
        input_dtype, input_shape = _tensor_signature(inputs[0])
        output_dtype, output_shape = _tensor_signature(outputs[0])
        _require(
            input_dtype == onnx.TensorProto.INT64
            and len(input_shape) == 2
            and input_shape[0] == 1
            and input_shape[1] >= 2,
            "Prefill input_ids ABI is invalid",
        )
        _require(
            output_dtype == onnx.TensorProto.FLOAT16
            and output_shape == (1, input_shape[1], contract.vocab_size),
            "Prefill logits ABI is invalid",
        )
        if save_kv_cache:
            expected_cache = (
                contract.cache_dtype(capability),
                (
                    1,
                    contract.key_value_heads,
                    input_shape[1],
                    contract.head_dim,
                ),
            )
            _require(
                len(outputs) == 1 + contract.layer_count * 2
                and all(
                    _tensor_signature(item) == expected_cache
                    for item in outputs[1:]
                ),
                "Prefill present cache ABI is invalid",
            )
        return

    _require(
        len(inputs) == 1 + contract.layer_count * 2,
        "Decode inputs are incomplete or out of order",
    )
    _require(
        _tensor_signature(inputs[0]) == (onnx.TensorProto.INT64, (1, 1)),
        "Decode input_ids ABI is invalid",
    )
    cache_signatures = tuple(_tensor_signature(item) for item in inputs[1:])
    cache_length = cache_signatures[0][1][2]
    expected_cache = (
        contract.cache_dtype(capability),
        (1, contract.key_value_heads, cache_length, contract.head_dim),
    )
    _require(
        cache_length >= 1
        and all(signature == expected_cache for signature in cache_signatures),
        "Decode cache ABI is invalid",
    )
    _require(
        _tensor_signature(outputs[0])
        == (onnx.TensorProto.FLOAT16, (1, 1, contract.vocab_size)),
        "Decode logits ABI is invalid",
    )
    if save_kv_cache:
        expected_present_cache = (
            contract.cache_dtype(capability),
            (
                1,
                contract.key_value_heads,
                cache_length + 1,
                contract.head_dim,
            ),
        )
        _require(
            len(outputs) == 1 + contract.layer_count * 2
            and all(
                _tensor_signature(item) == expected_present_cache
                for item in outputs[1:]
            ),
            "Decode present cache ABI is invalid",
        )


def _validate_operator_contract(
    counts: Counter[str],
    capability: Capability,
) -> None:
    _require(
        counts["FusedInferAttentionScore"] == _CONTRACT.layer_count,
        "Release graph must contain one attention node per layer",
    )
    _require(
        counts["ApplyRotaryPosEmb"] == _CONTRACT.layer_count,
        "Release graph must contain one rotary node per layer",
    )
    expected_moe = _CONTRACT.layer_count if capability.model is ModelKind.MOE else 0
    _require(
        counts["MoeExpert"] == expected_moe,
        "Release graph has invalid MoeExpert layer coverage",
    )
    _require(
        counts["NPURmsNorm"] > 0,
        "Release graph must contain NPURmsNorm nodes",
    )


def _validate_release_artifact(
    path: str | Path,
    capability: Capability,
) -> ReleaseValidationEvidence:
    artifact = load_validated_mdc_artifact(str(path))
    model = artifact.model
    metadata = artifact.metadata
    counts = Counter(dict(artifact.topology.operator_counts))
    _validate_metadata_contract(metadata, capability)
    _validate_io_contract(model, metadata, capability, _CONTRACT)
    _validate_operator_contract(counts, capability)
    observed_targets = artifact.topology.observed_quantized_targets
    expected_observed = (
        frozenset()
        if capability.algorithm is Algorithm.FP16
        else frozenset({_quantized_target(capability)})
    )
    _require(
        observed_targets == expected_observed,
        "Observed quantized targets do not match capability",
    )
    return ReleaseValidationEvidence(
        input_names=tuple(item.name for item in model.graph.input),
        output_names=tuple(item.name for item in model.graph.output),
        operator_counts=tuple(sorted(counts.items())),
        declared_targets=metadata.targets,
        observed_quantized_targets=observed_targets,
    )


def validate_release_artifact(
    path: str | Path,
    capability: Capability,
) -> ReleaseValidationEvidence:
    """Validate one serialized artifact against its release capability."""
    try:
        return _validate_release_artifact(path, capability)
    except OnnxExportError as error:
        raise OnnxExportError(
            f"Release artifact validation failed ({_identity(capability)}): {error}"
        ) from error


__all__ = [
    "ReleaseModelContract",
    "ReleaseValidationEvidence",
    "validate_release_artifact",
]
