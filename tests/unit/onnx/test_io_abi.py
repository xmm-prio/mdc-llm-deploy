from __future__ import annotations

import onnx
import pytest
from onnx import TensorProto, helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.graph.metadata import (
    FusionBoundary,
    GraphMetadata,
    GraphStage,
    TensorAbi,
)
from mdc_llm_deploy.onnx.validation.io import validate_io_abi


def _metadata(*, save_kv_cache: bool, decode: bool = False) -> GraphMetadata:
    stage = GraphStage.FLOAT_DECODE if decode else GraphStage.FLOAT_PREFILL
    input_abi = [TensorAbi("input_ids", "int64", (1, 1 if decode else 4))]
    if decode:
        input_abi.extend(
            (
                TensorAbi("past.0.key", "float16", (1, 2, 3, 8)),
                TensorAbi("past.0.value", "float16", (1, 2, 3, 8)),
            )
        )
    return GraphMetadata(
        schema_version=1,
        stage=stage,
        model_kind="dense",
        input_abi=tuple(input_abi),
        output_abi=(
            TensorAbi("logits", "float16", (1, 1 if decode else 4, 32)),
            TensorAbi("present.0.key", "float16", (1, 2, 4, 8)),
            TensorAbi("present.0.value", "float16", (1, 2, 4, 8)),
        ),
        boundaries=(
            FusionBoundary("attention", "model.layers.0.self_attn"),
        ),
        sequence_length=4,
        absolute_position=3 if decode else None,
        properties={"save_kv_cache": save_kv_cache},
    )


def _value_info(entry: TensorAbi) -> onnx.ValueInfoProto:
    dtype = {
        "float16": TensorProto.FLOAT16,
        "int64": TensorProto.INT64,
    }[entry.dtype]
    return helper.make_tensor_value_info(entry.name, dtype, entry.shape)


def _model(metadata: GraphMetadata) -> onnx.ModelProto:
    outputs = (
        metadata.output_abi
        if metadata.properties["save_kv_cache"]
        else metadata.output_abi[:1]
    )
    return helper.make_model(
        helper.make_graph(
            [],
            "io",
            [_value_info(item) for item in metadata.input_abi],
            [_value_info(item) for item in outputs],
        )
    )


@pytest.mark.parametrize(
    ("save_kv_cache", "decode"),
    [(True, False), (False, False), (True, True), (False, True)],
)
def test_validate_io_abi_accepts_derived_contract(
    save_kv_cache: bool,
    decode: bool,
) -> None:
    metadata = _metadata(save_kv_cache=save_kv_cache, decode=decode)

    validate_io_abi(_model(metadata), metadata)


@pytest.mark.parametrize("field", ["name", "dtype", "shape"])
def test_validate_io_abi_rejects_public_output_mismatch(field: str) -> None:
    metadata = _metadata(save_kv_cache=True)
    model = _model(metadata)
    output = model.graph.output[1]
    if field == "name":
        output.name = "present.1.key"
    elif field == "dtype":
        output.type.tensor_type.elem_type = TensorProto.INT8
    else:
        output.type.tensor_type.shape.dim[2].dim_value = 5

    with pytest.raises(OnnxExportError, match="artifact ABI"):
        validate_io_abi(model, metadata)


def test_validate_io_abi_accepts_explicit_lowered_dtype_override() -> None:
    metadata = _metadata(save_kv_cache=True)
    model = _model(metadata)
    model.graph.output[1].type.tensor_type.elem_type = TensorProto.INT8

    validate_io_abi(
        model,
        metadata,
        output_dtype_overrides={"present.0.key": "int8"},
    )
