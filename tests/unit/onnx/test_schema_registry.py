from __future__ import annotations

import subprocess
import sys

import onnx
import pytest
from onnx.defs import OpSchema

from mdc_llm_deploy.onnx.schema import (
    ALL_SCHEMA_NAMES,
    ASCEND_DEQUANT_OP,
    ASCEND_QUANT_OP,
    CANN_FIA_SOURCE_COMMIT,
    CANN_FIA_SOURCE_URL,
    FUSED_INFER_ATTENTION_SCORE_OP,
    RMS_NORM_OP,
    ROTARY_POSITION_EMBEDDING_OP,
    create_fused_infer_attention_score_schema,
    create_rms_norm_schema,
    create_rotary_position_embedding_schema,
)


def _run_script(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_registry_contains_only_pipeline_schemas() -> None:
    assert ALL_SCHEMA_NAMES == (
        ASCEND_QUANT_OP,
        ASCEND_DEQUANT_OP,
        RMS_NORM_OP,
        ROTARY_POSITION_EMBEDDING_OP,
        FUSED_INFER_ATTENTION_SCORE_OP,
    )


def test_fusion_schema_factories_match_public_operator_names() -> None:
    schemas = (
        create_rms_norm_schema(),
        create_rotary_position_embedding_schema(),
        create_fused_infer_attention_score_schema(),
    )

    assert tuple(schema.name for schema in schemas) == (
        RMS_NORM_OP,
        ROTARY_POSITION_EMBEDDING_OP,
        FUSED_INFER_ATTENTION_SCORE_OP,
    )
    assert all(schema.domain == "" for schema in schemas)
    assert all(schema.since_version == 18 for schema in schemas)


def test_central_fia_schema_matches_frozen_cann_master_proto() -> None:
    schema = create_fused_infer_attention_score_schema()

    assert CANN_FIA_SOURCE_COMMIT == "606a5ddb67c67d93c137a7b474fa7a5edd05f7c9"
    assert CANN_FIA_SOURCE_COMMIT in CANN_FIA_SOURCE_URL
    assert schema.name == FUSED_INFER_ATTENTION_SCORE_OP
    assert [parameter.name for parameter in schema.inputs] == [
        "query",
        "key",
        "value",
        "pse_shift",
        "atten_mask",
        "actual_seq_lengths",
        "actual_seq_lengths_kv",
        "dequant_scale1",
        "quant_scale1",
        "dequant_scale2",
        "quant_scale2",
        "quant_offset2",
        "antiquant_scale",
        "antiquant_offset",
        "block_table",
        "query_padding_size",
        "kv_padding_size",
        "key_antiquant_scale",
        "key_antiquant_offset",
        "value_antiquant_scale",
        "value_antiquant_offset",
        "key_shared_prefix",
        "value_shared_prefix",
        "actual_shared_prefix_len",
        "query_rope",
        "key_rope",
        "key_rope_antiquant_scale",
        "dequant_scale_query",
        "learnable_sink",
        "q_start_idx",
        "kv_start_idx",
    ]
    assert all(
        parameter.option == OpSchema.FormalParameterOption.Optional
        for parameter in schema.inputs[3:]
    )
    assert [parameter.name for parameter in schema.outputs] == [
        "attention_out",
        "softmax_lse",
    ]
    assert schema.attributes["num_heads"].required


def test_schema_import_is_lazy_and_selected_registration_checks_model() -> None:
    _run_script(
        """
import onnx
from mdc_llm_deploy.onnx.schema import (
    ALL_SCHEMA_NAMES, RMS_NORM_OP, register_schemas,
)

for name in ALL_SCHEMA_NAMES:
    try:
        onnx.defs.get_schema(name, 18, "")
    except onnx.defs.SchemaError:
        pass
    else:
        raise AssertionError(f"schema registered during import: {name}")

register_schemas(RMS_NORM_OP)
assert onnx.defs.get_schema(RMS_NORM_OP, 18, "").since_version == 18
for name in ALL_SCHEMA_NAMES:
    if name == RMS_NORM_OP:
        continue
    try:
        onnx.defs.get_schema(name, 18, "")
    except onnx.defs.SchemaError:
        pass
    else:
        raise AssertionError(f"unselected schema registered: {name}")

model = onnx.helper.make_model(
    onnx.helper.make_graph(
        [onnx.helper.make_node(
            RMS_NORM_OP, ["x", "gamma"], ["y", "rstd"], epsilon=1e-6
        )],
        "cold-process-schema-check",
        [
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [1, 8]),
            onnx.helper.make_tensor_value_info("gamma", onnx.TensorProto.FLOAT16, [8]),
        ],
        [
            onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [1, 8]),
            onnx.helper.make_tensor_value_info("rstd", onnx.TensorProto.FLOAT, [1]),
        ],
    ),
    opset_imports=[onnx.helper.make_opsetid("", 18)],
)
onnx.checker.check_model(model, full_check=True)
"""
    )


def test_unknown_schema_name_is_rejected_before_registry_writes() -> None:
    _run_script(
        """
import onnx
from mdc_llm_deploy.onnx.schema import ALL_SCHEMA_NAMES, register_schemas

try:
    register_schemas("MissingMdcSchema")
except KeyError as error:
    assert "MissingMdcSchema" in str(error)
else:
    raise AssertionError("unknown schema name was accepted")

for name in ALL_SCHEMA_NAMES:
    try:
        onnx.defs.get_schema(name, 18, "")
    except onnx.defs.SchemaError:
        pass
    else:
        raise AssertionError(f"schema written after selection failure: {name}")
"""
    )


def test_conflicting_existing_schema_is_rejected() -> None:
    _run_script(
        """
import onnx
from onnx.defs import OpSchema
from mdc_llm_deploy.onnx.schema import OnnxSchemaConflictError, RMS_NORM_OP, register_schemas

parameter = OpSchema.FormalParameter
onnx.defs.register_schema(OpSchema(
    RMS_NORM_OP,
    "",
    18,
    inputs=[parameter("x", "T")],
    outputs=[parameter("y", "T")],
    type_constraints=[("T", ["tensor(float)"], "type")],
))
try:
    register_schemas(RMS_NORM_OP)
except OnnxSchemaConflictError as error:
    assert error.name == RMS_NORM_OP
else:
    raise AssertionError("conflicting schema was accepted")
"""
    )


def test_conflicting_duplicate_batch_is_rejected_before_write() -> None:
    from mdc_llm_deploy.onnx.schema import (
        OnnxSchemaConflictError,
        register_schema_objects,
    )

    parameter = OpSchema.FormalParameter

    def schema(output_count: int) -> OpSchema:
        return OpSchema(
            "MdcRegistryDuplicateConflict",
            "",
            18,
            inputs=[parameter("input", "T")],
            outputs=[
                parameter(f"output_{index}", "T") for index in range(output_count)
            ],
            type_constraints=[("T", ["tensor(float)"], "type")],
        )

    with pytest.raises(OnnxSchemaConflictError):
        register_schema_objects((schema(1), schema(2)))

    with pytest.raises(onnx.defs.SchemaError):
        onnx.defs.get_schema("MdcRegistryDuplicateConflict", 18, "")
