from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from types import MappingProxyType
from typing import Any

import onnx
import pytest
import torch
from onnx.defs import OpSchema
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from mdc_llm_deploy.custom_ops import (
    OnnxOperatorSpec,
    OperatorPlugin,
    TorchOperatorSpec,
    create_onnx_export_profile,
    get_operator,
    register_operator,
    registered_operators,
)
from mdc_llm_deploy.onnx.schemas import (
    ALL_SCHEMA_NAMES,
    CANN_FIA_SOURCE_COMMIT,
    CANN_FIA_SOURCE_URL,
    FUSED_INFER_ATTENTION_SCORE_OP,
    create_fused_infer_attention_score_schema,
)


def _identity_cpu(input: torch.Tensor) -> torch.Tensor:
    return input.clone()


def _identity_cuda(input: torch.Tensor) -> torch.Tensor:
    return input.clone()


def _identity_fake(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)


def _scale_cpu(input: torch.Tensor) -> torch.Tensor:
    return input * 2


def _scale_cuda(input: torch.Tensor) -> torch.Tensor:
    return input * 2


def _scale_fake(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)


def _translation(*args: Any, **kwargs: Any) -> Any:
    return args, kwargs


def _schema(name: str, *, output_count: int = 1) -> OpSchema:
    parameter = OpSchema.FormalParameter
    return OpSchema(
        name,
        "",
        18,
        inputs=[parameter("input", "T")],
        outputs=[parameter(f"output_{index}", "T") for index in range(output_count)],
        type_constraints=[("T", ["tensor(float)"], "Supported tensor types.")],
    )


def _plugin(
    name: str,
    *,
    qualified_name: str,
    onnx_name: str,
    cpu: Any = _identity_cpu,
    cuda: Any = _identity_cuda,
    fake: Any = _identity_fake,
    schema: OpSchema | None = None,
) -> OperatorPlugin:
    return OperatorPlugin(
        name=name,
        torch=TorchOperatorSpec(
            qualified_name=qualified_name,
            schema="(Tensor input) -> Tensor",
            cpu_kernel=cpu,
            cuda_kernel=cuda,
            fake_kernel=fake,
        ),
        onnx=OnnxOperatorSpec(schema or _schema(onnx_name), _translation),
    )


def _remove_schema(name: str) -> None:
    with suppress(onnx.defs.SchemaError):
        onnx.defs.deregister_schema(name, 18, "")


def test_registration_is_torch_only_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _plugin(
        "registry_identity",
        qualified_name="mdc_registry_test::identity",
        onnx_name="MdcRegistryIdentity",
    )
    monkeypatch.setattr(
        torch.onnx,
        "register_custom_op_symbolic",
        lambda *_args, **_kwargs: pytest.fail("legacy symbolic registration called"),
    )
    assert _get_schema(plugin.onnx.name) is None

    first = register_operator(plugin)
    second = register_operator(plugin)

    assert first is second
    assert get_operator(plugin.name) is first
    assert first in registered_operators()
    assert _get_schema(plugin.onnx.name) is None
    assert torch.equal(
        first.torch.definition(torch.tensor([1.0])),
        torch.tensor([1.0]),
    )


def test_fake_interface_covers_fake_and_meta_tensors() -> None:
    plugin = _plugin(
        "registry_scale",
        qualified_name="mdc_registry_test::scale",
        onnx_name="MdcRegistryScale",
        cpu=_scale_cpu,
        cuda=_scale_cuda,
        fake=_scale_fake,
    )
    entry = register_operator(plugin).torch
    meta_output = entry.definition(torch.empty(2, 3, device="meta"))

    with FakeTensorMode() as mode:
        fake_input = mode.from_tensor(torch.empty(4, 5))
        fake_output = entry.definition(fake_input)

    assert meta_output.device.type == "meta"
    assert meta_output.shape == (2, 3)
    assert isinstance(fake_output, FakeTensor)
    assert fake_output.shape == (4, 5)


def test_autograd_is_explicitly_rejected() -> None:
    plugin = _plugin(
        "registry_autograd",
        qualified_name="mdc_registry_test::autograd",
        onnx_name="MdcRegistryAutograd",
    )
    entry = register_operator(plugin).torch
    output = entry.definition(torch.ones(2, requires_grad=True))

    with pytest.raises(RuntimeError, match="inference-only"):
        output.sum().backward()


def test_profile_registers_only_selected_schema_and_is_read_only() -> None:
    selected = _plugin(
        "registry_selected",
        qualified_name="mdc_registry_test::selected",
        onnx_name="MdcRegistrySelected",
    )
    unselected = _plugin(
        "registry_unselected",
        qualified_name="mdc_registry_test::unselected",
        onnx_name="MdcRegistryUnselected",
    )
    try:
        register_operator(selected)
        register_operator(unselected)

        profile = create_onnx_export_profile(selected.name)

        assert _get_schema(selected.onnx.name) is not None
        assert _get_schema(unselected.onnx.name) is None
        assert isinstance(profile.custom_translation_table, MappingProxyType)
        assert profile.operators == {selected.name: selected.onnx}
        dispatch_target = get_operator(selected.name).torch.dispatch_target
        assert profile.custom_translation_table == {
            dispatch_target: selected.onnx.translation
        }
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [onnx.helper.make_node(selected.onnx.name, ["input"], ["output"])],
                "local-schema-check",
                [
                    onnx.helper.make_tensor_value_info(
                        "input",
                        onnx.TensorProto.FLOAT,
                        [2],
                    )
                ],
                [
                    onnx.helper.make_tensor_value_info(
                        "output",
                        onnx.TensorProto.FLOAT,
                        [2],
                    )
                ],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model, full_check=True)
        with pytest.raises(TypeError):
            profile.operators[selected.name] = selected.onnx  # type: ignore[index]
    finally:
        _remove_schema(selected.onnx.name)
        _remove_schema(unselected.onnx.name)


def test_conflicting_plugin_contract_fails_before_onnx_registration() -> None:
    plugin = _plugin(
        "registry_conflict",
        qualified_name="mdc_registry_test::conflict",
        onnx_name="MdcRegistryConflict",
    )
    register_operator(plugin)
    conflicting = _plugin(
        plugin.name,
        qualified_name=plugin.torch.qualified_name,
        onnx_name=plugin.onnx.name,
        cpu=_scale_cpu,
        cuda=_scale_cuda,
        fake=_scale_fake,
    )

    with pytest.raises(ValueError, match="conflicting contract"):
        register_operator(conflicting)
    assert _get_schema(plugin.onnx.name) is None


def test_profile_registration_is_thread_safe_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _plugin(
        "registry_concurrent",
        qualified_name="mdc_registry_test::concurrent",
        onnx_name="MdcRegistryConcurrent",
    )
    register_operator(plugin)
    calls = 0
    real_register = onnx.defs.register_schema

    def counted_register(schema: OpSchema) -> None:
        nonlocal calls
        calls += 1
        real_register(schema)

    monkeypatch.setattr(onnx.defs, "register_schema", counted_register)
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            profiles = tuple(
                executor.map(
                    lambda _index: create_onnx_export_profile(plugin.name),
                    range(32),
                )
            )

        assert calls == 1
        assert all(tuple(profile.operators) == (plugin.name,) for profile in profiles)
    finally:
        _remove_schema(plugin.onnx.name)


def test_existing_different_onnx_schema_is_rejected() -> None:
    onnx_name = "MdcRegistrySchemaConflict"
    onnx.defs.register_schema(_schema(onnx_name, output_count=2))
    plugin = _plugin(
        "registry_schema_conflict",
        qualified_name="mdc_registry_test::schema_conflict",
        onnx_name=onnx_name,
    )
    try:
        register_operator(plugin)
        with pytest.raises(ValueError, match="different contract"):
            create_onnx_export_profile(plugin.name)
    finally:
        _remove_schema(onnx_name)


def test_profile_preflights_all_schemas_before_registering() -> None:
    script = """
import onnx
import torch
from onnx.defs import OpSchema
from mdc_llm_deploy.custom_ops import (
    OnnxOperatorSpec, OperatorPlugin, TorchOperatorSpec,
    create_onnx_export_profile, register_operator,
)

def identity(input: torch.Tensor) -> torch.Tensor:
    return input.clone()

def translate(*args, **kwargs):
    return args, kwargs

def schema(name, output_count=1):
    parameter = OpSchema.FormalParameter
    return OpSchema(
        name, "", 18,
        inputs=[parameter("input", "T")],
        outputs=[parameter(f"output_{index}", "T") for index in range(output_count)],
        type_constraints=[("T", ["tensor(float)"], "type")],
    )

def plugin(name, onnx_name):
    return OperatorPlugin(
        name,
        TorchOperatorSpec(
            f"mdc_registry_preflight::{name}",
            "(Tensor input) -> Tensor",
            identity, identity, identity,
        ),
        OnnxOperatorSpec(schema(onnx_name), translate),
    )

first = plugin("first", "MdcRegistryPreflightFirst")
second = plugin("second", "MdcRegistryPreflightSecond")
register_operator(first)
register_operator(second)
onnx.defs.register_schema(schema(second.onnx.name, output_count=2))
try:
    create_onnx_export_profile(first.name, second.name)
except ValueError as error:
    assert str(error) == (
        "ONNX schema 'MdcRegistryPreflightSecond' opset 18 is already "
        "registered with a different contract"
    )
else:
    raise AssertionError("profile accepted a conflicting later schema")

try:
    onnx.defs.get_schema(first.onnx.name, 18, "")
except onnx.defs.SchemaError:
    pass
else:
    raise AssertionError("first schema was written before preflight completed")
assert len(onnx.defs.get_schema(second.onnx.name, 18, "").outputs) == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_profile_rejects_conflicting_duplicate_batch_key_before_write() -> None:
    script = """
import onnx
import torch
from onnx.defs import OpSchema
from mdc_llm_deploy.custom_ops import (
    OnnxOperatorSpec, OperatorPlugin, TorchOperatorSpec,
    create_onnx_export_profile, register_operator,
)

def identity(input: torch.Tensor) -> torch.Tensor:
    return input.clone()

def translate(*args, **kwargs):
    return args, kwargs

def schema(output_count):
    parameter = OpSchema.FormalParameter
    return OpSchema(
        "MdcRegistryDuplicateConflict", "", 18,
        inputs=[parameter("input", "T")],
        outputs=[parameter(f"output_{index}", "T") for index in range(output_count)],
        type_constraints=[("T", ["tensor(float)"], "type")],
    )

for name, output_count in (("first", 1), ("second", 2)):
    register_operator(OperatorPlugin(
        name,
        TorchOperatorSpec(
            f"mdc_registry_duplicate::{name}",
            "(Tensor input) -> Tensor",
            identity, identity, identity,
        ),
        OnnxOperatorSpec(schema(output_count), translate),
    ))

try:
    create_onnx_export_profile("first", "second")
except ValueError as error:
    assert str(error) == (
        "ONNX schema 'MdcRegistryDuplicateConflict' opset 18 is already "
        "registered with a different contract"
    )
else:
    raise AssertionError("profile accepted conflicting schemas in one batch")

try:
    onnx.defs.get_schema("MdcRegistryDuplicateConflict", 18, "")
except onnx.defs.SchemaError:
    pass
else:
    raise AssertionError("conflicting batch schema was registered")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_profile_deduplicates_equivalent_schema_writes_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    onnx_name = "MdcRegistryDuplicateEquivalent"
    first = _plugin(
        "registry_duplicate_equivalent_first",
        qualified_name="mdc_registry_test::duplicate_equivalent_first",
        onnx_name=onnx_name,
    )
    second = _plugin(
        "registry_duplicate_equivalent_second",
        qualified_name="mdc_registry_test::duplicate_equivalent_second",
        onnx_name=onnx_name,
    )
    register_operator(first)
    register_operator(second)
    calls = 0
    real_register = onnx.defs.register_schema

    def counted_register(schema: OpSchema) -> None:
        nonlocal calls
        calls += 1
        real_register(schema)

    monkeypatch.setattr(onnx.defs, "register_schema", counted_register)
    try:
        profile = create_onnx_export_profile(first.name, second.name)

        assert calls == 1
        assert tuple(profile.operators) == (first.name, second.name)
        assert profile.operators == {
            first.name: first.onnx,
            second.name: second.onnx,
        }
    finally:
        _remove_schema(onnx_name)


def test_profile_keeps_registered_prefix_when_later_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _plugin(
        "registry_write_failure_first",
        qualified_name="mdc_registry_test::write_failure_first",
        onnx_name="MdcRegistryWriteFailureFirst",
    )
    second = _plugin(
        "registry_write_failure_second",
        qualified_name="mdc_registry_test::write_failure_second",
        onnx_name="MdcRegistryWriteFailureSecond",
    )
    register_operator(first)
    register_operator(second)
    calls = 0
    real_register = onnx.defs.register_schema

    def fail_second_write(schema: OpSchema) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("third-party registry write failed")
        real_register(schema)

    monkeypatch.setattr(onnx.defs, "register_schema", fail_second_write)
    try:
        with pytest.raises(RuntimeError, match="third-party registry write failed"):
            create_onnx_export_profile(first.name, second.name)

        assert _get_schema(first.onnx.name) is not None
        assert _get_schema(second.onnx.name) is None
    finally:
        _remove_schema(first.onnx.name)
        _remove_schema(second.onnx.name)


def test_equivalent_onnx_schema_with_different_docs_is_accepted() -> None:
    script = """
import onnx
import torch
from onnx.defs import OpSchema
from mdc_llm_deploy.custom_ops import (
    OnnxOperatorSpec, OperatorPlugin, TorchOperatorSpec,
    create_onnx_export_profile, get_operator, register_operator,
)

def identity(input: torch.Tensor) -> torch.Tensor:
    return input.clone()

def translate(*args, **kwargs):
    return args, kwargs

parameter = OpSchema.FormalParameter
onnx.defs.register_schema(OpSchema(
    "MdcRegistryDocumentEquivalent", "", 18, doc="External documentation.",
    inputs=[parameter("input", "T")],
    outputs=[parameter("output", "T")],
    type_constraints=[("T", ["tensor(float)"], "External type description.")],
))
schema = OpSchema(
    "MdcRegistryDocumentEquivalent", "", 18, doc="Project documentation.",
    inputs=[parameter("input", "T")],
    outputs=[parameter("output", "T")],
    type_constraints=[("T", ["tensor(float)"], "Project type description.")],
)
plugin = OperatorPlugin(
    "document_equivalent",
    TorchOperatorSpec(
        "mdc_registry_subprocess::document_equivalent",
        "(Tensor input) -> Tensor",
        identity, identity, identity,
    ),
    OnnxOperatorSpec(schema, translate),
)
register_operator(plugin)
profile = create_onnx_export_profile(plugin.name)
dispatch_target = get_operator(plugin.name).torch.dispatch_target
assert profile.operators == {plugin.name: plugin.onnx}
assert profile.custom_translation_table == {dispatch_target: translate}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_type_constraint_allowed_types_use_set_semantics() -> None:
    script = """
import onnx
import torch
from onnx.defs import OpSchema
from mdc_llm_deploy.custom_ops import (
    OnnxOperatorSpec, OperatorPlugin, TorchOperatorSpec,
    create_onnx_export_profile, register_operator,
)

def identity(input: torch.Tensor) -> torch.Tensor:
    return input.clone()

def translate(*args, **kwargs):
    return args, kwargs

parameter = OpSchema.FormalParameter
onnx.defs.register_schema(OpSchema(
    "MdcRegistryConstraintSet", "", 18,
    inputs=[parameter("input", "T")],
    outputs=[parameter("output", "T")],
    type_constraints=[
        ("T", ["tensor(float16)", "tensor(float)", "tensor(float16)"], "type")
    ],
))
schema = OpSchema(
    "MdcRegistryConstraintSet", "", 18,
    inputs=[parameter("input", "T")],
    outputs=[parameter("output", "T")],
    type_constraints=[("T", ["tensor(float)", "tensor(float16)"], "type")],
)
plugin = OperatorPlugin(
    "constraint_set",
    TorchOperatorSpec(
        "mdc_registry_subprocess::constraint_set",
        "(Tensor input) -> Tensor",
        identity, identity, identity,
    ),
    OnnxOperatorSpec(schema, translate),
)
register_operator(plugin)
profile = create_onnx_export_profile(plugin.name)
assert profile.operators == {plugin.name: plugin.onnx}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    (
        "dimension",
        "registered_parameter",
        "expected_parameter",
        "registered_attribute",
        "expected_attribute",
    ),
    [
        (
            "parameter_type",
            'parameter("input", "T")',
            'parameter("input", "tensor(float)")',
            'attribute("axis", attr_type.INT, "axis", required=True)',
            'attribute("axis", attr_type.INT, "axis", required=True)',
        ),
        (
            "parameter_option",
            'parameter("input", "T")',
            'parameter("input", "T", param_option=option.Optional)',
            'attribute("axis", attr_type.INT, "axis", required=True)',
            'attribute("axis", attr_type.INT, "axis", required=True)',
        ),
        (
            "attribute_type",
            'parameter("input", "T")',
            'parameter("input", "T")',
            'attribute("axis", attr_type.INT, "axis", required=True)',
            'attribute("axis", attr_type.FLOAT, "axis", required=True)',
        ),
        (
            "attribute_required",
            'parameter("input", "T")',
            'parameter("input", "T")',
            'attribute("axis", attr_type.INT, "axis", required=True)',
            'attribute("axis", attr_type.INT, "axis", required=False)',
        ),
    ],
)
def test_onnx_abi_dimension_conflicts_are_rejected(
    dimension: str,
    registered_parameter: str,
    expected_parameter: str,
    registered_attribute: str,
    expected_attribute: str,
) -> None:
    script = f"""
import onnx
import torch
from onnx.defs import OpSchema
from mdc_llm_deploy.custom_ops import (
    OnnxOperatorSpec, OperatorPlugin, TorchOperatorSpec,
    create_onnx_export_profile, register_operator,
)

def identity(input: torch.Tensor) -> torch.Tensor:
    return input.clone()

def translate(*args, **kwargs):
    return args, kwargs

parameter = OpSchema.FormalParameter
option = OpSchema.FormalParameterOption
attribute = OpSchema.Attribute
attr_type = OpSchema.AttrType
onnx.defs.register_schema(OpSchema(
    "MdcRegistryAbi{dimension.title()}", "", 18,
    inputs=[{registered_parameter}],
    outputs=[parameter("output", "T")],
    type_constraints=[("T", ["tensor(float)"], "type")],
    attributes=[{registered_attribute}],
))
schema = OpSchema(
    "MdcRegistryAbi{dimension.title()}", "", 18,
    inputs=[{expected_parameter}],
    outputs=[parameter("output", "T")],
    type_constraints=[("T", ["tensor(float)"], "type")],
    attributes=[{expected_attribute}],
)
plugin = OperatorPlugin(
    "abi_{dimension}",
    TorchOperatorSpec(
        "mdc_registry_subprocess::abi_{dimension}",
        "(Tensor input) -> Tensor",
        identity, identity, identity,
    ),
    OnnxOperatorSpec(schema, translate),
)
register_operator(plugin)
try:
    create_onnx_export_profile(plugin.name)
except ValueError as error:
    assert "different contract" in str(error)
else:
    raise AssertionError("{dimension} conflict was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_onnx_attribute_default_conflict_is_rejected() -> None:
    script = """
import onnx
import torch
from onnx import helper
from onnx.defs import OpSchema
from mdc_llm_deploy.custom_ops import (
    OnnxOperatorSpec, OperatorPlugin, TorchOperatorSpec,
    create_onnx_export_profile, register_operator,
)

def identity(input: torch.Tensor) -> torch.Tensor:
    return input.clone()

def translate(*args, **kwargs):
    return args, kwargs

parameter = OpSchema.FormalParameter
attribute = OpSchema.Attribute
onnx.defs.register_schema(OpSchema(
    "MdcRegistryAttributeConflict", "", 18,
    inputs=[parameter("input", "T")],
    outputs=[parameter("output", "T")],
    type_constraints=[("T", ["tensor(float)"], "type")],
    attributes=[attribute("axis", helper.make_attribute("axis", 1), "external")],
))
schema = OpSchema(
    "MdcRegistryAttributeConflict", "", 18,
    inputs=[parameter("input", "T")],
    outputs=[parameter("output", "T")],
    type_constraints=[("T", ["tensor(float)"], "type")],
    attributes=[attribute("axis", helper.make_attribute("axis", 2), "project")],
)
plugin = OperatorPlugin(
    "attribute_conflict",
    TorchOperatorSpec(
        "mdc_registry_subprocess::attribute_conflict",
        "(Tensor input) -> Tensor",
        identity, identity, identity,
    ),
    OnnxOperatorSpec(schema, translate),
)
register_operator(plugin)
try:
    create_onnx_export_profile(plugin.name)
except ValueError as error:
    assert "different contract" in str(error)
else:
    raise AssertionError("attribute default conflict was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_local_schema_is_not_available_in_a_new_process() -> None:
    plugin = _plugin(
        "registry_process_boundary",
        qualified_name="mdc_registry_test::process_boundary",
        onnx_name="MdcRegistryProcessBoundary",
    )
    try:
        register_operator(plugin)
        create_onnx_export_profile(plugin.name)
        script = (
            "import onnx\n"
            "try:\n"
            f"    onnx.defs.get_schema({plugin.onnx.name!r}, 18, '')\n"
            "except onnx.defs.SchemaError:\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(1)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
    finally:
        _remove_schema(plugin.onnx.name)


def test_package_import_does_not_load_operator_modules() -> None:
    script = (
        "import sys\n"
        "import mdc_llm_deploy.custom_ops\n"
        "modules = ('rms_norm', 'apply_rotary_pos_emb', "
        "'fused_infer_attention_score', 'moe_expert')\n"
        "loaded = [name for name in modules "
        "if f'mdc_llm_deploy.custom_ops.{name}' in sys.modules]\n"
        "raise SystemExit(bool(loaded))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_central_fia_schema_matches_frozen_cann_master_proto() -> None:
    schema = create_fused_infer_attention_score_schema()

    assert CANN_FIA_SOURCE_COMMIT == "606a5ddb67c67d93c137a7b474fa7a5edd05f7c9"
    assert CANN_FIA_SOURCE_COMMIT in CANN_FIA_SOURCE_URL
    assert schema.name == FUSED_INFER_ATTENTION_SCORE_OP
    assert schema.domain == ""
    assert schema.since_version == 18
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
    assert set(schema.attributes) == {
        "num_heads",
        "scale",
        "pre_tokens",
        "next_tokens",
        "input_layout",
        "num_key_value_heads",
        "sparse_mode",
        "inner_precise",
        "block_size",
        "antiquant_mode",
        "softmax_lse_flag",
        "key_antiquant_mode",
        "value_antiquant_mode",
        "query_quant_mode",
        "pse_type",
        "out_dtype",
    }
    assert schema.attributes["num_heads"].required
    assert onnx.helper.get_attribute_value(
        schema.attributes["inner_precise"].default_value
    ) == 1


def test_central_schema_import_is_lazy_and_selected_registration_checks_model() -> None:
    script = """
import onnx
from mdc_llm_deploy.onnx.schemas import (
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
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_central_registry_rejects_unknown_schema_name_without_writes() -> None:
    script = """
import onnx
from mdc_llm_deploy.onnx.schemas import ALL_SCHEMA_NAMES, register_schemas

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
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(ALL_SCHEMA_NAMES) == 6


def _get_schema(name: str) -> OpSchema | None:
    try:
        schema = onnx.defs.get_schema(name, 18, "")
    except onnx.defs.SchemaError:
        return None
    return schema if schema.since_version == 18 else None
