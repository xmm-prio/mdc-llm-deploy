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


def _get_schema(name: str) -> OpSchema | None:
    try:
        schema = onnx.defs.get_schema(name, 18, "")
    except onnx.defs.SchemaError:
        return None
    return schema if schema.since_version == 18 else None
