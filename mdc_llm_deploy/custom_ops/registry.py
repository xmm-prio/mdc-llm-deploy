"""Thread-safe Torch registration and local ONNX export-profile orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, NoReturn, Protocol, cast

import torch

from ..onnx.schemas import OnnxSchemaConflictError, register_schema_objects
from .base import OnnxOperatorSpec, OperatorPlugin, TorchOperatorSpec


class CustomOpDefinition(Protocol):
    """Describe Torch's custom-op definition methods used by this registry."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the custom operator."""

    def register_kernel(
        self, device_types: str, fn: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Attach one device kernel."""

    def register_fake(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Attach one FakeTensor implementation."""

    def register_autograd(self, backward: Callable[..., Any]) -> None:
        """Attach one autograd implementation."""


@dataclass(frozen=True, slots=True)
class RegisteredTorchOperator:
    """Hold one immutable Torch contract and its process-local registration."""

    spec: TorchOperatorSpec
    definition: CustomOpDefinition
    dispatch_target: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class RegisteredOperator:
    """Hold one plugin and its eagerly created Torch registration."""

    plugin: OperatorPlugin
    torch: RegisteredTorchOperator


@dataclass(frozen=True, slots=True)
class OnnxExportProfile:
    """Expose selected ONNX contracts and a read-only Dynamo translation table."""

    custom_translation_table: Mapping[Callable[..., Any], Callable[..., Any]]
    operators: Mapping[str, OnnxOperatorSpec]


class TorchOperatorRegistry:
    """Register broad Torch execution contracts without ONNX side effects."""

    def __init__(self) -> None:
        self._entries: dict[str, RegisteredTorchOperator] = {}
        self._lock = RLock()

    def register(self, spec: TorchOperatorSpec) -> RegisteredTorchOperator:
        """Register one Torch contract or return its existing registration."""
        with self._lock:
            existing = self._entries.get(spec.qualified_name)
            if existing is not None:
                if existing.spec != spec:
                    raise ValueError(
                        f"Torch operator {spec.qualified_name!r} has a conflicting contract"
                    )
                return existing

            definition = cast(
                CustomOpDefinition,
                torch.library.custom_op(
                    spec.qualified_name,
                    spec.cpu_kernel,
                    mutates_args=(),
                    device_types="cpu",
                    schema=spec.schema,
                ),
            )
            definition.register_kernel("cuda", spec.cuda_kernel)
            definition.register_fake(spec.fake_kernel)
            definition.register_autograd(
                _inference_only_backward(spec.qualified_name)
            )
            namespace, _, operator_name = spec.qualified_name.partition("::")
            dispatch_target = cast(
                Callable[..., Any],
                getattr(getattr(torch.ops, namespace), operator_name).default,
            )
            entry = RegisteredTorchOperator(spec, definition, dispatch_target)
            self._entries[spec.qualified_name] = entry
            return entry


class OnnxSchemaRegistry:
    """Install selected default-domain schemas with conflict detection."""

    def register_all(self, specs: Iterable[OnnxOperatorSpec]) -> None:
        """Preflight and register selected schemas as one project batch."""
        try:
            register_schema_objects(spec.schema for spec in specs)
        except OnnxSchemaConflictError as error:
            raise ValueError(
                f"ONNX schema {error.name!r} opset {error.since_version} is already "
                "registered with a different contract"
            ) from None


class OperatorRegistry:
    """Coordinate independently loaded plugins and lazily built ONNX profiles."""

    def __init__(self) -> None:
        self._operators: dict[str, RegisteredOperator] = {}
        self._torch = TorchOperatorRegistry()
        self._onnx = OnnxSchemaRegistry()
        self._lock = RLock()

    def register(self, plugin: OperatorPlugin) -> RegisteredOperator:
        """Register only the plugin's broad Torch contract."""
        with self._lock:
            existing = self._operators.get(plugin.name)
            if existing is not None:
                if existing.plugin != plugin:
                    raise ValueError(
                        f"Operator plugin {plugin.name!r} has a conflicting contract"
                    )
                return existing
            torch_entry = self._torch.register(plugin.torch)
            entry = RegisteredOperator(plugin, torch_entry)
            self._operators[plugin.name] = entry
            return entry

    def get(self, name: str) -> RegisteredOperator:
        """Return one loaded plugin by its public name."""
        with self._lock:
            try:
                return self._operators[name]
            except KeyError:
                raise KeyError(f"Operator plugin {name!r} is not loaded") from None

    def entries(self) -> tuple[RegisteredOperator, ...]:
        """Return an immutable snapshot of loaded plugins."""
        with self._lock:
            return tuple(self._operators.values())

    def create_profile(self, *operator_names: str) -> OnnxExportProfile:
        """Register selected ONNX schemas and build their Dynamo translations."""
        with self._lock:
            selected: dict[str, RegisteredOperator] = {}
            for name in operator_names:
                if name not in selected:
                    selected[name] = self.get(name)

            self._onnx.register_all(entry.plugin.onnx for entry in selected.values())

            translations: dict[Callable[..., Any], Callable[..., Any]] = {}
            contracts: dict[str, OnnxOperatorSpec] = {}
            for name, entry in selected.items():
                translations[entry.torch.dispatch_target] = entry.plugin.onnx.translation
                contracts[name] = entry.plugin.onnx

            return OnnxExportProfile(
                custom_translation_table=MappingProxyType(translations),
                operators=MappingProxyType(contracts),
            )


def _inference_only_backward(qualified_name: str) -> Callable[..., NoReturn]:
    def backward(_context: object, *_grad_outputs: object) -> NoReturn:
        raise RuntimeError(f"Custom operator {qualified_name!r} is inference-only")

    return backward


_REGISTRY = OperatorRegistry()


def register_operator(plugin: OperatorPlugin) -> RegisteredOperator:
    """Load one plugin's Torch contract into the process-wide registry."""
    return _REGISTRY.register(plugin)


def get_operator(name: str) -> RegisteredOperator:
    """Return one process-wide loaded operator plugin."""
    return _REGISTRY.get(name)


def registered_operators() -> tuple[RegisteredOperator, ...]:
    """Return an immutable snapshot of process-wide loaded plugins."""
    return _REGISTRY.entries()


def create_onnx_export_profile(*operator_names: str) -> OnnxExportProfile:
    """Create an ONNX profile for already loaded operator plugins."""
    return _REGISTRY.create_profile(*operator_names)
