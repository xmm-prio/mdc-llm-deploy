"""Generic registration orchestration for inference-only custom operators."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, NoReturn, Protocol, cast

import torch

from .base import CustomOp


class CustomOpDefinition(Protocol):
    """Describe the public registration surface returned by custom_op."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the custom operator."""

    def register_kernel(
        self, device_types: str, fn: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Attach one device kernel."""

    def register_fake(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Attach one FakeTensor/MetaTensor implementation."""

    def register_autograd(self, backward: Callable[..., Any]) -> None:
        """Attach one autograd implementation."""


@dataclass(frozen=True, slots=True)
class RegisteredCustomOp:
    """Hold the class contract and its PyTorch custom-op definition."""

    operator_type: type[CustomOp]
    definition: CustomOpDefinition


class CustomOpRegistry:
    """Register independent custom-op classes through one generic pipeline."""

    def __init__(self) -> None:
        self._entries: dict[str, RegisteredCustomOp] = {}
        self._lock = RLock()

    def register(self, operator_type: type[CustomOp]) -> RegisteredCustomOp:
        """Register one operator, returning the existing entry when repeated."""
        self._validate(operator_type)
        name = operator_type.qualified_name

        with self._lock:
            existing = self._entries.get(name)
            if existing is not None:
                if existing.operator_type is not operator_type:
                    raise ValueError(f"Custom operator {name!r} has a different registered class")
                return existing

            definition = cast(
                CustomOpDefinition,
                torch.library.custom_op(
                    name,
                    operator_type.cpu,
                    mutates_args=(),
                    device_types="cpu",
                    schema=operator_type.schema,
                ),
            )
            definition.register_kernel("cuda", operator_type.cuda)
            definition.register_fake(operator_type.meta)
            definition.register_autograd(self._inference_only_backward(name))
            register_onnx_symbolic = cast(
                Callable[[str, Callable[..., Any], int], None],
                torch.onnx.register_custom_op_symbolic,  # type: ignore[attr-defined]
            )
            register_onnx_symbolic(
                name,
                operator_type.onnx,
                operator_type.onnx_opset,
            )

            entry = RegisteredCustomOp(operator_type=operator_type, definition=definition)
            self._entries[name] = entry
            return entry

    def register_many(
        self, *operator_types: type[CustomOp]
    ) -> tuple[RegisteredCustomOp, ...]:
        """Register operator classes in the supplied order."""
        return tuple(self.register(operator_type) for operator_type in operator_types)

    def get(self, qualified_name: str) -> RegisteredCustomOp:
        """Return one registered entry by qualified name."""
        with self._lock:
            try:
                return self._entries[qualified_name]
            except KeyError:
                raise KeyError(f"Custom operator {qualified_name!r} is not registered") from None

    def entries(self) -> tuple[RegisteredCustomOp, ...]:
        """Return an immutable snapshot of registered entries."""
        with self._lock:
            return tuple(self._entries.values())

    @staticmethod
    def _validate(operator_type: type[CustomOp]) -> None:
        if not inspect.isclass(operator_type) or not issubclass(operator_type, CustomOp):
            raise TypeError("operator_type must be a CustomOp subclass")
        if inspect.isabstract(operator_type):
            raise TypeError("operator_type must implement every CustomOp interface")

        name = operator_type.qualified_name
        namespace, separator, operator_name = name.partition("::")
        if separator != "::" or not namespace or not operator_name or "::" in operator_name:
            raise ValueError("qualified_name must use the 'namespace::operator' form")
        if not operator_type.schema:
            raise ValueError("schema must not be empty")
        if operator_type.onnx_opset <= 0:
            raise ValueError("onnx_opset must be positive")

    @staticmethod
    def _inference_only_backward(qualified_name: str) -> Callable[..., NoReturn]:
        def backward(_context: object, *_grad_outputs: object) -> NoReturn:
            raise RuntimeError(f"Custom operator {qualified_name!r} is inference-only")

        return backward


_REGISTRY = CustomOpRegistry()


def register_custom_op(operator_type: type[CustomOp]) -> RegisteredCustomOp:
    """Register one custom-op class in the process-wide registry."""
    return _REGISTRY.register(operator_type)


def register_custom_ops(
    *operator_types: type[CustomOp],
) -> tuple[RegisteredCustomOp, ...]:
    """Register custom-op classes incrementally in the process-wide registry."""
    return _REGISTRY.register_many(*operator_types)


def get_custom_op(qualified_name: str) -> RegisteredCustomOp:
    """Return one process-wide custom-op registration."""
    return _REGISTRY.get(qualified_name)


def registered_custom_ops() -> tuple[RegisteredCustomOp, ...]:
    """Return a snapshot of process-wide custom-op registrations."""
    return _REGISTRY.entries()
