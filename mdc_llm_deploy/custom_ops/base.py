"""Immutable contracts for independently loaded custom-operator plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from onnx.defs import OpSchema

Kernel = Callable[..., Any]
OnnxTranslation = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class TorchOperatorSpec:
    """Describe broad Torch execution without ONNX export restrictions."""

    qualified_name: str
    schema: str
    cpu_kernel: Kernel
    cuda_kernel: Kernel
    fake_kernel: Kernel

    def __post_init__(self) -> None:
        namespace, separator, operator_name = self.qualified_name.partition("::")
        if separator != "::" or not namespace or not operator_name or "::" in operator_name:
            raise ValueError("qualified_name must use the 'namespace::operator' form")
        if not self.schema:
            raise ValueError("schema must not be empty")


@dataclass(frozen=True, slots=True)
class OnnxOperatorSpec:
    """Describe one narrow default-domain ONNX direct-export contract."""

    schema: OpSchema
    translation: OnnxTranslation

    def __post_init__(self) -> None:
        if self.schema.domain:
            raise ValueError("ONNX operator schema must use the default domain")
        if self.schema.since_version != 18:
            raise ValueError("ONNX operator schema must target opset 18")

    @property
    def name(self) -> str:
        """Return the emitted ONNX operator name."""
        return self.schema.name

    @property
    def opset(self) -> int:
        """Return the emitted ONNX opset version."""
        return self.schema.since_version


@dataclass(frozen=True, slots=True)
class OperatorPlugin:
    """Bind broad Torch execution to one narrow ONNX export contract."""

    name: str
    torch: TorchOperatorSpec
    onnx: OnnxOperatorSpec

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("plugin name must be a non-empty normalized string")
        if "::" in self.name:
            raise ValueError("plugin name must not contain a namespace separator")


class OperatorPluginContract(Protocol):
    """Allow immutable plugin-compatible descriptions from operator packages."""

    name: str
    torch: TorchOperatorSpec
    onnx: OnnxOperatorSpec
