"""Contracts for incrementally registered inference-only custom operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class CustomOp(ABC):
    """Define one custom operator without coupling it to registry orchestration."""

    qualified_name: ClassVar[str]
    schema: ClassVar[str]
    onnx_opset: ClassVar[int] = 18

    @staticmethod
    @abstractmethod
    def cpu(*args: Any, **kwargs: Any) -> Any:
        """Execute the CPU kernel."""

    @staticmethod
    @abstractmethod
    def cuda(*args: Any, **kwargs: Any) -> Any:
        """Execute the CUDA kernel."""

    @staticmethod
    @abstractmethod
    def fake(*args: Any, **kwargs: Any) -> Any:
        """Infer output metadata for FakeTensor and MetaTensor execution."""

    @classmethod
    def meta(cls, *args: Any, **kwargs: Any) -> Any:
        """Delegate MetaTensor execution to the shared abstract implementation."""
        return cls.fake(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def onnx(*args: Any, **kwargs: Any) -> Any:
        """Build the ONNX representation for this operator."""
