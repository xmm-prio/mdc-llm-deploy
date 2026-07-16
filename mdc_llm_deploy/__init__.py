"""Public API for MDC LLM Deploy."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

from .config import QuantizationConfig
from .errors import (
    GraphStateError,
    MdcDeployError,
    OnnxExportError,
    QuantizationConfigError,
    UnsupportedPatternError,
)

if TYPE_CHECKING:
    from .export import convert_to_decode, export
    from .onnx_export.api import onnx_export
    from .quantization import oneshot

__version__ = "0.1.0"

__all__ = [
    "GraphStateError",
    "MdcDeployError",
    "OnnxExportError",
    "QuantizationConfig",
    "QuantizationConfigError",
    "UnsupportedPatternError",
    "__version__",
    "convert_to_decode",
    "export",
    "oneshot",
    "onnx_export",
]

_LAZY_EXPORTS = {
    "convert_to_decode": ("mdc_llm_deploy.export", "convert_to_decode"),
    "export": ("mdc_llm_deploy.export", "export"),
    "oneshot": ("mdc_llm_deploy.quantization", "oneshot"),
    "onnx_export": ("mdc_llm_deploy.onnx_export.api", "onnx_export"),
}


def _load_lazy_export(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    """Load torch-dependent public APIs on first access."""
    return _load_lazy_export(name)


class _PublicApiModule(ModuleType):
    """Preserve lazy callables that share names with subpackages."""

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if (
            name in _LAZY_EXPORTS
            and isinstance(value, ModuleType)
        ):
            return _load_lazy_export(name)
        return value


def __dir__() -> list[str]:
    """Return public and initialized module names."""
    return sorted(set(globals()) | set(__all__))


sys.modules[__name__].__class__ = _PublicApiModule
