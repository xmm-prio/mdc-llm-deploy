"""Public API for MDC LLM Deploy."""

from __future__ import annotations

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


def __getattr__(name: str) -> Any:
    """Load torch-dependent public APIs on first access."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return public and initialized module names."""
    return sorted(set(globals()) | set(__all__))
