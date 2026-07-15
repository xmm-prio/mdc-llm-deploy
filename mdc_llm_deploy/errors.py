"""Public exception hierarchy for MDC LLM Deploy."""


class MdcDeployError(Exception):
    """Base class for all library-defined errors."""


class GraphStateError(MdcDeployError):
    """Raised when an operation is invalid for the current graph stage."""


class UnsupportedPatternError(MdcDeployError):
    """Raised when a required model or graph pattern cannot be represented."""


class QuantizationConfigError(MdcDeployError, ValueError):
    """Raised when a quantization configuration is invalid."""


class OnnxExportError(MdcDeployError):
    """Raised when MDC ONNX lowering or validation fails."""
