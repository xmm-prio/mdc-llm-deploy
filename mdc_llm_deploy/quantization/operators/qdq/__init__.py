"""Standard ONNX QDQ representation."""

from ._registration import register_qdq_operator, warn_unvalidated_torch_version
from .functional import qdq

__all__ = ["qdq", "register_qdq_operator", "warn_unvalidated_torch_version"]
