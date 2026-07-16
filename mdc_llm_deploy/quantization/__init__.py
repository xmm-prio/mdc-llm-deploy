"""PTQ planning, calibration, and fake quantization."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .config import QuantizationConfig

if TYPE_CHECKING:
    from .algorithms import (
        calculate_qparams,
        decode_dequant_scale,
        encode_dequant_scale,
        gptq_weight_quantize,
        quantize,
    )
    from .api import oneshot
    from .planning import (
        QuantizedTensor,
        TargetPlan,
        effective_selector,
        integer_range,
        pattern_matches,
        plan_quantization,
        selected,
    )

__all__ = [
    "QuantizationConfig",
    "QuantizedTensor",
    "TargetPlan",
    "calculate_qparams",
    "decode_dequant_scale",
    "effective_selector",
    "encode_dequant_scale",
    "gptq_weight_quantize",
    "integer_range",
    "oneshot",
    "pattern_matches",
    "plan_quantization",
    "quantize",
    "selected",
]

_LAZY_EXPORTS = {
    "calculate_qparams": ("mdc_llm_deploy.quantization.algorithms", "calculate_qparams"),
    "decode_dequant_scale": (
        "mdc_llm_deploy.quantization.algorithms",
        "decode_dequant_scale",
    ),
    "effective_selector": (
        "mdc_llm_deploy.quantization.planning",
        "effective_selector",
    ),
    "encode_dequant_scale": (
        "mdc_llm_deploy.quantization.algorithms",
        "encode_dequant_scale",
    ),
    "gptq_weight_quantize": (
        "mdc_llm_deploy.quantization.algorithms",
        "gptq_weight_quantize",
    ),
    "integer_range": ("mdc_llm_deploy.quantization.planning", "integer_range"),
    "oneshot": ("mdc_llm_deploy.quantization.api", "oneshot"),
    "pattern_matches": ("mdc_llm_deploy.quantization.planning", "pattern_matches"),
    "plan_quantization": (
        "mdc_llm_deploy.quantization.planning",
        "plan_quantization",
    ),
    "quantize": ("mdc_llm_deploy.quantization.algorithms", "quantize"),
    "QuantizedTensor": (
        "mdc_llm_deploy.quantization.planning",
        "QuantizedTensor",
    ),
    "selected": ("mdc_llm_deploy.quantization.planning", "selected"),
    "TargetPlan": ("mdc_llm_deploy.quantization.planning", "TargetPlan"),
}


def __getattr__(name: str) -> Any:
    """Load Torch-dependent quantization APIs on demand."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
