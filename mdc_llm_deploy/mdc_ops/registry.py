"""Torch Library registration and backend dispatch reporting."""
# mypy: disable-error-code="no-any-return,no-untyped-call"

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from ..operator_schema import (
    OPERATOR_SCHEMAS,
    TORCH_NAMESPACE,
    schema_for_torch_name,
)
from .attention import (
    fused_infer_attention_score_meta,
    fused_infer_attention_score_reference,
)
from .backend import (
    BackendImplementation,
    OperatorBackendStatus,
    backend_status_snapshot,
)
from .moe import moe_expert_meta, moe_expert_reference
from .normalization import (
    apply_rotary_pos_emb_meta,
    apply_rotary_pos_emb_reference,
    rms_norm_meta,
    rms_norm_reference,
)
from .quantized_io import (
    ascend_dequant_meta,
    ascend_dequant_reference,
    ascend_quant_v2_meta,
    ascend_quant_v2_reference,
)

Kernel = Callable[..., Tensor | tuple[Tensor, Tensor]]

_REFERENCE_KERNELS: Mapping[str, Kernel] = MappingProxyType({
    "rms_norm": rms_norm_reference,
    "apply_rotary_pos_emb": apply_rotary_pos_emb_reference,
    "fused_infer_attention_score": fused_infer_attention_score_reference,
    "ascend_quant_v2": ascend_quant_v2_reference,
    "ascend_dequant": ascend_dequant_reference,
    "moe_expert": moe_expert_reference,
})

_META_KERNELS: Mapping[str, Kernel] = MappingProxyType({
    "rms_norm": rms_norm_meta,
    "apply_rotary_pos_emb": apply_rotary_pos_emb_meta,
    "fused_infer_attention_score": fused_infer_attention_score_meta,
    "ascend_quant_v2": ascend_quant_v2_meta,
    "ascend_dequant": ascend_dequant_meta,
    "moe_expert": moe_expert_meta,
})
_SCHEMA_TORCH_NAMES = frozenset(
    schema.torch_name for schema in OPERATOR_SCHEMAS.values()
)
if (
    set(_REFERENCE_KERNELS) != _SCHEMA_TORCH_NAMES
    or set(_META_KERNELS) != _SCHEMA_TORCH_NAMES
):
    raise RuntimeError(
        "Every MDC operator schema requires reference and meta kernels"
    )


def _npu_is_available() -> bool:
    backend = getattr(torch, "npu", None)
    return bool(backend is not None and backend.is_available())


def _register_kernels(
    library: torch.library.Library,
    dispatch_key: str,
    kernels: Mapping[str, Kernel],
) -> None:
    for name, kernel in kernels.items():
        library.impl(name, kernel, dispatch_key)


_DEFINITION_LIBRARY = torch.library.Library(TORCH_NAMESPACE, "DEF")
for _schema in OPERATOR_SCHEMAS.values():
    _DEFINITION_LIBRARY.define(_schema.torch_schema)

_IMPLEMENTATION_LIBRARY = torch.library.Library(TORCH_NAMESPACE, "IMPL")
_register_kernels(_IMPLEMENTATION_LIBRARY, "CPU", _REFERENCE_KERNELS)
_register_kernels(_IMPLEMENTATION_LIBRARY, "Meta", _META_KERNELS)

_REGISTERED_DEVICE_DISPATCHES = ["CPU", "Meta"]
_backend_implementations: dict[str, BackendImplementation] = {
    "CPU": "reference"
}
if torch.cuda.is_available():
    _register_kernels(_IMPLEMENTATION_LIBRARY, "CUDA", _REFERENCE_KERNELS)
    _REGISTERED_DEVICE_DISPATCHES.append("CUDA")
    _backend_implementations["CUDA"] = "reference"
if _npu_is_available():
    _register_kernels(
        _IMPLEMENTATION_LIBRARY,
        "PrivateUse1",
        _REFERENCE_KERNELS,
    )
    _REGISTERED_DEVICE_DISPATCHES.append("PrivateUse1")
    _backend_implementations["PrivateUse1"] = "reference"

REGISTERED_DEVICE_DISPATCHES = tuple(_REGISTERED_DEVICE_DISPATCHES)
_BACKEND_IMPLEMENTATIONS = MappingProxyType(
    _backend_implementations
)
del _backend_implementations, _REGISTERED_DEVICE_DISPATCHES


def registered_device_dispatches() -> tuple[str, ...]:
    """Return dispatches registered in the current runtime."""
    return REGISTERED_DEVICE_DISPATCHES


def operator_backend_status(
    operator: str,
) -> tuple[OperatorBackendStatus, ...]:
    """Return explicit execution implementation status for one operator."""
    try:
        schema_for_torch_name(operator)
    except KeyError as error:
        raise KeyError(
            f"Unknown MDC Torch operator: {operator}"
        ) from error
    return backend_status_snapshot(operator, _BACKEND_IMPLEMENTATIONS)


def operator_schemas() -> Iterable[Any]:
    """Return immutable operator schema values."""
    return OPERATOR_SCHEMAS.values()
