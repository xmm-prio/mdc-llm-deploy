"""Numerical metrics and activation saturation collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from mdc_llm_deploy.quantization import MinMaxLinear


@dataclass(frozen=True, slots=True)
class AccuracyMetrics:
    """Describe numerical agreement between reference and actual tensors."""

    cosine: float
    max_absolute_error: float
    mean_absolute_error: float
    mean_relative_error: float
    finite: bool

    def to_dict(self) -> dict[str, float | bool]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def compare_arrays(reference: np.ndarray, actual: np.ndarray) -> AccuracyMetrics:
    """Compare equal-shape tensors in float64 for stable reporting."""
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, actual={actual.shape}")
    reference64 = reference.astype(np.float64, copy=False)
    actual64 = actual.astype(np.float64, copy=False)
    finite = bool(np.isfinite(reference64).all() and np.isfinite(actual64).all())
    if not finite:
        return AccuracyMetrics(
            cosine=float("nan"),
            max_absolute_error=float("nan"),
            mean_absolute_error=float("nan"),
            mean_relative_error=float("nan"),
            finite=False,
        )

    difference = np.abs(actual64 - reference64)
    denominator = np.linalg.norm(reference64.ravel()) * np.linalg.norm(actual64.ravel())
    if denominator == 0.0:
        cosine = 1.0 if np.array_equal(reference64, actual64) else 0.0
    else:
        cosine = float(np.dot(reference64.ravel(), actual64.ravel()) / denominator)
    relative = difference / np.maximum(np.abs(reference64), np.finfo(np.float32).eps)
    return AccuracyMetrics(
        cosine=cosine,
        max_absolute_error=float(difference.max(initial=0.0)),
        mean_absolute_error=float(difference.mean()),
        mean_relative_error=float(relative.mean()),
        finite=True,
    )


def compare_tensors(reference: Tensor, actual: Tensor) -> AccuracyMetrics:
    """Compare Torch tensors after moving them to CPU."""
    return compare_arrays(reference.detach().float().cpu().numpy(), actual.detach().float().cpu().numpy())


class SaturationCollector:
    """Collect clipped activation counts for every quantized Linear."""

    def __init__(self, module: nn.Module) -> None:
        self._counts: dict[str, tuple[int, int]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, child in module.named_modules():
            if isinstance(child, MinMaxLinear) and child.activation_scale is not None:
                self._handles.append(child.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name: str):
        def collect(module: nn.Module, args: tuple[Tensor, ...]) -> None:
            if not isinstance(module, MinMaxLinear) or module.activation_scale is None:
                return
            values = args[0]
            scale = module.activation_scale.to(device=values.device, dtype=values.dtype)
            clipped = int((values.abs() / scale > 127).sum().item())
            previous_clipped, previous_total = self._counts.get(name, (0, 0))
            self._counts[name] = (previous_clipped + clipped, previous_total + values.numel())

        return collect

    def close(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def report(self) -> dict[str, dict[str, float | int]]:
        """Return per-module and aggregate saturation statistics."""
        report: dict[str, dict[str, float | int]] = {}
        total_clipped = 0
        total_values = 0
        for name, (clipped, count) in sorted(self._counts.items()):
            total_clipped += clipped
            total_values += count
            report[name] = {
                "clipped": clipped,
                "total": count,
                "ratio": clipped / count if count else 0.0,
            }
        report["__all__"] = {
            "clipped": total_clipped,
            "total": total_values,
            "ratio": total_clipped / total_values if total_values else 0.0,
        }
        return report


def load_array(path: Path, *, dtype: str = "float16", shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Load NPY or raw binary output with explicit binary metadata."""
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if shape is None:
        raise ValueError("shape is required for raw binary input")
    array = np.fromfile(path, dtype=np.dtype(dtype))
    expected_size = int(np.prod(shape))
    if array.size != expected_size:
        raise ValueError(f"binary element count {array.size} does not match shape {shape}")
    return array.reshape(shape)


__all__ = [
    "AccuracyMetrics",
    "SaturationCollector",
    "compare_arrays",
    "compare_tensors",
    "load_array",
]
