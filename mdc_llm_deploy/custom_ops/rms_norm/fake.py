"""FakeTensor kernel for RmsNorm."""

from __future__ import annotations

import torch

from .contract import rstd_shape, validate_torch_inputs


def fake(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer output metadata without reading tensor values."""
    validate_torch_inputs(x, gamma, epsilon, check_values=False)
    y = torch.empty_like(x)
    rstd = torch.empty(rstd_shape(x, gamma), dtype=torch.float32, device=x.device)
    return y, rstd
