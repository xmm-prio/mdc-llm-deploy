"""FakeTensor kernel for ApplyRotaryPosEmb."""

from __future__ import annotations

import torch

from .contract import validate_torch_inputs


def fake(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int = 1,
    rotary_mode: str = "half",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate metadata and create shape-preserving abstract outputs."""
    validate_torch_inputs(
        query, key, cos, sin, layout, rotary_mode, check_values=False
    )
    return torch.empty_like(query), torch.empty_like(key)
