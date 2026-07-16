"""AscendDequant scale wire-format codec."""

from __future__ import annotations

import torch
from torch import Tensor


def encode_dequant_scale(scale: Tensor) -> Tensor:
    """Encode full FP32 bits in the low half of uint64 values."""
    source = scale.detach().contiguous().float()
    if not torch.isfinite(source).all():
        raise ValueError("scale contains NaN or Inf")
    if (source <= 0).any():
        raise ValueError("scale must be positive")
    low_bits = (
        source.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    )
    return low_bits.to(torch.uint64)


def decode_dequant_scale(encoded: Tensor) -> Tensor:
    """Decode restricted AscendDequant uint64 scale representation."""
    if encoded.dtype != torch.uint64:
        raise TypeError("encoded scale must use uint64")
    raw = encoded.to(torch.int64)
    if ((raw >> 32) != 0).any():
        raise ValueError("encoded scale high 32 bits must be zero")
    decoded = (
        (raw & 0xFFFFFFFF)
        .to(torch.int32)
        .view(torch.float32)
    )
    if not torch.isfinite(decoded).all():
        raise ValueError(
            "encoded scale decodes to NaN or Inf"
        )
    return decoded
