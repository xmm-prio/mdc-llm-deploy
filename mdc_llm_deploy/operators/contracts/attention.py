"""Canonical ONNX ABI for FusedInferAttentionScore."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType
from typing import Final


class AttentionInput(IntEnum):
    """Input slots used by the complete MDC attention ONNX node."""

    QUERY = 0
    KEY = 1
    VALUE = 2
    PSE_SHIFT = 3
    ATTEN_MASK = 4
    ACTUAL_SEQ_LENGTHS = 5
    ACTUAL_SEQ_LENGTHS_KV = 6
    DEQUANT_SCALE1 = 7
    QUANT_SCALE1 = 8
    DEQUANT_SCALE2 = 9
    QUANT_SCALE2 = 10
    QUANT_OFFSET2 = 11
    ANTIQUANT_SCALE = 12
    ANTIQUANT_OFFSET = 13
    BLOCK_TABLE = 14
    QUERY_PADDING_SIZE = 15
    KV_PADDING_SIZE = 16
    KEY_ANTIQUANT_SCALE = 17
    KEY_ANTIQUANT_OFFSET = 18
    VALUE_ANTIQUANT_SCALE = 19
    VALUE_ANTIQUANT_OFFSET = 20
    KEY_SHARED_PREFIX = 21
    VALUE_SHARED_PREFIX = 22
    ACTUAL_SHARED_PREFIX_LEN = 23
    QUERY_ROPE = 24
    KEY_ROPE = 25
    KEY_ROPE_ANTIQUANT_SCALE = 26
    DEQUANT_SCALE_QUERY = 27
    LEARNABLE_SINK = 28


ATTENTION_INPUT_COUNT: Final = 29
ATTENTION_OUTPUT_COUNT: Final = 2
RELEASE_ATTENTION_ATTRIBUTES: Final[Mapping[str, int | str]] = (
    MappingProxyType(
    {
        "input_layout": "BNSD",
        "sparse_mode": 0,
        "pre_tokens": 2147483647,
        "next_tokens": 2147483647,
        "inner_precise": 0,
        "block_size": 0,
        "antiquant_mode": 0,
        "softmax_lse_flag": 0,
        "key_antiquant_mode": 0,
        "value_antiquant_mode": 0,
        "query_quant_mode": 0,
    }
    )
)
