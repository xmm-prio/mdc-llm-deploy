"""Module-boundary tests for ONNX Attention lowering."""

from __future__ import annotations

import ast
import inspect

from mdc_llm_deploy.onnx import api
from mdc_llm_deploy.onnx.transform import (
    attention as attention_lowering,
)


def test_attention_lowering_exposes_public_stage_entries_without_api_dependency() -> None:
    assert callable(attention_lowering.lower_maskless_attention)
    assert callable(attention_lowering.lower_rms_norms)
    assert callable(attention_lowering.lower_rope_attention)

    tree = ast.parse(inspect.getsource(attention_lowering))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "api" not in imported_modules


def test_api_preserves_attention_lowering_order() -> None:
    source = inspect.getsource(api._lower)

    assert source.index("lower_maskless_attention") < source.index("lower_rms_norms")
    assert source.index("lower_rms_norms") < source.index("lower_rope_attention")
    assert source.index("lower_rope_attention") < source.index("append_quantized_linears")
