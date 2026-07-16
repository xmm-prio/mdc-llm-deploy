"""Operator registration entry contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_operator_entry_registers_once_with_import_side_effects() -> None:
    project_root = Path(__file__).parents[3]
    script = """
import importlib
import torch

calls = []
torch.onnx.register_custom_op_symbolic = lambda *args: calls.append(args)

operators = importlib.import_module("mdc_llm_deploy.operators")
expected = len(operators.OPERATOR_SCHEMAS)
assert len(calls) == expected
assert hasattr(torch.ops.mdc_llm_deploy, "rms_norm")

assert importlib.import_module("mdc_llm_deploy.operators") is operators
operators.register_operators()
operators.register_operators()
assert len(calls) == expected
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
