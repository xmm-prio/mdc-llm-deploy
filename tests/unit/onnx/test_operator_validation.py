"""Node-level MDC ONNX validator dispatch contracts."""

import pytest
from onnx import helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.onnx.validation.operator import (
    validate_operator,
)


def test_operator_validator_rejects_unknown_schema_entry() -> None:
    node = helper.make_node("FutureMdcOperator", [], ["output"])

    with pytest.raises(
        OnnxExportError,
        match="No MDC ONNX validator",
    ):
        validate_operator(node, "masked")


def test_operator_validator_accepts_valid_moe_abi() -> None:
    node = helper.make_node(
        "MoeExpert",
        [f"input.{index}" for index in range(6)],
        ["output"],
    )

    validate_operator(node, "masked")
