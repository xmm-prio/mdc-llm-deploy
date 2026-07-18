"""Final ApplyRotaryPosEmb initializer binding contracts."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.onnx.validation.model import (
    validate_mdc_model_structure,
)

_NODE_NAME = "rope.layer.0"
_FLOAT16_DTYPE = np.dtype(np.float16)


def _initializer(name: str, dtype: np.dtype[np.generic]) -> onnx.TensorProto:
    return numpy_helper.from_array(
        np.full((1, 2), 0.25, dtype=dtype),
        name=name,
    )


def _rope_model(
    *,
    tensor_dtype: int = TensorProto.FLOAT16,
    initializer_dtype: np.dtype[np.generic] = _FLOAT16_DTYPE,
    missing_role: str | None = None,
    indirect_role: str | None = None,
) -> onnx.ModelProto:
    role_names = {
        "cos": "private.cos.tensor",
        "sin": "private.sin.tensor",
    }
    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []
    for role, name in role_names.items():
        if role == missing_role:
            continue
        initializer_name = f"{name}.source" if role == indirect_role else name
        initializers.append(_initializer(initializer_name, initializer_dtype))
        if role == indirect_role:
            nodes.append(
                helper.make_node(
                    "Identity",
                    [initializer_name],
                    [name],
                    name=f"{role}.producer",
                )
            )
    nodes.append(
        helper.make_node(
            "ApplyRotaryPosEmb",
            ["query", "key", role_names["cos"], role_names["sin"]],
            ["query_out", "key_out"],
            name=_NODE_NAME,
            layout=1,
            rotary_mode="half",
        )
    )
    graph = helper.make_graph(
        nodes,
        "rope_contract",
        [
            helper.make_tensor_value_info(
                "query",
                tensor_dtype,
                (1, 2, 1, 2),
            ),
            helper.make_tensor_value_info(
                "key",
                tensor_dtype,
                (1, 2, 1, 2),
            ),
        ],
        [
            helper.make_tensor_value_info(
                "query_out",
                tensor_dtype,
                (1, 2, 1, 2),
            ),
            helper.make_tensor_value_info(
                "key_out",
                tensor_dtype,
                (1, 2, 1, 2),
            ),
        ],
        initializer=initializers,
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    )


@pytest.mark.parametrize(
    ("tensor_dtype", "initializer_dtype"),
    [
        (TensorProto.FLOAT16, np.dtype(np.float16)),
        (TensorProto.FLOAT, np.dtype(np.float32)),
    ],
)
def test_rope_accepts_direct_float_initializers(
    tensor_dtype: int,
    initializer_dtype: np.dtype[np.generic],
) -> None:
    validate_mdc_model_structure(
        _rope_model(
            tensor_dtype=tensor_dtype,
            initializer_dtype=initializer_dtype,
        )
    )


@pytest.mark.parametrize("role", ["cos", "sin"])
def test_rope_rejects_missing_initializer_without_leaking_tensor(
    role: str,
) -> None:
    with pytest.raises(OnnxExportError) as raised:
        validate_mdc_model_structure(_rope_model(missing_role=role))

    message = str(raised.value)
    assert _NODE_NAME in message
    assert role in message
    assert "directly reference an initializer" in message
    assert f"private.{role}.tensor" not in message


@pytest.mark.parametrize("role", ["cos", "sin"])
def test_rope_rejects_indirect_initializer_producer(role: str) -> None:
    with pytest.raises(OnnxExportError) as raised:
        validate_mdc_model_structure(_rope_model(indirect_role=role))

    message = str(raised.value)
    assert _NODE_NAME in message
    assert role in message
    assert "directly reference an initializer" in message
    assert f"private.{role}.tensor" not in message


def test_rope_rejects_unsupported_initializer_dtype() -> None:
    with pytest.raises(OnnxExportError) as raised:
        validate_mdc_model_structure(
            _rope_model(initializer_dtype=np.dtype(np.int32))
        )

    message = str(raised.value)
    assert _NODE_NAME in message
    assert "cos" in message
    assert "dtype must be FLOAT16 or FLOAT" in message
    assert "private.cos.tensor" not in message


def test_rope_rejects_initializer_dtype_mismatch() -> None:
    with pytest.raises(OnnxExportError) as raised:
        validate_mdc_model_structure(
            _rope_model(
                tensor_dtype=TensorProto.FLOAT16,
                initializer_dtype=np.dtype(np.float32),
            )
        )

    message = str(raised.value)
    assert _NODE_NAME in message
    assert "cos" in message
    assert "dtype must match query input" in message
    assert "private.cos.tensor" not in message
