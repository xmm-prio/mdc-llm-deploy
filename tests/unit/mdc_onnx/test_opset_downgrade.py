from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.mdc_onnx.opset_downgrade import downgrade_opset


def _identity_model(opset: int = 21) -> onnx.ModelProto:
    value = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="identity")],
        "identity",
        [value],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def test_downgrade_opset_in_place() -> None:
    model = _identity_model()

    returned = downgrade_opset(model)

    assert returned is model
    assert [(item.domain, item.version) for item in model.opset_import] == [("", 18)]


def test_mdc_default_domain_schema_is_accepted() -> None:
    model = _identity_model()
    model.graph.node[0].op_type = "NPUAscendQuantV2"
    model.graph.node[0].input.append("scale")
    model.graph.node[0].attribute.append(helper.make_attribute("dtype", 2))
    model.graph.initializer.append(
        numpy_helper.from_array(np.array(2.0, dtype=np.float32), "scale")
    )
    model.graph.output[0].type.tensor_type.elem_type = TensorProto.INT8

    downgrade_opset(model)

    assert model.opset_import[0].version == 18


def test_newer_operator_schema_is_rejected_and_rolled_back() -> None:
    model = _identity_model()
    model.graph.node[0].op_type = "Gelu"
    original = model.SerializeToString()

    with pytest.raises(ValueError, match="no default-domain schema at opset 18"):
        downgrade_opset(model)

    assert model.SerializeToString() == original


def test_opset_below_target_is_not_upgraded() -> None:
    model = _identity_model(opset=17)

    with pytest.raises(ValueError, match="opset upgrade is not supported"):
        downgrade_opset(model)


def test_duplicate_default_domain_import_is_rejected() -> None:
    model = _identity_model()
    model.opset_import.append(helper.make_opsetid("ai.onnx", 21))

    with pytest.raises(ValueError, match="exactly one"):
        downgrade_opset(model)


def test_subgraph_schema_is_checked() -> None:
    then_graph = helper.make_graph(
        [helper.make_node("Gelu", ["x"], ["then_y"])],
        "then",
        [],
        [helper.make_tensor_value_info("then_y", TensorProto.FLOAT, [1])],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["else_y"])],
        "else",
        [],
        [helper.make_tensor_value_info("else_y", TensorProto.FLOAT, [1])],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If",
                ["condition"],
                ["y"],
                then_branch=then_graph,
                else_branch=else_graph,
            )
        ],
        "conditional",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

    with pytest.raises(ValueError, match="Gelu"):
        downgrade_opset(model)
