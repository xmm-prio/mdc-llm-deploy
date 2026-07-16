from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.fx import Graph, GraphModule

from mdc_llm_deploy.errors import QuantizationConfigError
from mdc_llm_deploy.graph.lifecycle import (
    GraphMetadata,
    GraphStage,
    TensorAbi,
    set_metadata,
)
from mdc_llm_deploy.quantization.calibration import (
    _CalibrationInterpreter,
    collect_calibration_samples,
)
from mdc_llm_deploy.quantization.materialization import _require_same_device


def _graph() -> GraphModule:
    root = nn.Module()
    root.add_module("linear", nn.Linear(2, 2, bias=False))
    graph = Graph()
    value = graph.placeholder("x")
    weight = graph.get_attr("linear.weight")
    value = graph.call_function(
        torch.ops.aten.linear.default,
        (value, weight, None),
    )
    graph.output(value)
    module = GraphModule(root, graph)
    set_metadata(
        module,
        GraphMetadata(
            schema_version=1,
            stage=GraphStage.FLOAT_PREFILL,
            model_kind="dense",
            input_abi=(TensorAbi("x", "float32", (1, 2)),),
            output_abi=(TensorAbi("output", "float32", (1, 2)),),
            sequence_length=2,
        ),
    )
    return module


def _graph_with_bias_input() -> GraphModule:
    root = nn.Module()
    root.add_module("linear", nn.Linear(2, 2, bias=False))
    graph = Graph()
    value = graph.placeholder("x")
    bias = graph.placeholder("bias")
    weight = graph.get_attr("linear.weight")
    value = graph.call_function(
        torch.ops.aten.linear.default,
        (value, weight, bias),
    )
    graph.output(value)
    module = GraphModule(root, graph)
    set_metadata(
        module,
        GraphMetadata(
            schema_version=1,
            stage=GraphStage.FLOAT_PREFILL,
            model_kind="dense",
            input_abi=(
                TensorAbi("x", "float32", (1, 2)),
                TensorAbi("bias", "float32", (2,)),
            ),
            output_abi=(TensorAbi("output", "float32", (1, 2)),),
            sequence_length=2,
        ),
    )
    return module


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("self_attn", "self_attn"),
        ("self_attn.rotary", "self_attn"),
        ("self_attn2", None),
        ("block.self_attn", None),
    ],
)
def test_calibration_attention_owner_uses_shared_fqn_rule(
    candidate: str,
    expected: str | None,
) -> None:
    graph = _graph()
    node = next(item for item in graph.graph.nodes if item.op == "call_function")
    node.meta["nn_module_stack"] = {"owner": (candidate, object())}

    interpreter = _CalibrationInterpreter(graph, ("self_attn",))

    assert interpreter._attention_fqn(node) == expected


def test_collect_calibration_samples_aggregates_linear_inputs() -> None:
    first = torch.tensor([[1.0, 2.0]], requires_grad=True)
    second = torch.tensor([[3.0, 4.0]])

    samples = collect_calibration_samples(
        _graph(),
        [{"x": first}, {"x": second}],
    )

    torch.testing.assert_close(
        samples["linear"],
        torch.cat((first.detach(), second)),
    )
    assert samples["linear"].device.type == "cpu"
    assert not samples["linear"].requires_grad


def test_collect_calibration_samples_accepts_mapping_order_independent_of_abi() -> None:
    samples = collect_calibration_samples(
        _graph_with_bias_input(),
        [{"bias": torch.zeros(2), "x": torch.ones(1, 2)}],
    )

    torch.testing.assert_close(samples["linear"], torch.ones(1, 2))


def test_collect_calibration_samples_rejects_empty_dataloader() -> None:
    with pytest.raises(
        QuantizationConfigError,
        match="must yield at least one batch",
    ):
        collect_calibration_samples(_graph(), [])


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        ({"wrong": torch.ones(1, 2)}, "Calibration keys"),
        ({"x": torch.ones(2, 2)}, "Calibration shape"),
        ({"x": torch.ones(1, 2, dtype=torch.float64)}, "Calibration dtype"),
        ({"x": torch.ones(1, 2, device="meta")}, "Calibration device"),
        ({"x": torch.tensor([[float("inf"), 0.0]])}, "contains NaN or Inf"),
    ],
)
def test_collect_calibration_samples_rejects_invalid_batches(
    batch: dict[str, torch.Tensor],
    message: str,
) -> None:
    with pytest.raises(QuantizationConfigError, match=message):
        collect_calibration_samples(_graph(), [batch])


@pytest.mark.parametrize("algorithm", ["GPTQ", "MoeExpert"])
def test_materialization_rejects_calibration_device_mismatch(
    algorithm: str,
) -> None:
    with pytest.raises(
        QuantizationConfigError,
        match=rf"{algorithm} device mismatch",
    ):
        _require_same_device(
            torch.ones(2),
            torch.ones(2, device="meta"),
            algorithm=algorithm,
            fqn="target",
        )
