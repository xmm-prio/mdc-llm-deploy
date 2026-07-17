from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn
from torch.fx import Graph, GraphModule

from mdc_llm_deploy.errors import QuantizationConfigError
from mdc_llm_deploy.quantization.config import QuantizationConfig
from mdc_llm_deploy.quantization.planning import planner


def _register_parameter(
    root: nn.Module,
    name: str,
    parameter: nn.Parameter,
) -> None:
    parts = name.split(".")
    owner = root
    for part in parts[:-1]:
        child = owner._modules.get(part)
        if child is None:
            child = nn.Module()
            owner.add_module(part, child)
        owner = child
    owner.register_parameter(parts[-1], parameter)


def _graph_module(
    parameters: list[tuple[str, nn.Parameter]],
    *,
    linear_weights: tuple[str, ...] = (),
    non_aten_weights: tuple[str, ...] = (),
) -> GraphModule:
    root = nn.Module()
    for name, parameter in parameters:
        _register_parameter(root, name, parameter)
    graph = Graph()
    value = graph.placeholder("value")
    result = value
    for name in linear_weights:
        weight = graph.get_attr(name)
        result = graph.call_function(
            torch.ops.aten.linear.default,
            (value, weight, None),
        )
    for name in non_aten_weights:
        weight = graph.get_attr(name)
        result = graph.call_function(torch.add, (value, weight))
    graph.call_function(
        torch.ops.aten.linear.default,
        (value, value, None),
    )
    graph.output(result)
    return GraphModule(root, graph)


def _config(
    *includes: tuple[str, ...] | None,
) -> QuantizationConfig:
    weight = {"bits": 8, "granularity": "per_channel"}
    modifiers = []
    for include in includes or (None,):
        modifier: dict[str, object] = {
            "type": "minmax",
            "linear": {"weight": weight},
            "moe": {"weight": weight},
        }
        if include is not None:
            modifier["include"] = list(include)
        modifiers.append(modifier)
    return QuantizationConfig.from_dict({"modifiers": modifiers})


def _matrix_graph() -> GraphModule:
    parameters = [
        ("dense.weight", nn.Parameter(torch.ones(2, 2))),
        ("unused.weight", nn.Parameter(torch.ones(2, 2))),
        ("rank_one.weight", nn.Parameter(torch.ones(2))),
        ("experts.0.weight", nn.Parameter(torch.ones(2, 2))),
        ("experts.1.weight", nn.Parameter(torch.ones(2, 2))),
        ("shared_expert.weight", nn.Parameter(torch.ones(2, 2))),
        ("packed.expert_weights", nn.Parameter(torch.ones(2, 2))),
        ("packed.Expert_Weights", nn.Parameter(torch.ones(2, 2))),
        ("non_aten.weight", nn.Parameter(torch.ones(2, 2))),
    ]
    return _graph_module(
        parameters,
        linear_weights=(
            "dense.weight",
            "rank_one.weight",
            "experts.0.weight",
            "shared_expert.weight",
        ),
        non_aten_weights=(
            "unused.weight",
            "experts.1.weight",
            "packed.expert_weights",
            "packed.Expert_Weights",
            "non_aten.weight",
        ),
    )


def test_plan_discovers_weights_and_parameters_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _matrix_graph()
    weight_scans = 0
    enumerations: list[bool] = []
    original_weight_names = planner._linear_weight_names
    original_named_parameters = GraphModule.named_parameters

    def counted_weight_names(target: GraphModule) -> frozenset[str]:
        nonlocal weight_scans
        weight_scans += 1
        return original_weight_names(target)

    def counted_named_parameters(
        self: GraphModule,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        enumerations.append(remove_duplicate)
        return original_named_parameters(
            self,
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    monkeypatch.setattr(planner, "_linear_weight_names", counted_weight_names)
    monkeypatch.setattr(
        GraphModule,
        "named_parameters",
        counted_named_parameters,
    )

    planner.plan_quantization(graph, _config())

    assert weight_scans == 1
    assert enumerations == [False]


def test_plan_preserves_exact_parameter_classification() -> None:
    plan = planner.plan_quantization(_matrix_graph(), _config())

    assert [
        (target.target_type, target.parameter_name)
        for target in plan
    ] == [
        ("linear", "dense.weight"),
        ("moe", "experts.0.weight"),
        ("moe", "shared_expert.weight"),
        ("moe", "packed.expert_weights"),
    ]


def test_plan_preserves_modifier_and_category_order() -> None:
    graph = _graph_module(
        [
            ("dense_a.weight", nn.Parameter(torch.ones(2, 2))),
            ("experts.0.weight", nn.Parameter(torch.ones(2, 2))),
            ("dense_b.weight", nn.Parameter(torch.ones(2, 2))),
            ("shared_expert.weight", nn.Parameter(torch.ones(2, 2))),
        ],
        linear_weights=(
            "dense_a.weight",
            "experts.0.weight",
            "dense_b.weight",
            "shared_expert.weight",
        ),
    )

    plan = planner.plan_quantization(
        graph,
        _config(
            ("dense_a", "experts.0"),
            ("dense_b", "shared_expert"),
        ),
    )

    assert [
        (
            target.modifier_index,
            target.target_type,
            target.parameter_name,
            target.fqn,
        )
        for target in plan
    ] == [
        (0, "linear", "dense_a.weight", "dense_a"),
        (0, "moe", "experts.0.weight", "experts.0"),
        (1, "linear", "dense_b.weight", "dense_b"),
        (1, "moe", "shared_expert.weight", "shared_expert"),
    ]


def test_plan_preserves_parameter_alias_names() -> None:
    shared = nn.Parameter(torch.ones(2, 2))
    graph = _graph_module(
        [
            ("aliases_a.weight", shared),
            ("aliases_b.weight", shared),
        ],
        linear_weights=("aliases_a.weight", "aliases_b.weight"),
    )

    plan = planner.plan_quantization(graph, _config())

    assert [target.parameter_name for target in plan] == [
        "aliases_a.weight",
        "aliases_b.weight",
    ]


def test_plan_preserves_first_overlap_error() -> None:
    graph = _graph_module(
        [
            ("dense_a.weight", nn.Parameter(torch.ones(2, 2))),
            ("experts.0.weight", nn.Parameter(torch.ones(2, 2))),
            ("dense_b.weight", nn.Parameter(torch.ones(2, 2))),
        ],
        linear_weights=(
            "dense_a.weight",
            "experts.0.weight",
            "dense_b.weight",
        ),
    )

    with pytest.raises(
        QuantizationConfigError,
        match=r"^Target 'dense_a' is selected by modifiers 0 and 1$",
    ):
        planner.plan_quantization(graph, _config(None, None))
