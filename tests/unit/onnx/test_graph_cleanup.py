from __future__ import annotations

import random

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mdc_llm_deploy.errors import OnnxExportError
from mdc_llm_deploy.onnx.transform.cleanup import (
    remove_dynamic_value_info,
    remove_redundant_identities,
    topologically_sort,
)
from mdc_llm_deploy.onnx.validation.topology import CUSTOM_OPS, STANDARD_DOMAINS


def _model(
    nodes: list[onnx.NodeProto],
    *,
    outputs: tuple[str, ...] = ("output",),
) -> onnx.ModelProto:
    zero = numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), name="zero")
    graph = helper.make_graph(
        nodes,
        "cleanup",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1,))],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, (1,))
            for name in outputs
        ],
        initializer=[zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _node_names(model: onnx.ModelProto) -> list[str]:
    return [node.name for node in model.graph.node]


def _legacy_remove_redundant_identities(model: onnx.ModelProto) -> None:
    """Frozen reference for the pre-optimization cleanup behavior."""
    while True:
        nodes = list(model.graph.node)
        graph_outputs = {item.name for item in model.graph.output}
        producers = {
            output: node for node in model.graph.node for output in node.output
        }
        consumer_counts: dict[str, int] = {}
        for node in nodes:
            for name in node.input:
                if name:
                    consumer_counts[name] = consumer_counts.get(name, 0) + 1
        removable: tuple[onnx.NodeProto, onnx.NodeProto, str, str] | None = None
        for identity in nodes:
            if (
                identity.op_type != "Identity"
                or identity.domain not in STANDARD_DOMAINS
                or identity.attribute
                or len(identity.input) != 1
                or len(identity.output) != 1
            ):
                continue
            source = identity.input[0]
            output = identity.output[0]
            producer = producers.get(source)
            if (
                not source
                or not output
                or source in graph_outputs
                or consumer_counts.get(source) != 1
                or producer is None
                or producer.op_type in CUSTOM_OPS
            ):
                continue
            removable = identity, producer, source, output
            break
        if removable is None:
            return
        identity, producer, source, output = removable
        producer.output[:] = [
            output if name == source else name for name in producer.output
        ]
        typed_values = {item.name for item in model.graph.input}
        typed_values.update(item.name for item in model.graph.output)
        typed_values.update(item.name for item in model.graph.value_info)
        if output not in typed_values:
            source_info = next(
                (item for item in model.graph.value_info if item.name == source),
                None,
            )
            if source_info is not None:
                source_info.name = output
        model.graph.node.remove(identity)


def _legacy_order(model: onnx.ModelProto) -> list[str]:
    known = {item.name for item in model.graph.input}
    known.update(item.name for item in model.graph.initializer)
    pending = list(model.graph.node)
    ordered: list[str] = []
    while pending:
        ready = next(
            node
            for node in pending
            if all(not name or name in known for name in node.input)
        )
        pending.remove(ready)
        ordered.append(ready.name)
        known.update(ready.output)
    return ordered


def _assert_atomic_sort_failure(
    model: onnx.ModelProto,
    *message_parts: str,
) -> None:
    original_order = _node_names(model)

    with pytest.raises(OnnxExportError) as error:
        topologically_sort(model)

    message = str(error.value)
    assert message.startswith("Lowered ONNX graph cannot be topologically sorted:")
    assert all(part in message for part in message_parts)
    assert _node_names(model) == original_order


def test_topologically_sort_restores_dependency_order() -> None:
    model = _model(
        [
            helper.make_node("Relu", ["hidden"], ["output"], name="second"),
            helper.make_node("Add", ["input", "zero"], ["hidden"], name="first"),
        ]
    )

    topologically_sort(model)

    assert [node.name for node in model.graph.node] == ["first", "second"]


def test_topologically_sort_rejects_missing_producer() -> None:
    model = _model(
        [helper.make_node("Relu", ["missing"], ["output"], name="blocked")]
    )

    _assert_atomic_sort_failure(model, "blocked", "missing")


def test_topologically_sort_uses_legacy_stable_order_not_fifo_order() -> None:
    model = _model(
        [
            helper.make_node("Relu", ["from_first"], ["output"], name="unlocked"),
            helper.make_node("Identity", ["input"], ["from_first"], name="first"),
            helper.make_node("Neg", ["input"], ["side"], name="independent"),
        ],
        outputs=("output", "side"),
    )

    topologically_sort(model)

    assert _node_names(model) == ["first", "unlocked", "independent"]


def test_topologically_sort_matches_legacy_order_for_branched_dag() -> None:
    model = _model(
        [
            helper.make_node("Add", ["left", "right"], ["output"], name="join"),
            helper.make_node("Relu", ["source"], ["left"], name="left"),
            helper.make_node("Neg", ["input"], ["side"], name="independent"),
            helper.make_node("Identity", ["input"], ["source"], name="source"),
            helper.make_node("Abs", ["source"], ["right"], name="right"),
        ],
        outputs=("output", "side"),
    )
    expected = _legacy_order(model)

    topologically_sort(model)

    assert expected == ["independent", "source", "left", "right", "join"]
    assert _node_names(model) == expected


def test_topologically_sort_deduplicates_producer_consumer_dependencies() -> None:
    model = _model(
        [
            helper.make_node(
                "Add",
                ["left", "left", "right"],
                ["output"],
                name="consumer",
            ),
            helper.make_node(
                "Split",
                ["input"],
                ["left", "right"],
                name="producer",
            ),
        ]
    )

    topologically_sort(model)

    assert _node_names(model) == ["producer", "consumer"]


def test_topologically_sort_ignores_optional_empty_inputs() -> None:
    model = _model(
        [
            helper.make_node(
                "Example",
                ["input", "", "zero"],
                ["output"],
                name="optional",
            )
        ]
    )

    topologically_sort(model)

    assert _node_names(model) == ["optional"]


@pytest.mark.parametrize(
    ("nodes", "blocked_values"),
    [
        (
            [
                helper.make_node("Identity", ["right"], ["left"], name="left"),
                helper.make_node("Identity", ["left"], ["right"], name="right"),
            ],
            ("left", "right"),
        ),
        (
            [helper.make_node("Identity", ["loop"], ["loop"], name="self")],
            ("loop",),
        ),
    ],
)
def test_topologically_sort_rejects_cycles_atomically(
    nodes: list[onnx.NodeProto],
    blocked_values: tuple[str, ...],
) -> None:
    model = _model(nodes, outputs=(nodes[-1].output[0],))

    _assert_atomic_sort_failure(model, *blocked_values)


@pytest.mark.parametrize(
    ("nodes", "message_parts"),
    [
        (
            [
                helper.make_node("Identity", ["input"], ["same"], name="first"),
                helper.make_node("Neg", ["input"], ["same"], name="second"),
            ],
            ("'same'", "node 0", "node 1"),
        ),
        (
            [
                helper.make_node(
                    "Split",
                    ["input"],
                    ["same", "same"],
                    name="repeated",
                )
            ],
            ("'same'", "node 0"),
        ),
        (
            [helper.make_node("Identity", ["input"], ["input"], name="conflict")],
            ("'input'", "graph input or initializer", "node 0"),
        ),
        (
            [helper.make_node("Identity", ["input"], ["zero"], name="conflict")],
            ("'zero'", "graph input or initializer", "node 0"),
        ),
    ],
)
def test_topologically_sort_rejects_duplicate_producers_atomically(
    nodes: list[onnx.NodeProto],
    message_parts: tuple[str, ...],
) -> None:
    model = _model(nodes, outputs=(nodes[-1].output[0],))

    _assert_atomic_sort_failure(model, "duplicate producers", *message_parts)


def test_topologically_sort_accepts_empty_graph() -> None:
    model = _model([], outputs=())

    topologically_sort(model)

    assert not model.graph.node


def test_remove_dynamic_value_info_retains_only_static_shapes() -> None:
    model = _model(
        [helper.make_node("Identity", ["input"], ["output"])],
    )
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info("static", TensorProto.FLOAT, (1, 2)),
            helper.make_tensor_value_info(
                "dynamic",
                TensorProto.FLOAT,
                ("sequence", 2),
            ),
        ]
    )

    remove_dynamic_value_info(model)

    assert [item.name for item in model.graph.value_info] == ["static"]


def test_remove_redundant_identities_preserves_fan_out_source() -> None:
    model = _model(
        [
            helper.make_node("Add", ["input", "zero"], ["hidden"], name="source"),
            helper.make_node("Identity", ["hidden"], ["output"], name="identity"),
            helper.make_node("Neg", ["hidden"], ["side"], name="side"),
        ],
        outputs=("output", "side"),
    )

    remove_redundant_identities(model)

    assert [node.name for node in model.graph.node] == [
        "source",
        "identity",
        "side",
    ]


def _clone_model(model: onnx.ModelProto) -> onnx.ModelProto:
    clone = onnx.ModelProto()
    clone.CopyFrom(model)
    return clone


def _assert_cleanup_matches_legacy(model: onnx.ModelProto) -> None:
    expected = _clone_model(model)
    actual = _clone_model(model)
    expected_error: Exception | None = None
    actual_error: Exception | None = None
    try:
        _legacy_remove_redundant_identities(expected)
    except Exception as error:  # pragma: no cover - guards protobuf failure parity
        expected_error = error
    try:
        remove_redundant_identities(actual)
    except Exception as error:  # pragma: no cover - guards protobuf failure parity
        actual_error = error

    assert (
        type(actual_error),
        None if actual_error is None else str(actual_error),
    ) == (
        type(expected_error),
        None if expected_error is None else str(expected_error),
    )
    assert actual.SerializeToString(deterministic=True) == expected.SerializeToString(
        deterministic=True
    )
    assert [
        (node.name, node.op_type, node.domain, tuple(node.input), tuple(node.output))
        for node in actual.graph.node
    ] == [
        (node.name, node.op_type, node.domain, tuple(node.input), tuple(node.output))
        for node in expected.graph.node
    ]
    for actual_items, expected_items in (
        (actual.graph.input, expected.graph.input),
        (actual.graph.output, expected.graph.output),
        (actual.graph.value_info, expected.graph.value_info),
    ):
        assert [item.name for item in actual_items] == [
            item.name for item in expected_items
        ]


def test_remove_redundant_identities_matches_legacy_for_chains() -> None:
    model = _model(
        [
            helper.make_node("Relu", ["input"], ["v0"], name="source"),
            *[
                helper.make_node(
                    "Identity",
                    [f"v{index}"],
                    [f"v{index + 1}"],
                    name=f"identity_{index}",
                )
                for index in range(4)
            ],
            helper.make_node("Neg", ["input"], ["side_0"], name="side_source"),
            helper.make_node("Identity", ["side_0"], ["side_1"], name="side_first"),
            helper.make_node("Identity", ["side_1"], ["side"], name="side_second"),
        ],
        outputs=("v4", "side"),
    )

    _assert_cleanup_matches_legacy(model)

    remove_redundant_identities(model)
    assert _node_names(model) == ["source", "side_source"]
    assert [tuple(node.output) for node in model.graph.node] == [("v4",), ("side",)]


def test_remove_redundant_identities_uses_original_index_not_fifo() -> None:
    model = _model(
        [
            helper.make_node("Identity", ["blocked"], ["shared"], name="early"),
            helper.make_node("Identity", ["unlock"], ["blocked"], name="unlock"),
            helper.make_node("NPURmsNorm", ["input"], ["blocked"], name="custom"),
            helper.make_node("Relu", ["input"], ["unlock"], name="unlock_source"),
            helper.make_node("Neg", ["input"], ["ready"], name="ready_source"),
            helper.make_node("Identity", ["ready"], ["shared"], name="late"),
        ],
        outputs=("final",),
    )
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info("blocked", TensorProto.FLOAT, (1,)),
            helper.make_tensor_value_info("ready", TensorProto.FLOAT, (1,)),
        ]
    )

    _assert_cleanup_matches_legacy(model)


def test_remove_redundant_identities_matches_legacy_name_round_trip() -> None:
    model = _model(
        [
            helper.make_node("Relu", ["input"], ["source"], name="producer"),
            helper.make_node("Identity", ["source"], ["middle"], name="forward"),
            helper.make_node("Identity", ["middle"], ["source"], name="backward"),
        ],
        outputs=("final",),
    )
    model.graph.value_info.add().CopyFrom(
        helper.make_tensor_value_info("source", TensorProto.FLOAT, (1,))
    )

    _assert_cleanup_matches_legacy(model)


def test_remove_redundant_identities_matches_legacy_partial_convergence() -> None:
    model = _model(
        [
            helper.make_node("Relu", ["input"], ["fold_source"], name="fold_source"),
            helper.make_node(
                "Identity",
                ["fold_source"],
                ["folded"],
                name="folded_identity",
            ),
            helper.make_node("Neg", ["input"], ["blocked_source"], name="blocked_source"),
            helper.make_node(
                "Identity",
                ["blocked_source"],
                ["blocked"],
                name="blocked_identity",
            ),
            helper.make_node("Abs", ["blocked_source"], ["side"], name="fanout"),
            helper.make_node(
                "Identity",
                ["blocked"],
                ["malformed", "extra"],
                name="malformed",
            ),
        ],
        outputs=("folded", "blocked", "side"),
    )
    expected = _clone_model(model)
    _legacy_remove_redundant_identities(expected)

    _assert_cleanup_matches_legacy(model)

    assert _node_names(expected) == [
        "fold_source",
        "blocked_source",
        "blocked_identity",
        "fanout",
        "malformed",
    ]


@pytest.mark.parametrize(
    "nodes",
    [
        [
            helper.make_node("Relu", ["input"], ["source"], name="producer"),
            helper.make_node("Identity", ["source"], ["result"], name="identity"),
        ],
        [
            helper.make_node("Relu", ["input"], ["source"], name="producer"),
            helper.make_node("Identity", ["source"], ["result"], name="identity"),
            helper.make_node("Neg", ["source"], ["side"], name="fanout"),
        ],
        [
            helper.make_node("Relu", ["input"], ["source"], name="producer"),
            helper.make_node("Add", ["source", "source"], ["side"], name="duplicate"),
            helper.make_node("Identity", ["source"], ["result"], name="identity"),
        ],
        [helper.make_node("Identity", ["missing"], ["result"], name="identity")],
        [
            helper.make_node("NPURmsNorm", ["input"], ["source"], name="custom"),
            helper.make_node("Identity", ["source"], ["result"], name="identity"),
        ],
    ],
)
def test_remove_redundant_identities_matches_legacy_boundaries(
    nodes: list[onnx.NodeProto],
) -> None:
    outputs = ("source",) if len(nodes) == 2 and nodes[0].op_type == "Relu" else ("result",)
    _assert_cleanup_matches_legacy(_model(nodes, outputs=outputs))


def test_remove_redundant_identities_matches_legacy_malformed_nodes() -> None:
    malformed = [
        helper.make_node("Identity", ["v0"], ["v1"], name="foreign", domain="other"),
        helper.make_node("Identity", ["v1"], ["v2"], name="attributed", axis=0),
        helper.make_node("Identity", [], ["v3"], name="no_input"),
        helper.make_node("Identity", ["v2", "v3"], ["v4"], name="many_inputs"),
        helper.make_node("Identity", ["v4"], [], name="no_output"),
        helper.make_node("Identity", ["v4"], ["v5", "v6"], name="many_outputs"),
        helper.make_node("Identity", [""], ["v7"], name="empty_input"),
        helper.make_node("Identity", ["v7"], [""], name="empty_output"),
        helper.make_node("Identity", ["loop"], ["loop"], name="self_loop"),
    ]
    model = _model(
        [helper.make_node("Relu", ["input"], ["v0"], name="source"), *malformed],
        outputs=("final",),
    )

    _assert_cleanup_matches_legacy(model)


def test_remove_redundant_identities_matches_legacy_duplicate_producers() -> None:
    models = [
        _model(
            [
                helper.make_node("Relu", ["input"], ["same"], name="standard"),
                helper.make_node("NPURmsNorm", ["input"], ["same"], name="custom"),
                helper.make_node("Identity", ["same"], ["result"], name="identity"),
            ],
            outputs=("result",),
        ),
        _model(
            [
                helper.make_node("NPURmsNorm", ["input"], ["same"], name="custom"),
                helper.make_node("Relu", ["input"], ["same"], name="standard"),
                helper.make_node(
                    "Split",
                    ["input"],
                    ["repeat", "repeat"],
                    name="repeated_output",
                ),
                helper.make_node("Identity", ["same"], ["repeat"], name="rename"),
                helper.make_node("Identity", ["repeat"], ["result"], name="consume"),
            ],
            outputs=("result",),
        ),
    ]

    for model in models:
        _assert_cleanup_matches_legacy(model)


@pytest.mark.parametrize("typed_location", ["none", "input", "output", "value_info"])
def test_remove_redundant_identities_matches_legacy_value_info(
    typed_location: str,
) -> None:
    model = _model(
        [
            helper.make_node("Relu", ["input"], ["source"], name="producer"),
            helper.make_node("Identity", ["source"], ["middle"], name="first"),
            helper.make_node("Identity", ["middle"], ["result"], name="second"),
        ],
        outputs=("final",),
    )
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info("source", TensorProto.FLOAT, (1,)),
            helper.make_tensor_value_info("source", TensorProto.FLOAT, (1,)),
            helper.make_tensor_value_info("middle", TensorProto.FLOAT, (1,)),
        ]
    )
    if typed_location == "input":
        model.graph.input.add().CopyFrom(
            helper.make_tensor_value_info("result", TensorProto.FLOAT, (1,))
        )
    elif typed_location == "output":
        model.graph.output.add().CopyFrom(
            helper.make_tensor_value_info("result", TensorProto.FLOAT, (1,))
        )
    elif typed_location == "value_info":
        model.graph.value_info.add().CopyFrom(
            helper.make_tensor_value_info("result", TensorProto.FLOAT, (1,))
        )

    _assert_cleanup_matches_legacy(model)


def test_remove_redundant_identities_fixed_seed_differential() -> None:
    randomizer = random.Random(160100)
    value_names = ["a", "b", "c", "d", ""]
    op_types = ["Identity", "Relu", "NPURmsNorm", "Add"]
    for sample in range(80):
        nodes: list[onnx.NodeProto] = [
            helper.make_node("Relu", ["input"], ["a"], name=f"root_{sample}")
        ]
        for index in range(randomizer.randint(1, 10)):
            op_type = randomizer.choice(op_types)
            input_count = randomizer.choice([0, 1, 1, 1, 2])
            output_count = randomizer.choice([0, 1, 1, 1, 2])
            inputs = [randomizer.choice(value_names) for _ in range(input_count)]
            outputs = [randomizer.choice(value_names) for _ in range(output_count)]
            kwargs: dict[str, int] = {}
            if op_type == "Identity" and randomizer.randrange(7) == 0:
                kwargs["axis"] = 0
            nodes.append(
                helper.make_node(
                    op_type,
                    inputs,
                    outputs,
                    name=f"sample_{sample}_node_{index}",
                    domain="other"
                    if op_type == "Identity" and randomizer.randrange(9) == 0
                    else "",
                    **kwargs,
                )
            )
        model = _model(nodes, outputs=("final",))
        for name in randomizer.choices(value_names[:-1], k=randomizer.randrange(4)):
            model.graph.value_info.add().CopyFrom(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, (1,))
            )

        try:
            _assert_cleanup_matches_legacy(model)
        except AssertionError as error:
            raise AssertionError(f"differential sample {sample}") from error
