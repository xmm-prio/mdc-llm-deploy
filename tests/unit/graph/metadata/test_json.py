from __future__ import annotations

import copy
import pickle
from collections import UserDict
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from typing import Any

import pytest

from mdc_llm_deploy.errors import GraphStateError
from mdc_llm_deploy.graph.metadata.json import FrozenJsonMapping, freeze_json
from mdc_llm_deploy.graph.metadata.types import (
    GRAPH_SCHEMA_VERSION,
    GraphMetadata,
    GraphStage,
    TensorAbi,
)


def _mapping(*items: tuple[Any, Any]) -> FrozenJsonMapping:
    return FrozenJsonMapping(items)  # type: ignore[arg-type]


def _metadata(properties: dict[str, Any]) -> GraphMetadata:
    return GraphMetadata(
        schema_version=GRAPH_SCHEMA_VERSION,
        stage=GraphStage.FLOAT_PREFILL,
        model_kind="dense",
        input_abi=(TensorAbi("x", "float32", (1,)),),
        output_abi=(TensorAbi("output", "float32", (1,)),),
        sequence_length=1,
        properties=properties,
    )


@pytest.mark.parametrize("key", ["first", "missing"])
def test_string_lookup_builds_lazy_index_atomically(key: str) -> None:
    mapping = _mapping(("first", 1), ("second", 2))

    assert mapping._index is None
    if key == "first":
        assert mapping[key] == 1
    else:
        with pytest.raises(KeyError) as error:
            mapping[key]
        assert error.value.args == ("missing",)

    assert mapping._index == {"first": 1, "second": 2}
    assert mapping["second"] == 2
    with pytest.raises(KeyError) as error:
        mapping["absent"]
    assert error.value.args == ("absent",)


def test_empty_mapping_caches_empty_index_after_lookup() -> None:
    mapping = _mapping()

    with pytest.raises(KeyError):
        mapping["missing"]

    assert mapping._index == {}


def test_mapping_protocol_preserves_order_length_repr_and_equality() -> None:
    mapping = _mapping(("first", 1), ("second", 2))

    assert mapping["first"] == 1
    assert mapping.get("first") == 1
    assert mapping.get("missing") is None
    assert "second" in mapping
    assert "missing" not in mapping
    assert list(mapping) == ["first", "second"]
    assert len(mapping) == 2
    assert repr(mapping) == "{'first': 1, 'second': 2}"
    assert mapping == {"first": 1, "second": 2}
    assert mapping == UserDict({"first": 1, "second": 2})


def test_duplicate_keys_keep_first_lookup_and_last_repr_value() -> None:
    mapping = _mapping(("x", 1), ("x", 2), ("y", 3))

    assert mapping["x"] == 1
    assert mapping.get("x") == 1
    assert "x" in mapping
    assert mapping == {"x": 1, "y": 3}
    assert mapping != {"x": 2, "y": 3}
    assert list(mapping) == ["x", "x", "y"]
    assert len(mapping) == 3
    assert repr(mapping) == "{'x': 2, 'y': 3}"
    assert mapping._index == {"x": 1, "y": 3}


class StringSubclass(str):
    pass


def test_non_exact_string_queries_keep_linear_lookup() -> None:
    mapping = _mapping(("x", 1))

    assert mapping[StringSubclass("x")] == 1
    with pytest.raises(KeyError) as error:
        mapping[["missing"]]  # type: ignore[index]

    assert error.value.args == (["missing"],)
    assert mapping._index is None


def test_invalid_candidate_keys_prevent_index_publication() -> None:
    mapping = _mapping(("x", 1), (["invalid"], 2))

    assert mapping["x"] == 1
    assert mapping[["invalid"]] == 2  # type: ignore[index]
    with pytest.raises(KeyError) as error:
        mapping["missing"]

    assert error.value.args == ("missing",)
    assert mapping._index is None


@pytest.mark.parametrize("attribute", ["_items", "_index", "new_attribute"])
def test_mapping_rejects_all_external_assignment(attribute: str) -> None:
    mapping = _mapping(("x", 1))

    with pytest.raises(AttributeError, match="is immutable"):
        setattr(mapping, attribute, None)


def test_published_index_rejects_direct_item_assignment() -> None:
    mapping = _mapping(("x", 1))
    assert mapping["x"] == 1
    index = mapping._index
    assert index is not None

    with pytest.raises(TypeError):
        index["x"] = 2  # type: ignore[index]

    assert mapping["x"] == 1
    assert repr(mapping) == "{'x': 1}"
    assert dict(mapping) == {"x": 1}


def test_deepcopy_identity_and_unhashability_hold_before_and_after_index() -> None:
    mapping = _mapping(("x", 1))

    assert copy.deepcopy(mapping) is mapping
    with pytest.raises(TypeError):
        hash(mapping)

    assert mapping["x"] == 1
    assert copy.deepcopy(mapping) is mapping
    with pytest.raises(TypeError):
        hash(mapping)


@pytest.mark.parametrize("build_index", [False, True])
def test_pickle_still_fails_while_restoring_immutable_mapping(
    build_index: bool,
) -> None:
    mapping = _mapping(("x", 1))
    if build_index:
        assert mapping["x"] == 1
    payload = pickle.dumps(mapping)

    with pytest.raises(AttributeError, match="is immutable"):
        pickle.loads(payload)


def test_freeze_json_captures_deeply_immutable_snapshot() -> None:
    source = {
        "mapping": {"items": [1, {"enabled": True}]},
        "tuple": (2, [3]),
    }

    frozen = freeze_json(source)
    source["mapping"]["items"].append(4)
    source["mapping"]["items"][1]["enabled"] = False
    source["tuple"][1].append(5)

    assert isinstance(frozen, FrozenJsonMapping)
    assert isinstance(frozen["mapping"], FrozenJsonMapping)
    assert isinstance(frozen["mapping"]["items"], tuple)
    assert isinstance(frozen["mapping"]["items"][1], FrozenJsonMapping)
    assert frozen == {
        "mapping": {"items": (1, {"enabled": True})},
        "tuple": (2, (3,)),
    }


def test_freeze_json_rejects_circular_references() -> None:
    source: dict[str, Any] = {}
    source["self"] = source

    with pytest.raises(GraphStateError, match="circular references"):
        freeze_json(source)


def test_graph_metadata_dataclass_integration_preserves_value_semantics() -> None:
    metadata = _metadata({"nested": {"items": [1, 2]}})
    properties = metadata.properties
    assert isinstance(properties, FrozenJsonMapping)
    assert properties["nested"]["items"] == (1, 2)
    assert properties._index is not None

    replaced = replace(metadata, sequence_length=2)
    copied = copy.deepcopy(metadata)

    assert is_dataclass(GraphMetadata)
    assert GraphMetadata.__dataclass_params__.frozen is True
    assert "__dict__" not in GraphMetadata.__slots__
    assert {field.name for field in fields(GraphMetadata)}
    assert replaced.properties == properties
    assert copied == metadata
    assert copied is not metadata
    assert copied.properties is properties
    assert replaced.properties is not properties
    assert replaced.properties == properties
    assert replaced.properties._index is not properties._index
    with pytest.raises(FrozenInstanceError):
        metadata.sequence_length = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        replaced.properties["nested"]["items"][0] = 3  # type: ignore[index]
