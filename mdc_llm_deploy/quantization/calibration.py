"""Calibration sample collection at FX quantization boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor
from torch.fx import GraphModule, Interpreter, Node

from ..errors import QuantizationConfigError
from ..graph.fx.inspection import linear_weight_name
from ..graph.fx.ownership import NodeOwnershipIndex
from ..graph.lifecycle import metadata
from .planning import CalibrationPlan


def _first_owner_by_node(
    index: NodeOwnershipIndex,
    nodes: tuple[Node, ...],
    candidates: tuple[str, ...],
) -> dict[Node, str | None]:
    owners = dict.fromkeys(nodes)
    for candidate in candidates:
        for node in index.nodes_belonging_to(candidate):
            if owners[node] is None:
                owners[node] = candidate
    return owners


@dataclass(frozen=True, slots=True)
class _CalibrationOwnershipSnapshot:
    """Resolved calibration ownership for one immutable graph execution phase."""

    attention_by_node: Mapping[Node, str | None]
    moe_by_node: Mapping[Node, str | None]

    @classmethod
    def capture(
        cls,
        graph: GraphModule,
        attention_fqns: tuple[str, ...],
        moe_fqns: tuple[str, ...],
    ) -> _CalibrationOwnershipSnapshot:
        """Resolve ordered calibration owners once for all graph executions."""
        nodes = tuple(
            node for node in graph.graph.nodes if node.op == "call_function"
        )
        index = NodeOwnershipIndex(nodes)
        attention = _first_owner_by_node(index, nodes, attention_fqns)
        moe = _first_owner_by_node(index, nodes, moe_fqns)
        if len(moe_fqns) == 1:
            fallback = moe_fqns[0]
            for node in nodes:
                if (
                    node.target
                    == torch.ops.mdc_llm_deploy.moe_expert.default
                    and moe[node] is None
                ):
                    moe[node] = fallback
        return cls(
            attention_by_node=MappingProxyType(attention),
            moe_by_node=MappingProxyType(moe),
        )


@dataclass(frozen=True, slots=True)
class _CalibrationBoundaryMap:
    """Map actual FX tensor value nodes to calibration target FQNs."""

    targets_by_node: Mapping[Node, tuple[str, ...]]
    nodes_by_target: Mapping[str, tuple[Node, ...]]

    @classmethod
    def capture(
        cls,
        graph: GraphModule,
        required_fqns: frozenset[str],
        ownership: _CalibrationOwnershipSnapshot,
    ) -> _CalibrationBoundaryMap:
        """Resolve calibration targets onto graph tensor-value nodes."""
        targets_by_node: dict[Node, list[str]] = {}
        nodes_by_target: dict[str, list[Node]] = {
            fqn: [] for fqn in required_fqns
        }

        def bind(node: object, target: str) -> None:
            if target not in required_fqns or not isinstance(node, Node):
                return
            target_nodes = nodes_by_target[target]
            if node in target_nodes:
                return
            target_nodes.append(node)
            targets_by_node.setdefault(node, []).append(target)

        for node in graph.graph.nodes:
            if node.op != "call_function":
                continue
            attention_fqn = ownership.attention_by_node.get(node)
            weight_name = linear_weight_name(node)
            if weight_name is not None:
                fqn = weight_name.removesuffix(".weight")
                if node.args:
                    bind(node.args[0], fqn)
                edge = {
                    "q_proj": "query",
                    "k_proj": "key",
                    "v_proj": "value",
                }.get(fqn.rsplit(".", 1)[-1])
                if edge is not None and attention_fqn is not None:
                    bind(node, f"{attention_fqn}.{edge}")
            if (
                attention_fqn is not None
                and node.target == torch.ops.aten.mul.Tensor
                and any(
                    isinstance(argument, (float, int))
                    for argument in node.args
                )
            ):
                value = node.meta.get("val")
                shape = getattr(value, "shape", None)
                if (
                    shape is None
                    or (
                        len(shape) == 4
                        and shape[-2] == shape[-1]
                    )
                ):
                    bind(node, f"{attention_fqn}.score")
            if (
                node.target
                == torch.ops.mdc_llm_deploy.moe_expert.default
            ):
                owner = ownership.moe_by_node.get(node)
                if owner is not None and node.args:
                    bind(node.args[0], f"{owner}.expert_weights")
        return cls(
            targets_by_node=MappingProxyType(
                {
                    node: tuple(targets)
                    for node, targets in targets_by_node.items()
                }
            ),
            nodes_by_target=MappingProxyType(
                {
                    target: tuple(nodes)
                    for target, nodes in nodes_by_target.items()
                }
            ),
        )


class CalibrationSamples(Mapping[str, Tensor]):
    """Aggregated boundary samples with a backward-compatible FQN view."""

    def __init__(
        self,
        by_node: Mapping[Node, Tensor],
        nodes_by_target: Mapping[str, tuple[Node, ...]],
    ) -> None:
        self._by_node = MappingProxyType(dict(by_node))
        self._nodes_by_target = MappingProxyType(dict(nodes_by_target))
        self._target_cache: dict[str, Tensor] = {}

    def __iter__(self) -> Iterator[str]:
        return (
            target
            for target, nodes in self._nodes_by_target.items()
            if nodes
        )

    def __len__(self) -> int:
        return sum(bool(nodes) for nodes in self._nodes_by_target.values())

    def __getitem__(self, target: str) -> Tensor:
        nodes = self._nodes_by_target[target]
        if not nodes:
            raise KeyError(target)
        cached = self._target_cache.get(target)
        if cached is not None:
            return cached
        values = tuple(self._by_node[node] for node in nodes)
        value = values[0] if len(values) == 1 else torch.cat(values)
        self._target_cache[target] = value
        return value

    def sample_items(self, target: str) -> tuple[tuple[Node, Tensor], ...]:
        """Return stable boundary identities and samples for one target."""
        try:
            nodes = self._nodes_by_target[target]
            return tuple((node, self._by_node[node]) for node in nodes)
        except KeyError as error:
            raise KeyError(target) from error

    def boundary_key(self, target: str) -> tuple[Node, ...]:
        """Return graph boundary identity used for qparam caching."""
        try:
            nodes = self._nodes_by_target[target]
        except KeyError as error:
            raise KeyError(target) from error
        if not nodes:
            raise KeyError(target)
        return nodes


class _CalibrationInterpreter(Interpreter):
    """Capture each actual FX tensor boundary once per graph execution."""

    def __init__(
        self,
        graph: GraphModule,
        boundaries: _CalibrationBoundaryMap,
    ) -> None:
        super().__init__(graph, garbage_collect_values=True)
        self.boundaries = boundaries
        self.samples: dict[Node, Tensor] = {}

    def _record(self, node: Node, value: Any) -> None:
        if node not in self.boundaries.targets_by_node:
            return
        if isinstance(value, Tensor) and value.is_floating_point():
            self.samples[node] = value.detach()

    def run_node(self, node: Node) -> Any:
        """Execute one node and retain only quantization boundary tensors."""
        result = super().run_node(node)
        self._record(node, result)
        return result


def collect_calibration_samples(
    graph: GraphModule,
    dataloader: Iterable[Mapping[str, Tensor]],
    plan: CalibrationPlan,
) -> CalibrationSamples:
    """Validate calibration batches and collect quantization-boundary samples."""
    graph_metadata = metadata(graph)
    expected = tuple(item.name for item in graph_metadata.input_abi)
    expected_abi = {
        item.name: item for item in graph_metadata.input_abi
    }
    attention_fqns = tuple(
        boundary.fqn
        for boundary in graph_metadata.boundaries
        if boundary.kind == "attention"
    )
    moe_fqns = tuple(
        boundary.fqn
        for boundary in graph_metadata.boundaries
        if boundary.kind == "moe"
    )
    captured: dict[Node, list[Tensor]] = {}
    boundaries: _CalibrationBoundaryMap | None = None
    observed = 0
    for batch in dataloader:
        if not isinstance(batch, Mapping):
            raise TypeError("calibration batches must be mappings")
        if set(batch) != set(expected):
            raise QuantizationConfigError(
                f"Calibration keys must be {expected}, got {tuple(batch)}"
            )
        for name, tensor in batch.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Calibration value {name!r} must be a tensor")
            abi = expected_abi[name]
            expected_shape = abi.shape
            if tuple(tensor.shape) != expected_shape:
                raise QuantizationConfigError(
                    f"Calibration shape for {name!r} must be {expected_shape}"
                )
            dtype_name = str(tensor.dtype).removeprefix("torch.")
            if dtype_name != abi.dtype:
                raise QuantizationConfigError(
                    f"Calibration dtype for {name!r} must be {abi.dtype}"
                )
            if tensor.device.type not in {"cpu", "cuda", "npu"}:
                raise QuantizationConfigError(
                    f"Calibration device for {name!r} is unsupported: "
                    f"{tensor.device}"
                )
            if not torch.isfinite(tensor).all():
                raise QuantizationConfigError(
                    f"Calibration value {name!r} contains NaN or Inf"
                )
        try:
            if boundaries is None:
                ownership = _CalibrationOwnershipSnapshot.capture(
                    graph,
                    attention_fqns,
                    moe_fqns,
                )
                boundaries = _CalibrationBoundaryMap.capture(
                    graph,
                    plan.required_fqns,
                    ownership,
                )
            recorder = _CalibrationInterpreter(
                graph,
                boundaries,
            )
            with torch.inference_mode():
                recorder.run(*(batch[name] for name in expected))
        except Exception as error:
            raise QuantizationConfigError(
                f"Calibration graph execution failed: {error}"
            ) from error
        for node, value in recorder.samples.items():
            captured.setdefault(node, []).append(value)
        observed += 1
    if observed == 0:
        raise QuantizationConfigError(
            "Calibration dataloader must yield at least one batch"
        )
    if boundaries is None:
        raise RuntimeError("Calibration boundaries were not captured")
    try:
        aggregated: dict[Node, Tensor] = {}
        for node, values in captured.items():
            devices = {value.device for value in values}
            if len(devices) != 1:
                targets = boundaries.targets_by_node[node]
                raise QuantizationConfigError(
                    f"Calibration samples for {targets!r} span devices "
                    f"{sorted(str(device) for device in devices)}"
                )
            aggregated[node] = torch.cat(
                tuple(value.reshape(-1, value.shape[-1]) for value in values)
            )
        return CalibrationSamples(aggregated, boundaries.nodes_by_target)
    except QuantizationConfigError:
        raise
    except Exception as error:
        raise QuantizationConfigError(
            f"Calibration sample aggregation failed: {error}"
        ) from error
