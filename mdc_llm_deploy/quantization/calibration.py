"""Calibration sample collection at FX quantization boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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


class _CalibrationInterpreter(Interpreter):
    """Capture actual linear inputs and attention edges from one FX execution."""

    def __init__(
        self,
        graph: GraphModule,
        required_fqns: frozenset[str],
        ownership: _CalibrationOwnershipSnapshot,
    ) -> None:
        super().__init__(graph, garbage_collect_values=True)
        self.required_fqns = required_fqns
        self.ownership = ownership
        self.samples: dict[str, list[Tensor]] = {}

    def _record(self, name: str, value: Any) -> None:
        if name not in self.required_fqns:
            return
        if isinstance(value, Tensor) and value.is_floating_point():
            self.samples.setdefault(name, []).append(value.detach())

    def run_node(self, node: Node) -> Any:
        """Execute one node and retain only quantization boundary tensors."""
        args, _ = self.fetch_args_kwargs_from_env(node)
        result = super().run_node(node)
        if node.op != "call_function":
            return result
        attention_fqn = self.ownership.attention_by_node.get(node)
        weight_name = linear_weight_name(node)
        if weight_name is not None:
            fqn = weight_name.removesuffix(".weight")
            self._record(fqn, args[0])
            edge = {
                "q_proj": "query",
                "k_proj": "key",
                "v_proj": "value",
            }.get(fqn.rsplit(".", 1)[-1])
            if edge is not None and attention_fqn is not None:
                self._record(f"{attention_fqn}.{edge}", result)
        if (
            attention_fqn is not None
            and node.target == torch.ops.aten.mul.Tensor
            and isinstance(result, Tensor)
            and result.ndim == 4
            and result.shape[-2] == result.shape[-1]
            and any(isinstance(argument, (float, int)) for argument in args)
        ):
            self._record(f"{attention_fqn}.score", result)
        if (
            node.target
            == torch.ops.mdc_llm_deploy.moe_expert.default
        ):
            owner = self.ownership.moe_by_node.get(node)
            if owner is not None:
                self._record(f"{owner}.expert_weights", args[0])
        return result


def collect_calibration_samples(
    graph: GraphModule,
    dataloader: Iterable[Mapping[str, Tensor]],
    plan: CalibrationPlan,
) -> dict[str, Tensor]:
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
    captured: dict[str, list[Tensor]] = {}
    ownership: _CalibrationOwnershipSnapshot | None = None
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
            if ownership is None:
                ownership = _CalibrationOwnershipSnapshot.capture(
                    graph,
                    attention_fqns,
                    moe_fqns,
                )
            recorder = _CalibrationInterpreter(
                graph,
                plan.required_fqns,
                ownership,
            )
            with torch.inference_mode():
                recorder.run(*(batch[name] for name in expected))
        except Exception as error:
            raise QuantizationConfigError(
                f"Calibration graph execution failed: {error}"
            ) from error
        for fqn, values in recorder.samples.items():
            captured.setdefault(fqn, []).extend(values)
        observed += 1
    if observed == 0:
        raise QuantizationConfigError(
            "Calibration dataloader must yield at least one batch"
        )
    try:
        aggregated: dict[str, Tensor] = {}
        for fqn, values in captured.items():
            devices = {value.device for value in values}
            if len(devices) != 1:
                raise QuantizationConfigError(
                    f"Calibration samples for {fqn!r} span devices "
                    f"{sorted(str(device) for device in devices)}"
                )
            aggregated[fqn] = torch.cat(
                tuple(value.reshape(-1, value.shape[-1]) for value in values)
            )
        return aggregated
    except QuantizationConfigError:
        raise
    except Exception as error:
        raise QuantizationConfigError(
            f"Calibration sample aggregation failed: {error}"
        ) from error
