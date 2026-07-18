"""Calibration sample collection at FX quantization boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sized
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor
from torch.fx import GraphModule, Interpreter, Node

from ..errors import QuantizationConfigError
from ..graph.fx.inspection import linear_weight_name
from ..graph.fx.ownership import NodeOwnershipIndex
from ..graph.lifecycle import metadata
from ..observability import StageProgress, StageReporter, get_logger
from .algorithms.math import _qparams_from_extrema
from .config import ActivationSpec
from .planning import CalibrationPlan, CalibrationRequirement


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


@dataclass(slots=True)
class _BoundaryAccumulator:
    """Collect only artifacts required at one physical FX boundary."""

    activation_specs: frozenset[ActivationSpec]
    full_samples: bool
    devices: set[torch.device] = field(default_factory=set)
    samples: list[Tensor] = field(default_factory=list)
    tensor_minimum: Tensor | None = None
    tensor_maximum: Tensor | None = None
    token_minima: list[Tensor] = field(default_factory=list)
    token_maxima: list[Tensor] = field(default_factory=list)

    def observe(self, value: Tensor) -> None:
        """Merge one batch observation into bounded or full-sample state."""
        detached = value.detach()
        self.devices.add(detached.device)
        if self.full_samples:
            self.samples.append(detached)
        source = detached.float().reshape(-1, detached.shape[-1])
        granularities = {spec.granularity for spec in self.activation_specs}
        if "per_tensor" in granularities:
            minimum = source.amin()
            maximum = source.amax()
            self.tensor_minimum = (
                minimum
                if self.tensor_minimum is None
                else torch.minimum(self.tensor_minimum, minimum)
            )
            self.tensor_maximum = (
                maximum
                if self.tensor_maximum is None
                else torch.maximum(self.tensor_maximum, maximum)
            )
        if "per_token" in granularities:
            self.token_minima.append(source.amin(dim=1, keepdim=True))
            self.token_maxima.append(source.amax(dim=1, keepdim=True))

    def extrema(self, granularity: str) -> tuple[Tensor, Tensor]:
        """Finalize extrema for one activation granularity."""
        if granularity == "per_tensor":
            if self.tensor_minimum is None or self.tensor_maximum is None:
                raise KeyError(granularity)
            return self.tensor_minimum, self.tensor_maximum
        if not self.token_minima or not self.token_maxima:
            raise KeyError(granularity)
        return torch.cat(self.token_minima), torch.cat(self.token_maxima)

    def full_sample(self) -> Tensor:
        """Finalize the complete row-major activation matrix."""
        if not self.full_samples or not self.samples:
            raise KeyError("full_samples")
        return torch.cat(
            tuple(
                value.reshape(-1, value.shape[-1])
                for value in self.samples
            )
        )


class CalibrationArtifacts:
    """Typed calibration outputs consumed by materialization."""

    @classmethod
    def empty(cls) -> CalibrationArtifacts:
        """Create calibration artifacts with no planned outputs."""
        return cls({}, {}, {}, {})

    def __init__(
        self,
        qparams: Mapping[tuple[str, ActivationSpec], tuple[Tensor, Tensor]],
        samples_by_node: Mapping[Node, Tensor],
        nodes_by_target: Mapping[str, tuple[Node, ...]],
        requirements: Mapping[str, CalibrationRequirement],
    ) -> None:
        self._qparams = MappingProxyType(dict(qparams))
        self._samples_by_node = MappingProxyType(dict(samples_by_node))
        self._nodes_by_target = MappingProxyType(dict(nodes_by_target))
        self._requirements = MappingProxyType(dict(requirements))
        self._sample_cache: dict[str, Tensor] = {}

    def qparams(
        self,
        fqn: str,
        spec: ActivationSpec,
    ) -> tuple[Tensor, Tensor]:
        """Return finalized activation qparams for one logical target."""
        try:
            return self._qparams[(fqn, spec)]
        except KeyError as error:
            raise KeyError(fqn) from error

    def samples(self, fqn: str) -> Tensor:
        """Return full samples only when explicitly planned."""
        requirement = self._requirements.get(fqn)
        if requirement is None or not requirement.full_samples:
            raise KeyError(fqn)
        cached = self._sample_cache.get(fqn)
        if cached is not None:
            return cached
        items = self.sample_items(fqn)
        if not items:
            raise KeyError(fqn)
        values = tuple(value for _, value in items)
        result = values[0] if len(values) == 1 else torch.cat(values)
        self._sample_cache[fqn] = result
        return result

    def sample_items(self, fqn: str) -> tuple[tuple[object, Tensor], ...]:
        """Return stable physical boundaries and their full samples."""
        requirement = self._requirements.get(fqn)
        if requirement is None or not requirement.full_samples:
            raise KeyError(fqn)
        try:
            nodes = self._nodes_by_target[fqn]
            return tuple(
                (node, self._samples_by_node[node])
                for node in nodes
            )
        except KeyError as error:
            raise KeyError(fqn) from error


def _boundary_accumulators(
    boundaries: _CalibrationBoundaryMap,
    plan: CalibrationPlan,
) -> dict[Node, _BoundaryAccumulator]:
    result: dict[Node, _BoundaryAccumulator] = {}
    for node, targets in boundaries.targets_by_node.items():
        specs: set[ActivationSpec] = set()
        full_samples = False
        for target in targets:
            requirement = plan.requirements[target]
            specs.update(requirement.activation_specs)
            full_samples = full_samples or requirement.full_samples
        result[node] = _BoundaryAccumulator(
            activation_specs=frozenset(specs),
            full_samples=full_samples,
        )
    return result


def _finalize_artifacts(
    boundaries: _CalibrationBoundaryMap,
    accumulators: Mapping[Node, _BoundaryAccumulator],
    plan: CalibrationPlan,
) -> CalibrationArtifacts:
    for node, accumulator in accumulators.items():
        if len(accumulator.devices) > 1:
            targets = boundaries.targets_by_node[node]
            raise QuantizationConfigError(
                f"Calibration samples for {targets!r} span devices "
                f"{sorted(str(device) for device in accumulator.devices)}"
            )
    samples_by_node = {
        node: accumulator.full_sample()
        for node, accumulator in accumulators.items()
        if accumulator.full_samples and accumulator.samples
    }
    qparams: dict[
        tuple[str, ActivationSpec],
        tuple[Tensor, Tensor],
    ] = {}
    shared: dict[
        tuple[tuple[Node, ...], ActivationSpec],
        tuple[Tensor, Tensor],
    ] = {}
    for fqn, requirement in plan.requirements.items():
        nodes = boundaries.nodes_by_target[fqn]
        if not nodes:
            continue
        devices = {
            device
            for node in nodes
            for device in accumulators[node].devices
        }
        if len(devices) > 1:
            raise QuantizationConfigError(
                f"Calibration samples for {fqn!r} span devices "
                f"{sorted(str(device) for device in devices)}"
            )
        for spec in requirement.activation_specs:
            cache_key = (nodes, spec)
            result = shared.get(cache_key)
            if result is None:
                extrema = tuple(
                    accumulators[node].extrema(spec.granularity)
                    for node in nodes
                )
                if spec.granularity == "per_tensor":
                    minimum = extrema[0][0]
                    maximum = extrema[0][1]
                    for node_minimum, node_maximum in extrema[1:]:
                        minimum = torch.minimum(minimum, node_minimum)
                        maximum = torch.maximum(maximum, node_maximum)
                else:
                    minimum = torch.cat(
                        tuple(item[0] for item in extrema)
                    )
                    maximum = torch.cat(
                        tuple(item[1] for item in extrema)
                    )
                result = _qparams_from_extrema(
                    minimum,
                    maximum,
                    bits=spec.bits,
                    symmetric=spec.symmetric,
                )
                shared[cache_key] = result
            qparams[(fqn, spec)] = result
    return CalibrationArtifacts(
        qparams,
        samples_by_node,
        boundaries.nodes_by_target,
        plan.requirements,
    )


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


def _collect_calibration_artifacts(
    graph: GraphModule,
    dataloader: Iterable[Mapping[str, Tensor]],
    plan: CalibrationPlan,
    progress: StageProgress,
) -> tuple[CalibrationArtifacts, int]:
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
    boundaries: _CalibrationBoundaryMap | None = None
    accumulators: dict[Node, _BoundaryAccumulator] | None = None
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
                accumulators = _boundary_accumulators(boundaries, plan)
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
        if accumulators is None:
            raise RuntimeError("Calibration accumulators were not captured")
        for node, value in recorder.samples.items():
            accumulators[node].observe(value)
        observed += 1
        progress.advance()
    if observed == 0:
        raise QuantizationConfigError(
            "Calibration dataloader must yield at least one batch"
        )
    if boundaries is None or accumulators is None:
        raise RuntimeError("Calibration boundaries were not captured")
    try:
        return _finalize_artifacts(boundaries, accumulators, plan), observed
    except QuantizationConfigError:
        raise
    except ValueError:
        raise
    except Exception as error:
        raise QuantizationConfigError(
            f"Calibration artifact finalization failed: {error}"
        ) from error


def _batch_total(
    dataloader: Iterable[Mapping[str, Tensor]],
) -> int | None:
    """Read a declared length without probing or consuming iterable values."""
    if not isinstance(dataloader, Sized):
        return None
    try:
        return len(dataloader)
    except Exception:
        return None


def collect_calibration_artifacts(
    graph: GraphModule,
    dataloader: Iterable[Mapping[str, Tensor]],
    plan: CalibrationPlan,
) -> CalibrationArtifacts:
    """Validate batches and collect planned calibration artifacts."""
    logger = get_logger("quantization.calibration")
    with StageReporter("Quantization calibration") as reporter:
        boundary_count = len(plan.requirements)
        reporter.update(calibration_boundary_count=boundary_count)
        if not plan.requires_collection:
            logger.info("Quantization calibration skipped: no required artifacts")
            reporter.update(outcome="skipped", batch_count=0)
            return CalibrationArtifacts.empty()
        for fqn, requirement in plan.requirements.items():
            logger.debug(
                "Planned calibration boundary: fqn=%s activation_specs=%d "
                "full_samples=%s",
                fqn,
                len(requirement.activation_specs),
                requirement.full_samples,
            )
        total = _batch_total(dataloader)
        reporter.update(batch_total="unknown" if total is None else total)
        with reporter.progress(
            "Collecting calibration batches",
            total=total,
        ) as progress:
            artifacts, observed = _collect_calibration_artifacts(
                graph,
                dataloader,
                plan,
                progress,
            )
        reporter.update(outcome="completed", batch_count=observed)
        return artifacts
