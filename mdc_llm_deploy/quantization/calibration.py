"""Calibration sample collection at FX quantization boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor
from torch.fx import GraphModule, Interpreter, Node

from ..errors import QuantizationConfigError
from ..fx_inspection import linear_weight_name
from ..fx_ownership import node_belongs_to
from ..graph import metadata


class _CalibrationInterpreter(Interpreter):
    """Capture actual linear inputs and attention edges from one FX execution."""

    def __init__(
        self,
        graph: GraphModule,
        attention_fqns: tuple[str, ...],
        moe_fqns: tuple[str, ...] = (),
    ) -> None:
        super().__init__(graph, garbage_collect_values=True)
        self.attention_fqns = attention_fqns
        self.moe_fqns = moe_fqns
        self.samples: dict[str, list[Tensor]] = {}

    def _record(self, name: str, value: Any) -> None:
        if isinstance(value, Tensor) and value.is_floating_point():
            self.samples.setdefault(name, []).append(value.detach().cpu())

    def _attention_fqn(self, node: Node) -> str | None:
        return next(
            (
                fqn
                for fqn in self.attention_fqns
                if node_belongs_to(node, fqn)
            ),
            None,
        )

    def run_node(self, node: Node) -> Any:
        """Execute one node and retain only quantization boundary tensors."""
        args, _ = self.fetch_args_kwargs_from_env(node)
        result = super().run_node(node)
        if node.op != "call_function":
            return result
        weight_name = linear_weight_name(node)
        if weight_name is not None:
            fqn = weight_name.removesuffix(".weight")
            self._record(fqn, args[0])
            edge = {
                "q_proj": "query",
                "k_proj": "key",
                "v_proj": "value",
            }.get(fqn.rsplit(".", 1)[-1])
            attention_fqn = self._attention_fqn(node)
            if edge is not None and attention_fqn is not None:
                self._record(f"{attention_fqn}.{edge}", result)
        attention_fqn = self._attention_fqn(node)
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
            and self.moe_fqns
        ):
            owner = next(
                (
                    fqn
                    for fqn in self.moe_fqns
                    if node_belongs_to(node, fqn)
                ),
                self.moe_fqns[0] if len(self.moe_fqns) == 1 else None,
            )
            if owner is not None:
                self._record(f"{owner}.expert_weights", args[0])
        return result


def collect_calibration_samples(
    graph: GraphModule,
    dataloader: Iterable[Mapping[str, Tensor]],
) -> dict[str, Tensor]:
    """Validate calibration batches and collect quantization-boundary samples."""
    graph_metadata = metadata(graph)
    expected = tuple(item.name for item in graph_metadata.input_abi)
    expected_abi = {
        item.name: item for item in graph_metadata.input_abi
    }
    graph_devices = {
        tensor.device
        for tensor in (*tuple(graph.parameters()), *tuple(graph.buffers()))
    }
    if len(graph_devices) > 1:
        raise QuantizationConfigError(
            "Calibration graph parameters and buffers must use one device"
        )
    expected_device = next(iter(graph_devices), None)
    observed_device = expected_device
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
            if observed_device is None:
                observed_device = tensor.device
            if tensor.device != observed_device:
                raise QuantizationConfigError(
                    f"Calibration device for {name!r} must be {observed_device}"
                )
            if not torch.isfinite(tensor).all():
                raise QuantizationConfigError(
                    f"Calibration value {name!r} contains NaN or Inf"
                )
        recorder = _CalibrationInterpreter(
            graph,
            attention_fqns,
            moe_fqns,
        )
        try:
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
        return {
            fqn: torch.cat(
                tuple(value.reshape(-1, value.shape[-1]) for value in values)
            )
            for fqn, values in captured.items()
        }
    except Exception as error:
        raise QuantizationConfigError(
            f"Calibration sample aggregation failed: {error}"
        ) from error
