"""MinMax lifecycle implementation for dense linear modules."""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..base import CalibrationBatch, QuantizationState, Quantizer, run_calibration
from .config import MinMaxConfig
from .linear import MinMaxLinear
from .observer import MinMaxObserver
from .qparams import _QParamBinding, _QParamSpec


@dataclass(slots=True)
class _TargetBuilder:
    module: nn.Linear
    names: list[str]
    selected: list[bool]


@dataclass(slots=True)
class _PreparedTarget:
    module: nn.Linear
    names: tuple[str, ...]
    weight_binding: _QParamBinding | None
    activation_observer: MinMaxObserver | None


@dataclass(frozen=True, slots=True)
class _Replacement:
    parent: nn.Module
    attribute: str
    original: nn.Linear
    replacement: MinMaxLinear


class MinMaxQuantizer(Quantizer[MinMaxConfig]):
    """Apply configured MinMax INT8 fake quantization to dense Linear modules."""

    def __init__(self, config: MinMaxConfig) -> None:
        super().__init__(config)
        self._weight_spec, self._activation_spec = _QParamSpec.from_config(config)
        self._model_reference: weakref.ReferenceType[nn.Module] | None = None
        self._targets: tuple[_PreparedTarget, ...] = ()

    def _prepare(self, model: nn.Module) -> None:
        self._discover_targets(model, collect_weight=True)

    def _discover_targets(self, model: nn.Module, *, collect_weight: bool) -> None:
        grouped: dict[int, _TargetBuilder] = {}
        for name, module in model.named_modules(remove_duplicate=False):
            if not isinstance(module, nn.Linear):
                continue
            builder = grouped.setdefault(id(module), _TargetBuilder(module, [], []))
            builder.names.append(name)
            builder.selected.append(self.config.targets.matches(name))

        prepared: list[_PreparedTarget] = []
        for builder in grouped.values():
            if any(builder.selected) and not all(builder.selected):
                aliases = ", ".join(repr(name) for name in builder.names)
                raise ValueError(f"shared Linear aliases must have consistent selection: {aliases}")
            if not all(builder.selected):
                continue
            if "" in builder.names:
                raise ValueError("a root Linear cannot be replaced in place")
            weight_binding = None
            if collect_weight:
                weight_observer = self._weight_spec.create_observer()
                if weight_observer is not None:
                    weight_observer.observe(builder.module.weight)
                    weight_binding = self._weight_spec.freeze(
                        weight_observer,
                        builder.module,
                    )
            prepared.append(
                _PreparedTarget(
                    module=builder.module,
                    names=tuple(builder.names),
                    weight_binding=weight_binding,
                    activation_observer=self._activation_spec.create_observer(),
                )
            )

        if not prepared:
            raise ValueError("target selector did not match any Linear modules")
        self._model_reference = weakref.ref(model)
        self._targets = tuple(prepared)

    def _calibrate(
        self,
        model: nn.Module,
        batches: Iterable[CalibrationBatch],
    ) -> None:
        self._validate_model(model)
        self._validate_structure(model)
        handles: list[torch.utils.hooks.RemovableHandle] = []
        try:
            for target in self._targets:
                if target.activation_observer is None:
                    continue
                observer = target.activation_observer

                def observe_input(
                    _module: nn.Module,
                    args: tuple[object, ...],
                    *,
                    current_observer: MinMaxObserver = observer,
                ) -> None:
                    if not args or not isinstance(args[0], Tensor):
                        raise TypeError("Linear calibration input must be a Tensor")
                    inputs = args[0]
                    if self._activation_spec.axis is not None and inputs.ndim < 2:
                        raise ValueError("per-token activation input must have rank at least two")
                    current_observer.observe(inputs)

                handles.append(target.module.register_forward_pre_hook(observe_input))
            run_calibration(model, batches)
        finally:
            for handle in handles:
                handle.remove()

    def _convert(self, model: nn.Module) -> None:
        self._validate_model(model)
        locations = self._validate_structure(model)
        wrappers = self._build_calibrated_wrappers()
        self._replace(locations, wrappers)

    def _build_calibrated_wrappers(self) -> dict[int, MinMaxLinear]:
        missing = [
            target.names[0]
            for target in self._targets
            if target.activation_observer is not None and not target.activation_observer.observed
        ]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise RuntimeError(f"activation calibration did not cover target Linear modules: {names}")

        wrappers = {
            id(target.module): self._make_wrapper(
                target,
                weight_binding=target.weight_binding,
                activation_binding=(
                    None
                    if target.activation_observer is None
                    else self._activation_spec.freeze(
                        target.activation_observer,
                        target.module,
                    )
                ),
            )
            for target in self._targets
        }
        return wrappers

    def _make_wrapper(
        self,
        target: _PreparedTarget,
        *,
        weight_binding: _QParamBinding | None,
        activation_binding: _QParamBinding | None,
    ) -> MinMaxLinear:
        return MinMaxLinear(
            target.module,
            weight_qparams=(
                None if weight_binding is None else weight_binding.parameters
            ),
            activation_qparams=(
                None if activation_binding is None else activation_binding.parameters
            ),
            weight_axis=None if weight_binding is None else weight_binding.axis,
            activation_axis=(
                None if activation_binding is None else activation_binding.axis
            ),
        )

    def _replace(
        self,
        locations: tuple[tuple[nn.Module, str, nn.Linear], ...],
        wrappers: Mapping[int, MinMaxLinear],
    ) -> None:
        replacements = tuple(
            _Replacement(
                parent=parent,
                attribute=attribute,
                original=original,
                replacement=wrappers[id(original)],
            )
            for parent, attribute, original in locations
        )

        applied: list[_Replacement] = []
        try:
            for replacement in replacements:
                current = getattr(replacement.parent, replacement.attribute)
                if current is replacement.replacement:
                    continue
                if current is not replacement.original:
                    raise RuntimeError("model structure changed during conversion")
                setattr(replacement.parent, replacement.attribute, replacement.replacement)
                applied.append(replacement)
        except Exception:
            for replacement in reversed(applied):
                setattr(replacement.parent, replacement.attribute, replacement.original)
            raise

    def restore(self, model: nn.Module, state_dict: Mapping[str, Tensor]) -> nn.Module:
        """Rebuild converted wrappers from frozen checkpoint qparams."""
        self._require_state(QuantizationState.UNPREPARED, "restore")
        self._discover_targets(model, collect_weight=False)
        locations = self._validate_structure(model)
        self._validate_checkpoint_keys(model, state_dict)
        wrappers: dict[int, MinMaxLinear] = {}
        for target in self._targets:
            weight_binding = self._weight_spec.from_checkpoint(
                state_dict,
                target.names,
                target.module,
            )
            activation_binding = self._activation_spec.from_checkpoint(
                state_dict,
                target.names,
                target.module,
            )
            wrappers[id(target.module)] = self._make_wrapper(
                target,
                weight_binding=weight_binding,
                activation_binding=activation_binding,
            )
        self._replace(locations, wrappers)
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            for parent, attribute, original in locations:
                setattr(parent, attribute, original)
            raise
        self._state = QuantizationState.CONVERTED
        return model

    def _validate_checkpoint_keys(
        self,
        model: nn.Module,
        state_dict: Mapping[str, Tensor],
    ) -> None:
        expected = set(model.state_dict())
        for target in self._targets:
            for name in target.names:
                expected.update(self._weight_spec.expected_checkpoint_keys(name))
                expected.update(self._activation_spec.expected_checkpoint_keys(name))
        actual = set(state_dict)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise RuntimeError(
                f"quantized state_dict keys mismatch; missing={missing}, unexpected={unexpected}"
            )

    def _validate_model(self, model: nn.Module) -> None:
        prepared_model = None if self._model_reference is None else self._model_reference()
        if prepared_model is not model:
            raise ValueError("lifecycle operations must use the model passed to prepare")

    def _validate_structure(
        self,
        model: nn.Module,
    ) -> tuple[tuple[nn.Module, str, nn.Linear], ...]:
        locations: dict[tuple[int, str], tuple[nn.Module, str, nn.Linear]] = {}
        for target in self._targets:
            for name in target.names:
                parent_name, _, attribute = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                current = getattr(parent, attribute, None)
                if current is not target.module:
                    raise RuntimeError(f"target module {name!r} changed after prepare")
                locations.setdefault((id(parent), attribute), (parent, attribute, target.module))
        return tuple(locations.values())


__all__ = ["MinMaxQuantizer"]
