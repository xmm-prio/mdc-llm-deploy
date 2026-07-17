"""Bind FX linear parameters to ONNX initializers by exact tensor identity."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper
from torch.fx import GraphModule

from ...errors import OnnxExportError
from ...graph.fx.inspection import linear_weight_name

_CHUNK_BYTES = 1024 * 1024
_ONNX_DTYPE_NAMES = {
    int(TensorProto.FLOAT16): "float16",
    int(TensorProto.FLOAT): "float32",
    int(TensorProto.BFLOAT16): "bfloat16",
}
_ONNX_BIT_DTYPES: dict[int, np.dtype[Any]] = {
    int(TensorProto.FLOAT16): np.dtype(np.uint16),
    int(TensorProto.FLOAT): np.dtype(np.uint32),
    int(TensorProto.BFLOAT16): np.dtype(np.uint16),
}
_TORCH_DTYPES = {
    torch.float16: ("float16", torch.uint16),
    torch.float32: ("float32", torch.uint32),
    torch.bfloat16: ("bfloat16", torch.uint16),
}


@dataclass(frozen=True)
class _TensorIdentity:
    shape: tuple[int, int]
    dtype: str
    digest: bytes


@dataclass(frozen=True)
class _FxWeight:
    fqn: str
    tensor: torch.Tensor
    call_count: int

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.tensor.shape[0]), int(self.tensor.shape[1]))

    @property
    def dtype(self) -> str:
        return _TORCH_DTYPES[self.tensor.dtype][0]

    def chunks(self) -> Iterator[bytes]:
        _, bit_dtype = _TORCH_DTYPES[self.tensor.dtype]
        rows_per_chunk = _rows_per_chunk(self.shape[1], self.tensor.element_size())
        value = self.tensor.detach()
        for start in range(0, self.shape[0], rows_per_chunk):
            chunk = value[start : start + rows_per_chunk]
            yield chunk.contiguous().view(bit_dtype).cpu().numpy().tobytes()


@dataclass(frozen=True)
class _OnnxWeight:
    name: str
    initializer: onnx.TensorProto
    transpose: bool

    @property
    def shape(self) -> tuple[int, int]:
        rows, columns = (int(item) for item in self.initializer.dims)
        return (columns, rows) if self.transpose else (rows, columns)

    @property
    def dtype(self) -> str:
        return _ONNX_DTYPE_NAMES[int(self.initializer.data_type)]

    def chunks(self) -> Iterator[bytes]:
        try:
            value = numpy_helper.to_array(self.initializer)
        except Exception as error:
            raise OnnxExportError(
                f"Unsupported ONNX linear weight representation for {self.name!r}: {error}"
            ) from error
        bit_dtype = _ONNX_BIT_DTYPES[int(self.initializer.data_type)]
        bits = value.view(bit_dtype)
        rows_per_chunk = _rows_per_chunk(self.shape[1], bit_dtype.itemsize)
        for start in range(0, self.shape[0], rows_per_chunk):
            if self.transpose:
                chunk = bits[:, start : start + rows_per_chunk].T
            else:
                chunk = bits[start : start + rows_per_chunk]
            yield np.ascontiguousarray(chunk).tobytes()


@dataclass(frozen=True)
class _UnsupportedOnnxWeight:
    name: str
    shape: tuple[int, ...]
    data_type: int
    transpose: bool

    def could_represent(self, weight: _FxWeight) -> bool:
        """Return whether dimensions could encode the requested FX weight."""
        if len(self.shape) == 2:
            rows, columns = self.shape
            canonical_shape = (
                (columns, rows) if self.transpose else (rows, columns)
            )
            return canonical_shape == weight.shape
        element_count = int(np.prod(self.shape, dtype=np.int64))
        return element_count == weight.shape[0] * weight.shape[1]

    def describe(self) -> str:
        """Return stable shape and dtype diagnostics."""
        try:
            dtype = TensorProto.DataType.Name(self.data_type)
        except ValueError:
            dtype = f"UNKNOWN({self.data_type})"
        return f"{self.name!r} (shape={self.shape!r}, dtype={dtype})"


@dataclass(frozen=True)
class _OnnxWeightInventory:
    supported: tuple[_OnnxWeight, ...]
    unsupported: tuple[_UnsupportedOnnxWeight, ...]


def _rows_per_chunk(columns: int, item_size: int) -> int:
    return max(1, _CHUNK_BYTES // max(1, columns * item_size))


def _identity(weight: _FxWeight | _OnnxWeight) -> _TensorIdentity:
    digest = hashlib.sha256()
    for chunk in weight.chunks():
        digest.update(chunk)
    return _TensorIdentity(weight.shape, weight.dtype, digest.digest())


def _equal(left: _FxWeight | _OnnxWeight, right: _FxWeight | _OnnxWeight) -> bool:
    return all(
        left_chunk == right_chunk
        for left_chunk, right_chunk in zip(
            left.chunks(),
            right.chunks(),
            strict=True,
        )
    )


def _fx_weights(graph: GraphModule) -> tuple[_FxWeight, ...]:
    call_counts: dict[str, int] = {}
    for node in graph.graph.nodes:
        name = linear_weight_name(node)
        if name is not None:
            call_counts[name] = call_counts.get(name, 0) + 1

    weights: list[_FxWeight] = []
    for name, call_count in call_counts.items():
        try:
            tensor = graph.get_parameter(name)
        except (AttributeError, TypeError) as error:
            raise OnnxExportError(
                f"Unsupported FX linear weight representation for {name!r}: "
                f"weight is not a parameter (FX call count: {call_count})"
            ) from error
        if tensor.ndim != 2 or tensor.dtype not in _TORCH_DTYPES:
            raise OnnxExportError(
                f"Unsupported FX linear weight representation for {name!r}: "
                f"shape={tuple(tensor.shape)!r}, dtype={tensor.dtype} "
                f"(FX call count: {call_count})"
            )
        weights.append(_FxWeight(name, tensor, call_count))
    return tuple(weights)


def _gemm_transpose(node: onnx.NodeProto) -> bool:
    trans_b = next(
        (
            int(attribute.i)
            for attribute in node.attribute
            if attribute.name == "transB"
        ),
        0,
    )
    if trans_b not in {0, 1}:
        raise OnnxExportError(
            f"Unsupported ONNX linear weight representation for {node.input[1]!r}: "
            f"Gemm transB={trans_b}"
        )
    return trans_b == 0


def _onnx_weights(model: onnx.ModelProto) -> _OnnxWeightInventory:
    initializers: dict[str, onnx.TensorProto] = {}
    duplicate_names: set[str] = set()
    for initializer in model.graph.initializer:
        if initializer.name in initializers:
            duplicate_names.add(initializer.name)
        else:
            initializers[initializer.name] = initializer

    representations: dict[str, bool] = {}
    order: list[str] = []
    unsupported: list[_UnsupportedOnnxWeight] = []
    for node in model.graph.node:
        if node.op_type not in {"Gemm", "MatMul"} or len(node.input) < 2:
            continue
        name = node.input[1]
        initializer = initializers.get(name)
        if initializer is None:
            continue
        if name in duplicate_names:
            raise OnnxExportError(
                f"Ambiguous ONNX linear initializers share name {name!r}"
            )
        transpose = node.op_type == "MatMul" or _gemm_transpose(node)
        if (
            len(initializer.dims) != 2
            or int(initializer.data_type) not in _ONNX_DTYPE_NAMES
        ):
            if not any(candidate.name == name for candidate in unsupported):
                unsupported.append(
                    _UnsupportedOnnxWeight(
                        name=name,
                        shape=tuple(int(item) for item in initializer.dims),
                        data_type=int(initializer.data_type),
                        transpose=transpose,
                    )
                )
            continue
        previous = representations.get(name)
        if previous is not None and previous != transpose:
            raise OnnxExportError(
                f"Unsupported ONNX linear weight representation for {name!r}: "
                "initializer is used with conflicting layouts"
            )
        if previous is None:
            representations[name] = transpose
            order.append(name)
    return _OnnxWeightInventory(
        supported=tuple(
            _OnnxWeight(name, initializers[name], representations[name])
            for name in order
        ),
        unsupported=tuple(unsupported),
    )


def _reject_duplicate_fx_identities(
    weights: tuple[_FxWeight, ...],
    identities: dict[str, _TensorIdentity],
) -> None:
    groups: dict[_TensorIdentity, list[_FxWeight]] = {}
    for weight in weights:
        groups.setdefault(identities[weight.fqn], []).append(weight)
    for candidates in groups.values():
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                if _equal(left, right):
                    raise OnnxExportError(
                        "Ambiguous FX linear parameters have identical tensor "
                        f"identity: {left.fqn!r} (FX call count: {left.call_count}), "
                        f"{right.fqn!r} (FX call count: {right.call_count})"
                    )


def _rename_plan(
    model: onnx.ModelProto,
    fx_weights: tuple[_FxWeight, ...],
    onnx_weights: _OnnxWeightInventory,
) -> dict[str, str]:
    fx_identities = {weight.fqn: _identity(weight) for weight in fx_weights}
    _reject_duplicate_fx_identities(fx_weights, fx_identities)

    onnx_by_identity: dict[_TensorIdentity, list[_OnnxWeight]] = {}
    for weight in onnx_weights.supported:
        onnx_by_identity.setdefault(_identity(weight), []).append(weight)

    renames: dict[str, str] = {}
    for fx_weight in fx_weights:
        candidates = [
            candidate
            for candidate in onnx_by_identity.get(fx_identities[fx_weight.fqn], ())
            if _equal(fx_weight, candidate)
        ]
        if not candidates:
            unsupported = [
                candidate
                for candidate in onnx_weights.unsupported
                if candidate.could_represent(fx_weight)
            ]
            if unsupported:
                details = ", ".join(
                    candidate.describe() for candidate in unsupported
                )
                raise OnnxExportError(
                    "Unsupported ONNX linear weight representation while binding "
                    f"FX linear parameter {fx_weight.fqn!r} "
                    f"(FX call count: {fx_weight.call_count}): {details}"
                )
            raise OnnxExportError(
                f"Missing ONNX initializer for FX linear parameter "
                f"{fx_weight.fqn!r} (FX call count: {fx_weight.call_count})"
            )
        if len(candidates) > 1:
            names = ", ".join(repr(candidate.name) for candidate in candidates)
            raise OnnxExportError(
                f"Ambiguous ONNX initializers for FX linear parameter "
                f"{fx_weight.fqn!r} (FX call count: {fx_weight.call_count}): "
                f"{names}"
            )
        renames[candidates[0].name] = f"graph.{fx_weight.fqn}"
    _validate_renames(model, renames)
    return renames


def _validate_renames(model: onnx.ModelProto, renames: dict[str, str]) -> None:
    old_names = set(renames)
    occupied = {
        initializer.name
        for initializer in model.graph.initializer
        if initializer.name not in old_names
    }
    occupied.update(
        output
        for node in model.graph.node
        for output in node.output
        if output and output not in old_names
    )
    occupied.update(
        value.name
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
        if value.name and value.name not in old_names
    )
    conflicts = sorted(set(renames.values()).intersection(occupied))
    if conflicts:
        raise OnnxExportError(
            f"Cannot rename ONNX linear initializers; target names are occupied: "
            f"{', '.join(repr(name) for name in conflicts)}"
        )


def _apply_renames(model: onnx.ModelProto, renames: dict[str, str]) -> None:
    for initializer in model.graph.initializer:
        replacement = renames.get(initializer.name)
        if replacement is not None:
            initializer.name = replacement
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            replacement = renames.get(name)
            if replacement is not None:
                node.input[index] = replacement
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        replacement = renames.get(value.name)
        if replacement is not None:
            value.name = replacement


def canonicalize_linear_initializers(
    model: onnx.ModelProto,
    graph: GraphModule,
) -> None:
    """Rename standard ONNX linear weights using exact FX parameter identity."""
    try:
        fx_weights = _fx_weights(graph)
        if not fx_weights:
            return
        renames = _rename_plan(model, fx_weights, _onnx_weights(model))
        _apply_renames(model, renames)
    except OnnxExportError:
        raise
    except Exception as error:
        raise OnnxExportError(
            f"Unsupported linear weight identity representation: {error}"
        ) from error
