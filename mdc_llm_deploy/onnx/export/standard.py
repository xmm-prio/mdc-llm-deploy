"""Validated export from an ATen FX graph to standard ONNX."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper, shape_inference
from torch.fx import GraphModule

from ...errors import OnnxExportError
from ...graph.fx.inspection import linear_weight_name
from ...graph.metadata import GraphMetadata
from ...operators.contracts.onnx import MDC_ONNX_OPSET
from ...operators.contracts.schema import OPERATOR_SCHEMAS
from ...placement.inputs import resolve_input_devices
from ..validation.model import validate_mdc_model

_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
_FLOAT_ONNX_DTYPES: set[int] = {
    int(TensorProto.FLOAT16),
    int(TensorProto.FLOAT),
    int(TensorProto.BFLOAT16),
}


def _seed_custom_value_info(model: onnx.ModelProto) -> None:
    """Seed shape propagation across custom operators with identity-shaped output."""
    values = {
        item.name: item
        for item in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        )
    }
    for node in model.graph.node:
        if node.op_type != "MoeExpert" or not node.input or not node.output:
            continue
        source = values.get(node.input[0])
        if source is None or node.output[0] in values:
            continue
        output = onnx.ValueInfoProto()
        output.CopyFrom(source)
        output.name = node.output[0]
        model.graph.value_info.append(output)
        values[output.name] = output


class _PositionalGraph(torch.nn.Module):
    """Adapt torch.export keyword-only call specs to positional ONNX tracing."""

    def __init__(
        self,
        graph: GraphModule,
        names: tuple[str, ...],
        *,
        use_kwargs: bool,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.names = names
        self.use_kwargs = use_kwargs
        self.training = False

    def forward(self, *args: torch.Tensor) -> Any:
        """Call the captured graph with its original input convention."""
        if self.use_kwargs:
            return self.graph(**dict(zip(self.names, args, strict=True)))
        return self.graph(*args)


def _example_arguments(
    metadata: GraphMetadata,
) -> tuple[torch.Tensor, ...]:
    try:
        devices = resolve_input_devices(metadata)
    except ValueError as error:
        raise OnnxExportError(str(error)) from error
    result: list[torch.Tensor] = []
    for item, device in zip(metadata.input_abi, devices, strict=True):
        try:
            dtype = _TORCH_DTYPES[item.dtype]
        except KeyError as error:
            raise OnnxExportError(f"Unsupported input dtype: {item.dtype}") from error
        try:
            result.append(torch.zeros(item.shape, dtype=dtype, device=device))
        except Exception as error:
            raise OnnxExportError(
                f"Cannot create ONNX input {item.name!r} on {device}: {error}"
            ) from error
    return tuple(result)


def _fold_linear_weight_transposes(model: onnx.ModelProto) -> None:
    """Fold only static 2-D linear weight transposes in ONNX protobuf space."""
    initializers = {item.name: item for item in model.graph.initializer}
    producers = {
        output: node for node in model.graph.node for output in node.output
    }
    folded_nodes: list[onnx.NodeProto] = []
    source_names: set[str] = set()
    for node in model.graph.node:
        if node.op_type not in {"Gemm", "MatMul"} or len(node.input) < 2:
            continue
        weight_name = node.input[1]
        transpose = producers.get(weight_name)
        if (
            transpose is None
            or transpose.op_type != "Transpose"
            or len(transpose.input) != 1
            or len(transpose.output) != 1
            or transpose in folded_nodes
        ):
            continue
        source = initializers.get(transpose.input[0])
        if (
            source is None
            or len(source.dims) != 2
            or source.data_type not in _FLOAT_ONNX_DTYPES
        ):
            continue
        permutation = next(
            (
                tuple(attribute.ints)
                for attribute in transpose.attribute
                if attribute.name == "perm"
            ),
            (1, 0),
        )
        if permutation != (1, 0):
            continue
        array = numpy_helper.to_array(source)
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.ascontiguousarray(array.T),
                name=weight_name,
            )
        )
        folded_nodes.append(transpose)
        source_names.add(source.name)

    for node in folded_nodes:
        model.graph.node.remove(node)
    used_inputs = {
        input_name
        for node in model.graph.node
        for input_name in node.input
        if input_name
    }
    retained = [
        item
        for item in model.graph.initializer
        if item.name not in source_names or item.name in used_inputs
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)


def _rename_initializer_references(
    model: onnx.ModelProto,
    renames: list[tuple[onnx.TensorProto, str, str]],
) -> None:
    """Rename initializers and references with a safe single-scan fast path."""
    old_names = [old_name for _, old_name, _ in renames]
    new_names = [new_name for _, _, new_name in renames]
    old_name_set = set(old_names)
    occupied_names = {
        initializer.name
        for initializer in model.graph.initializer
        if initializer.name not in old_name_set
    }
    occupied_names.update(
        value.name
        for value in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        )
    )
    occupied_names.update(
        output_name
        for node in model.graph.node
        for output_name in node.output
        if output_name
    )
    use_fast_path = (
        len(old_name_set) == len(old_names)
        and len(set(new_names)) == len(new_names)
        and all(
            new_name == old_name or new_name not in old_name_set
            for _, old_name, new_name in renames
        )
        and not occupied_names.intersection(new_names)
    )
    if use_fast_path:
        replacements = {
            old_name: new_name for _, old_name, new_name in renames
        }
        for initializer, _, new_name in renames:
            initializer.name = new_name
        for node in model.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name in replacements:
                    node.input[index] = replacements[input_name]
        return

    for initializer, old_name, new_name in renames:
        initializer.name = new_name
        for node in model.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name == old_name:
                    node.input[index] = new_name


def _restore_linear_initializer_names(
    model: onnx.ModelProto,
    graph: GraphModule,
) -> None:
    """Restore FX parameter FQNs lost by the legacy ONNX exporter."""
    _fold_linear_weight_transposes(model)
    parameter_names = [
        weight_name
        for node in graph.graph.nodes
        if (weight_name := linear_weight_name(node)) is not None
    ]
    initializers = {item.name: item for item in model.graph.initializer}
    onnx_weight_names = [
        node.input[1]
        for node in model.graph.node
        if node.op_type in {"Gemm", "MatMul"}
        and len(node.input) >= 2
        and node.input[1] in initializers
        and len(initializers[node.input[1]].dims) == 2
        and initializers[node.input[1]].data_type in _FLOAT_ONNX_DTYPES
        and "embed" not in node.input[1]
    ]
    if len(parameter_names) != len(onnx_weight_names):
        raise OnnxExportError(
            "Cannot map ATen linear parameters to standard ONNX initializers"
        )
    _rename_initializer_references(
        model,
        [
            (
                initializers[old_name],
                old_name,
                f"graph.{parameter_name}",
            )
            for parameter_name, old_name in zip(
                parameter_names,
                onnx_weight_names,
                strict=True,
            )
        ],
    )


def _fold_initializer_alias(
    model: onnx.ModelProto,
    canonical_name: str,
) -> str | None:
    """Fold an Identity-exported parameter value into its initializer."""
    initializers = {item.name: item for item in model.graph.initializer}
    if canonical_name in initializers:
        return None
    producers = {
        output: node for node in model.graph.node for output in node.output
    }
    aliases: set[str] = set()
    source = canonical_name
    identities: list[onnx.NodeProto] = []
    while source not in initializers:
        producer = producers.get(source)
        if (
            producer is None
            or producer.op_type != "Identity"
            or len(producer.input) != 1
            or len(producer.output) != 1
        ):
            raise OnnxExportError(
                f"Parameter {canonical_name!r} is not backed by an initializer"
            )
        identities.append(producer)
        aliases.add(source)
        source = producer.input[0]
    initializer = onnx.TensorProto()
    initializer.CopyFrom(initializers[source])
    initializer.name = canonical_name
    model.graph.initializer.append(initializer)
    identity_ids = {id(node) for node in identities}
    for node in model.graph.node:
        if id(node) in identity_ids:
            continue
        for index, input_name in enumerate(node.input):
            if input_name in aliases:
                node.input[index] = canonical_name
    for identity in identities:
        model.graph.node.remove(identity)
    retained_values = [
        item
        for item in model.graph.value_info
        if item.name not in aliases or item.name == canonical_name
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_values)
    return source


@dataclass(frozen=True)
class _InitializerAliasFold:
    canonical_name: str
    source_initializer: onnx.TensorProto
    aliases: frozenset[str]
    identity_indices: frozenset[int]


@dataclass(frozen=True)
class _InitializerAliasFoldPlan:
    folds: tuple[_InitializerAliasFold, ...]

    @classmethod
    def try_build(
        cls,
        model: onnx.ModelProto,
        canonical_names: tuple[str, ...],
    ) -> _InitializerAliasFoldPlan | None:
        """Build a plan only when all alias folds are independent."""
        if len(set(canonical_names)) != len(canonical_names):
            return None

        initializers: dict[str, onnx.TensorProto] = {}
        for initializer in model.graph.initializer:
            if initializer.name in initializers:
                return None
            initializers[initializer.name] = initializer

        producers: dict[str, tuple[int, onnx.NodeProto]] = {}
        for node_index, node in enumerate(model.graph.node):
            for output_name in node.output:
                if not output_name:
                    continue
                if output_name in producers:
                    return None
                producers[output_name] = (node_index, node)

        folds: list[_InitializerAliasFold] = []
        no_op_names: set[str] = set()
        for canonical_name in canonical_names:
            if canonical_name in initializers:
                no_op_names.add(canonical_name)
                continue

            aliases: set[str] = set()
            identity_indices: set[int] = set()
            visited: set[str] = set()
            source = canonical_name
            while source not in initializers:
                if source in visited:
                    return None
                visited.add(source)
                produced = producers.get(source)
                if produced is None:
                    return None
                node_index, producer = produced
                if (
                    producer.op_type != "Identity"
                    or len(producer.input) != 1
                    or len(producer.output) != 1
                ):
                    return None
                aliases.add(source)
                identity_indices.add(node_index)
                source = producer.input[0]

            folds.append(
                _InitializerAliasFold(
                    canonical_name=canonical_name,
                    source_initializer=initializers[source],
                    aliases=frozenset(aliases),
                    identity_indices=frozenset(identity_indices),
                )
            )

        all_aliases: set[str] = set()
        all_identity_indices: set[int] = set()
        canonical_set = set(canonical_names)
        for fold in folds:
            if (
                all_aliases.intersection(fold.aliases)
                or all_identity_indices.intersection(fold.identity_indices)
                or fold.aliases.intersection(no_op_names)
                or fold.source_initializer.name in canonical_set
            ):
                return None
            all_aliases.update(fold.aliases)
            all_identity_indices.update(fold.identity_indices)

        for fold in folds:
            other_aliases = all_aliases.difference(fold.aliases)
            for node_index in fold.identity_indices:
                if model.graph.node[node_index].input[0] in other_aliases:
                    return None

        return cls(tuple(folds))

    def apply(self, model: onnx.ModelProto) -> set[str]:
        """Apply all independent alias folds in bulk."""
        replacements: dict[str, str] = {}
        removed_node_indices: set[int] = set()
        alias_sources: set[str] = set()
        canonical_names = {fold.canonical_name for fold in self.folds}

        for fold in self.folds:
            initializer = onnx.TensorProto()
            initializer.CopyFrom(fold.source_initializer)
            initializer.name = fold.canonical_name
            model.graph.initializer.append(initializer)
            replacements.update(
                dict.fromkeys(fold.aliases, fold.canonical_name)
            )
            removed_node_indices.update(fold.identity_indices)
            alias_sources.add(fold.source_initializer.name)

        for node_index, node in enumerate(model.graph.node):
            if node_index in removed_node_indices:
                continue
            for input_index, input_name in enumerate(node.input):
                replacement = replacements.get(input_name)
                if replacement is not None:
                    node.input[input_index] = replacement

        retained_nodes = [
            node
            for node_index, node in enumerate(model.graph.node)
            if node_index not in removed_node_indices
        ]
        del model.graph.node[:]
        model.graph.node.extend(retained_nodes)

        retained_values = [
            item
            for item in model.graph.value_info
            if item.name not in replacements or item.name in canonical_names
        ]
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_values)
        return alias_sources


def _fold_rms_norm_initializers(
    model: onnx.ModelProto,
    metadata: GraphMetadata,
) -> None:
    """Make every RmsNorm weight a direct canonical initializer."""
    canonical_names = tuple(
        f"graph.{boundary.fqn}.weight"
        for boundary in metadata.boundaries
        if boundary.kind == "rms_norm"
    )
    plan = _InitializerAliasFoldPlan.try_build(model, canonical_names)
    if plan is not None:
        alias_sources = plan.apply(model)
    else:
        alias_sources = set()
        for canonical_name in canonical_names:
            source = _fold_initializer_alias(
                model,
                canonical_name,
            )
            if source is not None:
                alias_sources.add(source)
    used_inputs = {
        input_name
        for node in model.graph.node
        for input_name in node.input
        if input_name
    }
    retained_initializers = [
        item
        for item in model.graph.initializer
        if item.name not in alias_sources or item.name in used_inputs
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)


def export_standard_onnx(
    graph: GraphModule,
    metadata: GraphMetadata,
    directory: Path,
) -> onnx.ModelProto:
    """Export and validate the standard ONNX intermediate model."""
    try:
        with tempfile.TemporaryDirectory(
            prefix=".mdc-standard.",
            dir=directory,
            ignore_cleanup_errors=True,
        ) as temporary_directory:
            temporary = Path(temporary_directory) / "model.onnx"
            torch.onnx.export(
                _PositionalGraph(
                    graph,
                    tuple(item.name for item in metadata.input_abi),
                    use_kwargs=metadata.stage.is_prefill,
                ),
                _example_arguments(metadata),
                temporary,
                export_params=True,
                opset_version=MDC_ONNX_OPSET,
                # Legacy JIT constant folding evaluates parameters and constants
                # together, which is unsafe for captured mixed-device graphs.
                do_constant_folding=False,
                input_names=[item.name for item in metadata.input_abi],
                output_names=[item.name for item in metadata.output_abi],
                # ExportedProgram graph modules reject recursive train() calls;
                # the adapter is already fixed in eval mode.
                training=torch.onnx.TrainingMode.PRESERVE,
                dynamo=False,
            )
            standard = onnx.load(temporary, load_external_data=True)
            _restore_linear_initializer_names(standard, graph)
            _fold_rms_norm_initializers(standard, metadata)
            custom_names = {
                schema.onnx_name for schema in OPERATOR_SCHEMAS.values()
            }
            contains_custom = any(
                node.op_type in custom_names for node in standard.graph.node
            )
            if contains_custom:
                _seed_custom_value_info(standard)
            standard = shape_inference.infer_shapes(
                standard,
                strict_mode=not contains_custom,
                data_prop=True,
            )
            if contains_custom:
                validate_mdc_model(standard)
            else:
                onnx.checker.check_model(standard, full_check=True)
            return standard
    except OnnxExportError:
        raise
    except Exception as error:
        raise OnnxExportError(f"Standard ONNX validation failed: {error}") from error
