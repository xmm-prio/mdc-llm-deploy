"""Validated export from an ATen FX graph to standard ONNX."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import TensorProto, shape_inference
from torch.fx import GraphModule

from ..errors import OnnxExportError
from ..fx_inspection import linear_weight_name
from ..graph_types import GraphMetadata
from ..input_placement import resolve_input_devices
from ..onnx_protocol import MDC_ONNX_OPSET
from ..operator_schema import OPERATOR_SCHEMAS
from .validation.model import validate_mdc_model

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


def _restore_linear_initializer_names(
    model: onnx.ModelProto,
    graph: GraphModule,
) -> None:
    """Restore FX parameter FQNs lost by the legacy ONNX exporter."""
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
    for parameter_name, old_name in zip(parameter_names, onnx_weight_names, strict=True):
        new_name = f"graph.{parameter_name}"
        initializers[old_name].name = new_name
        for node in model.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name == old_name:
                    node.input[index] = new_name


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


def _fold_rms_norm_initializers(
    model: onnx.ModelProto,
    metadata: GraphMetadata,
) -> None:
    """Make every RmsNorm weight a direct canonical initializer."""
    alias_sources: set[str] = set()
    for boundary in metadata.boundaries:
        if boundary.kind == "rms_norm":
            source = _fold_initializer_alias(
                model,
                f"graph.{boundary.fqn}.weight",
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
                do_constant_folding=True,
                input_names=[item.name for item in metadata.input_abi],
                output_names=[item.name for item in metadata.output_abi],
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
