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
from ..onnx_protocol import MDC_ONNX_OPSET

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


def _device(graph: GraphModule) -> torch.device:
    for tensor in (*tuple(graph.parameters()), *tuple(graph.buffers())):
        return tensor.device
    return torch.device("cpu")


def _example_arguments(
    graph: GraphModule,
    metadata: GraphMetadata,
) -> tuple[torch.Tensor, ...]:
    device = _device(graph)
    result: list[torch.Tensor] = []
    for item in metadata.input_abi:
        try:
            dtype = _TORCH_DTYPES[item.dtype]
        except KeyError as error:
            raise OnnxExportError(f"Unsupported input dtype: {item.dtype}") from error
        result.append(torch.zeros(item.shape, dtype=dtype, device=device))
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
                _example_arguments(graph, metadata),
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
            onnx.checker.check_model(standard, full_check=True)
            standard = shape_inference.infer_shapes(
                standard,
                strict_mode=True,
                data_prop=True,
            )
            onnx.checker.check_model(standard, full_check=True)
            return standard
    except OnnxExportError:
        raise
    except Exception as error:
        raise OnnxExportError(f"Standard ONNX validation failed: {error}") from error
