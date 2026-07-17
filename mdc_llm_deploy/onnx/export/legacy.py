"""Legacy PyTorch ONNX exporter adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from torch.fx import GraphModule

from ...errors import OnnxExportError
from ...graph.metadata import GraphMetadata
from ...operators.contracts.onnx import MDC_ONNX_OPSET
from ...placement.inputs import resolve_input_devices

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


def _example_arguments(metadata: GraphMetadata) -> tuple[torch.Tensor, ...]:
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


def export_legacy_onnx(
    graph: GraphModule,
    metadata: GraphMetadata,
    directory: Path,
) -> onnx.ModelProto:
    """Export and materialize a legacy PyTorch ONNX model."""
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
        return onnx.load(temporary, load_external_data=True)
