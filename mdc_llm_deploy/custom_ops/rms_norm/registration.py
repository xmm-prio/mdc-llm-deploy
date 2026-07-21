"""RmsNorm plugin description and process-local Torch registration."""

from __future__ import annotations

from ..base import OnnxOperatorSpec, OperatorPlugin, TorchOperatorSpec
from ..registry import register_operator
from .fake import fake
from .kernels import cpu, cuda
from .onnx import ONNX_SCHEMA, translate

PLUGIN = OperatorPlugin(
    name="rms_norm",
    torch=TorchOperatorSpec(
        qualified_name="mdc_llm_deploy::rms_norm",
        schema="(Tensor x, Tensor gamma, float epsilon=1e-6) -> (Tensor y, Tensor rstd)",
        cpu_kernel=cpu,
        cuda_kernel=cuda,
        fake_kernel=fake,
    ),
    onnx=OnnxOperatorSpec(
        schema=ONNX_SCHEMA,
        translation=translate,
    ),
)

_REGISTERED = register_operator(PLUGIN)
rms_norm = _REGISTERED.torch.definition
