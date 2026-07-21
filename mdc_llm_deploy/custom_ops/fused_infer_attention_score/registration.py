"""Plugin assembly and process-local Torch registration."""

from __future__ import annotations

from ..base import OnnxOperatorSpec, OperatorPlugin, TorchOperatorSpec
from ..registry import RegisteredOperator, register_operator
from .contract import PLUGIN_NAME, QUALIFIED_NAME, TORCH_SCHEMA
from .fake import fake_attention
from .kernels import attention_kernel
from .onnx import create_schema, translate

PLUGIN = OperatorPlugin(
    name=PLUGIN_NAME,
    torch=TorchOperatorSpec(
        qualified_name=QUALIFIED_NAME,
        schema=TORCH_SCHEMA,
        cpu_kernel=attention_kernel,
        cuda_kernel=attention_kernel,
        fake_kernel=fake_attention,
    ),
    onnx=OnnxOperatorSpec(
        schema=create_schema(),
        translation=translate,
    ),
)

REGISTERED_OPERATOR: RegisteredOperator = register_operator(PLUGIN)
fused_infer_attention_score = REGISTERED_OPERATOR.torch.definition
