"""Plugin registration for ApplyRotaryPosEmb."""

from __future__ import annotations

from ..base import OnnxOperatorSpec, OperatorPlugin, TorchOperatorSpec
from ..registry import RegisteredOperator, register_operator
from .contract import QUALIFIED_NAME, TORCH_SCHEMA
from .fake import fake
from .kernels import cpu, cuda
from .onnx import create_schema, translate

PLUGIN = OperatorPlugin(
    name="apply_rotary_pos_emb",
    torch=TorchOperatorSpec(
        qualified_name=QUALIFIED_NAME,
        schema=TORCH_SCHEMA,
        cpu_kernel=cpu,
        cuda_kernel=cuda,
        fake_kernel=fake,
    ),
    onnx=OnnxOperatorSpec(schema=create_schema(), translation=translate),
)

REGISTERED_OPERATOR: RegisteredOperator = register_operator(PLUGIN)
apply_rotary_pos_emb = REGISTERED_OPERATOR.torch.definition
