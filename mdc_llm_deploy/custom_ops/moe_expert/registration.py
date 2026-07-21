"""MoeExpert plugin description and process-local Torch registration."""

from __future__ import annotations

from ..base import OnnxOperatorSpec, OperatorPlugin, TorchOperatorSpec
from ..registry import RegisteredOperator, register_operator
from .kernels import cpu, cuda, fake
from .onnx import create_schema, translate

PLUGIN = OperatorPlugin(
    name="moe_expert",
    torch=TorchOperatorSpec(
        qualified_name="mdc_llm_deploy::moe_expert",
        schema=(
            "(Tensor x, Tensor topk_ids, Tensor topk_weight, Tensor expert_weights, "
            "Tensor? quant_scales=None, Tensor? quant_offsets=None) -> Tensor"
        ),
        cpu_kernel=cpu,
        cuda_kernel=cuda,
        fake_kernel=fake,
    ),
    onnx=OnnxOperatorSpec(schema=create_schema(), translation=translate),
)

REGISTERED_OPERATOR: RegisteredOperator = register_operator(PLUGIN)
moe_expert = REGISTERED_OPERATOR.torch.definition
