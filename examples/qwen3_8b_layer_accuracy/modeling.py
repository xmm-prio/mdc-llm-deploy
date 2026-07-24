"""Qwen3 decoder-layer model and deterministic prompt input construction."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODEL_ID = "Qwen/Qwen3-8B"
SEQUENCE_LENGTH = 128

CALIBRATION_PROMPTS = (
    "自动驾驶系统需要同时理解道路结构, 车辆运动, 交通信号和突发障碍物, 并在严格时延约束下做出稳定决策.",
    "大语言模型部署前需要验证算子兼容性, 量化误差, 内存占用和端到端性能, 任何阶段的数值偏差都应单独定位.",
    "静态量化使用校准数据估计激活范围, 数据分布必须覆盖实际输入, 同时不能与最终精度评估样本重合.",
    "Transformer 层包含归一化, 自注意力和前馈网络, 残差连接会传播局部误差, 因此需要比较完整层输出.",
)

EVALUATION_PROMPTS = (
    "在固定输入上分别运行 Torch, ONNX 和 MDC, 逐段比较输出可以区分模型量化误差, 图转换误差与硬件执行误差.",
    "精度验收不仅检查余弦相似度, 还应检查最大绝对误差, 平均绝对误差, 相对误差以及是否出现非有限数值.",
    "可靠的软件验证流程应保存模型配置, 输入张量, 量化参数和运行环境, 使失败能够稳定复现并被自动回归测试覆盖.",
)


class Qwen3DecoderLayerHarness(nn.Module):
    """Expose one Qwen3 decoder layer through a stable tensor-only ABI."""

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_model(cls, source: PreTrainedModel) -> Qwen3DecoderLayerHarness:
        """Build a lightweight one-layer Qwen3 orchestrator without final norm."""
        model = copy.copy(source)
        model._modules = source._modules.copy()
        backbone = copy.copy(source.model)
        backbone._modules = source.model._modules.copy()
        backbone.embed_tokens = nn.Identity()
        backbone.layers = nn.ModuleList([copy.deepcopy(source.model.layers[0])])
        backbone.norm = nn.Identity()
        backbone.rotary_emb = copy.deepcopy(source.model.rotary_emb)
        model.model = backbone
        model.lm_head = nn.Identity()
        model.config = copy.deepcopy(source.config)
        model.config.num_hidden_layers = 1
        model.config.use_cache = False
        return cls(model).eval()

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Run one prefill pass without a KV cache."""
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0]


@dataclass(slots=True)
class LayerSource:
    """Own source model pieces needed to derive layer inputs and copies."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def clone_harness(self) -> Qwen3DecoderLayerHarness:
        """Create an independent copy of the first decoder layer."""
        return Qwen3DecoderLayerHarness.from_model(self.model)


def load_layer_source(model_id: str, device: torch.device) -> LayerSource:
    """Load pretrained Qwen3 with only its first decoder layer materialized."""
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 1
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.set_attn_implementation("eager")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return LayerSource(model=model.eval().to(device), tokenizer=tokenizer)


def prepare_prompt_inputs(
    source: LayerSource,
    prompts: Sequence[str],
    *,
    sequence_length: int,
    device: torch.device,
) -> list[dict[str, Tensor]]:
    """Create fixed-shape real-text hidden states and rotary inputs."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not prompts:
        raise ValueError("prompts must not be empty")

    inputs: list[dict[str, Tensor]] = []
    for prompt in prompts:
        expanded_prompt = " ".join([prompt] * sequence_length)
        tokenized = source.tokenizer(
            expanded_prompt,
            max_length=sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)
        token_mask = tokenized["attention_mask"].to(device)
        if not bool(token_mask.all()):
            raise ValueError("expanded prompt did not fill the requested sequence length")
        position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
        with torch.inference_mode():
            hidden_states = source.model.model.embed_tokens(input_ids)
        inputs.append(
            {
                "inputs_embeds": hidden_states.detach(),
                "attention_mask": token_mask,
                "position_ids": position_ids,
            }
        )
    return inputs


def move_inputs(inputs: dict[str, Tensor], device: torch.device | str) -> dict[str, Tensor]:
    """Move all layer inputs to one device."""
    return {name: value.to(device) for name, value in inputs.items()}


__all__ = [
    "CALIBRATION_PROMPTS",
    "EVALUATION_PROMPTS",
    "MODEL_ID",
    "SEQUENCE_LENGTH",
    "LayerSource",
    "Qwen3DecoderLayerHarness",
    "load_layer_source",
    "move_inputs",
    "prepare_prompt_inputs",
]
