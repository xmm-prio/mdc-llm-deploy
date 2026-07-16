# MDC LLM Deploy

面向 MDC 部署的 Qwen3/Qwen3-MoE 推理、量化与 ONNX 导出库。模型实现兼容
Transformers checkpoint 命名，但不继承 Transformers 官方模型。

## 安装

支持 Python 3.11 和 3.12：

```bash
python -m pip install -e .
```

`requirements.txt` 是当前验证环境的精确快照；依赖声明真源仍是 `pyproject.toml`。

## 加载 Qwen3

`ExportModelConfig` 在模型构造时固定序列长度、RoPE cos/sin 和 mask 语义。
`mask_mode="causal"` 生成 BOOL 因果 mask；`mask_mode="none"` 不产生 mask 运算。

```python
import torch

from mdc_llm_deploy import AutoExportModel, ExportModelConfig

model = AutoExportModel.from_pretrained(
    "Qwen/Qwen3-4B",
    ExportModelConfig(sequence_length=3072, mask_mode="causal"),
    dtype=torch.float16,
)
```

loader 支持本地目录或 Hugging Face Hub、单文件或分片 safetensors。Qwen3-MoE
checkpoint 的 gate/up/down 专家权重在加载时整理为 expert-major rank-2 packed 权重。
tied embedding 会复制为独立 `lm_head.weight`，避免导出时共享参数产生歧义。

也可直接用正规化配置构造模型：

```python
from mdc_llm_deploy import (
    ExportModelConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

model = Qwen3ForCausalLM(
    Qwen3Config(),
    ExportModelConfig(sequence_length=3072),
)
```

## 导出流程

```python
from mdc_llm_deploy import convert_to_decode, export, oneshot, onnx_export

inputs = {
    "input_ids": torch.zeros((1, 3072), dtype=torch.int64),
}
prefill = export(model, inputs)

# 可选：必须在 prefill 阶段量化。
oneshot(prefill, "configs/minmax-linear-w8a8.json", [inputs])

onnx_export(prefill, "qwen3-prefill.onnx")
convert_to_decode(prefill)
onnx_export(prefill, "qwen3-decode.onnx")
```

`export()` 捕获静态 FX 图，并按 RMSNorm、RoPE、FIA 顺序规范化。未命中融合模式时保留标准
小算子。每层 KV 输出命名为 `present.N.key/value`；decode 输入命名为
`past.N.key/value`。

`onnx_export(graph, output_path, *, external_data=True)` 保留 FX 输入输出顺序，验证后原子
覆盖目标。默认 external data 文件名为 `<模型文件名>.data`。mask 语义来自构造模型时的
`ExportModelConfig`，ONNX API 不再接受 mask 选择参数。

## MoE ABI

`MoeExpert` 接收：

- 浮点 activation `[token_count, hidden_size]`；
- routing id/weight `[token_count, top_k]`，`top_k` 由 shape 决定；
- expert-major packed 权重 `[expert_count, packed_width]`，每行顺序固定为
  `gate_proj, up_proj, down_proj`；
- 浮点权重不传量化参数；INT8 权重传每 expert、每 projection 的 scale 和可选 offset。

算子只支持推理，不提供 autograd。

## 发布组合

发布矩阵覆盖 Dense/MoE、prefill/decode、FP16/受支持 MinMax。产物 mask 语义不再作为
独立矩阵维度，由模型配置决定。GPTQ 保持 FX-only，不导出 ONNX。
