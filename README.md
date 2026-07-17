# MDC LLM Deploy

面向 MDC 的 Qwen3/Qwen3-MoE 静态图捕获、量化和 ONNX 导出库。模型参数命名兼容
Transformers checkpoint，但模型实现不继承 Transformers 官方模型。

[安装](#安装) · [API](#api) · [示例](#示例) · [配置文件](#配置文件)

## 安装

要求 Python 3.11 或 3.12。当前未提供 PyPI 发布包，请在仓库根目录使用 editable 安装：

```bash
python -m pip install -e .
```

开发环境可安装测试、类型检查和构建工具：

```bash
python -m pip install -e ".[dev]"
```

`pyproject.toml` 是依赖声明真源；`requirements.txt` 仅记录当前验收环境的精确快照。
仓库根目录的 `configs/quantization/*.json` 不会打入 wheel，因此下文预设路径仅适用于仓库内安装和运行。

## API

所有稳定入口均可从 `mdc_llm_deploy` 根包导入。

### 核心流程

- `export(model, example_inputs) -> torch.fx.GraphModule`
捕获 eval 模型，生成带 MDC 元数据的静态 prefill ATen FX 图。当前输入必须包含 rank-2
`int64` 的 `input_ids`，且模型、输入和参数必须位于同一设备。
- `oneshot(graph, config, calibration_dataloader) -> torch.fx.GraphModule`
在 `FLOAT_PREFILL` 阶段校准并插入 fake-quant。`config` 可为 `QuantizationConfig`、字典或
UTF-8 JSON 路径；量化必须先于 decode 转换。
- `convert_to_decode(graph) -> torch.fx.GraphModule`
将序列长度至少为 2 的 prefill 图原子改写为单 token decode 图。函数更新并返回同一图；
decode 输入增加每层 `past.N.key/value`。
- `onnx_export(graph, output_path, *, external_data=True) -> onnx.ModelProto`
将 FX 图转换并校验为 MDC ONNX。路径必须以 `.onnx` 结尾；默认同时写入
`<output_path>.data`，设置 `external_data=False` 可将权重内嵌到模型文件。
- `standard_onnx_export(graph, output_path, *, external_data=True) -> onnx.ModelProto`
将带合法图元数据的模型无关 FX 图导出为标准 ONNX，不执行 MDC lowering、不写入
`mdc.*` 属性，也不保证产物可部署到 MDC。适用于通用算子小图和标准 ONNX 工具链。

prefill 和 decode ONNX 都只公开 `logits` 输出。decode KV 缓存由调用方通过
`past.N.key/value` 输入提供。

### 模型

- `AutoExportModel.from_pretrained(source, export_config, *, dtype=torch.float16, revision=None, local_files_only=False)`
从本地目录或 Hugging Face Hub 加载单文件或分片 safetensors，并按 checkpoint 自动选择
Qwen3 Dense 或 Qwen3-MoE。只有已存在目录会按本地源处理；已存在的非目录路径会抛出
`ValueError`，其他值按 Hugging Face Hub 仓库 ID 解析。
- `ExportModelConfig(sequence_length, mask_mode="causal")`
固定序列长度、RoPE cache 和 mask 语义。`mask_mode` 仅接受 `"causal"` 或 `"none"`；
ONNX API 不再单独接收 mask 参数。
- `Qwen3Config` / `Qwen3MoeConfig`
Qwen3 Dense/MoE 架构的正规化配置。
- `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM`
导出专用模型。构造后处于 eval 模式、关闭梯度，输入形状固定为
`(1, sequence_length)`。



### 配置、版本和异常

- `QuantizationConfig`：使用 `load()`、`from_dict()`、`to_dict()`、`to_json_string()` 和
`fingerprint` 加载、校验、序列化及标识量化配置。
- `__version__`：库版本。
- `MdcDeployError`：库异常基类。
- `GraphStateError`：操作与当前图阶段不匹配。
- `UnsupportedPatternError`：模型或图模式不受支持。
- `QuantizationConfigError`：量化配置无效。
- `OnnxExportError`：ONNX lowering、校验或写入失败。



## 示例



### 导出并量化 Qwen3

以下流程从 Hugging Face Hub 加载 Qwen3，将 Linear 权重和激活量化为 W8A8，然后依次生成
prefill 和 decode ONNX：

```python
import torch

from mdc_llm_deploy import (
    AutoExportModel,
    ExportModelConfig,
    convert_to_decode,
    export,
    oneshot,
    onnx_export,
)

sequence_length = 3072
device = torch.device("cuda")
model = AutoExportModel.from_pretrained(
    "Qwen/Qwen3-4B",
    ExportModelConfig(
        sequence_length=sequence_length,
        mask_mode="causal",
    ),
    dtype=torch.float16,
).to(device)
inputs = {
    "input_ids": torch.zeros(
        (1, sequence_length),
        dtype=torch.int64,
        device=device,
    )
}

graph = export(model, inputs)
graph = oneshot(
    graph,
    "configs/quantization/minmax-linear-w8a8.json",
    [inputs],
)

onnx_export(graph, "output/qwen3-prefill.onnx")
graph = convert_to_decode(graph)
onnx_export(graph, "output/qwen3-decode.onnx")
```

> [!WARNING]
> `convert_to_decode()` 会更新同一 FX 图。请先导出 prefill，再执行转换和 decode 导出；转换后不能
> 再从该图生成 prefill。量化同样必须在转换前完成。
>
> 模型、导出输入和 `oneshot()` 的校准批次必须位于同一设备。GPU 导出时，请在创建输入及校准
> 数据时显式传入同一个 CUDA `device`。

不需要量化时，跳过 `oneshot()` 即可导出 FP16 产物。默认输出为：

```text
output/qwen3-prefill.onnx
output/qwen3-prefill.onnx.data
output/qwen3-decode.onnx
output/qwen3-decode.onnx.data
```



### 从本地 checkpoint 加载

```python
model = AutoExportModel.from_pretrained(
    "./checkpoints/Qwen3-4B",
    ExportModelConfig(sequence_length=3072),
    local_files_only=True,
)
```



### 直接构造小型模型

直接构造适合接口测试；生产权重应通过 `from_pretrained()` 加载。

```python
from mdc_llm_deploy import (
    ExportModelConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

config = Qwen3Config(
    vocab_size=128,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=16,
)
model = Qwen3ForCausalLM(
    config,
    ExportModelConfig(sequence_length=16),
)
```



### 读取并校验配置

```python
from mdc_llm_deploy import QuantizationConfig

config = QuantizationConfig.load(
    "configs/quantization/minmax-linear-w8a8.json"
)
print(config.fingerprint)
print(config.to_json_string())
```



## 配置文件

量化配置是严格 JSON：未知字段、重复键和无效组合会抛出 `QuantizationConfigError`。完整约束见
[配置设计](docs/designs/config.md) 和
[JSON Schema](mdc_llm_deploy/config/schema.json)。

仓库提供五份版本化预设：

- `[minmax-linear-w8a8.json](configs/quantization/minmax-linear-w8a8.json)`：Dense/MoE Linear 的
W8 per-channel + A8 static per-tensor；MoE router 也属于 Linear。
- `[minmax-attention-a8.json](configs/quantization/minmax-attention-a8.json)`：Dense/MoE Attention 的
query、key、value 和 score A8 static per-tensor。
- `[minmax-moe-w8a8.json](configs/quantization/minmax-moe-w8a8.json)`：Qwen3-MoE 专家权重和激活
W8A8 static per-tensor，不包含 router。
- `[gptq-linear-w4a8.json](configs/quantization/gptq-linear-w4a8.json)`：Linear W4 per-channel +
A8 static per-tensor，仅支持 FX。
- `[gptq-moe-w8a8.json](configs/quantization/gptq-moe-w8a8.json)`：schema 合法的 MoE GPTQ W8A8
配置，但当前 packed Qwen3-MoE 权重会被 `oneshot()` 明确拒绝。

所有 GPTQ 路径均为 FX-only，不支持 ONNX 或 ATC。

### 基本结构

```json
{
  "include": [],
  "exclude": [],
  "modifiers": [
    {
      "type": "minmax",
      "linear": {
        "weight": {
          "bits": 8,
          "granularity": "per_channel",
          "symmetric": true
        },
        "activation": {
          "bits": 8,
          "granularity": "per_tensor",
          "mode": "static",
          "symmetric": true
        }
      }
    }
  ]
}
```

- `modifiers`：按声明顺序执行的 `minmax` 或 `gptq` 操作；空数组表示 no-op。
- `include` / `exclude`：按模块完全限定名匹配。空 `include` 选择所有已发现 target，
`exclude` 优先。
- modifier 内的 `include` / `exclude`：提供后会替换根级选择器，不与根级值合并。
- `linear`：Linear 权重和激活；MoE router 也由此 target 选择。
- `attention`：分别配置 `query`、`key`、`value`、`score` 激活。
- `moe`：配置 routed expert 权重和激活，不包含 router。

权重 `bits` 支持 4/8，`granularity` 支持 `per_tensor`/`per_channel`。激活 `bits` 支持
4/8，`granularity` 支持 `per_tensor`/`per_token`，`mode` 支持 `static`/`dynamic`；
`symmetric` 默认值为 `true`。

GPTQ 额外支持 `percdamp`、`actorder` 和 `block_size`，不允许配置 `attention`。同一 FQN
target 只能由一个 modifier 命中；重叠配置不会覆盖，而会在规划阶段报错。