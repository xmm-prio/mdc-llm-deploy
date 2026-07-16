**MDC LLM Deploy**

将 Transformers/PyTorch 大语言模型转换为静态 ATen FX 图，执行 PTQ 和 KV Cache
改写，并导出面向 MDC 的 ONNX。当前版本为 `0.1.0`。

# 安装

支持 Python 3.11 和 3.12。维护与发布检查使用 Python 3.12。

在仓库根目录创建虚拟环境，并按锁文件安装可复现依赖：

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
```

只安装核心依赖时可直接执行：

```bash
python -m pip install -e .
```

需要 Transformers、Accelerate、Datasets、Evaluate、ONNXScript 和 Triton 等扩展依赖时：

```bash
python -m pip install -e ".[full]"
```

开发依赖可通过 `python -m pip install -e ".[dev]"` 安装。Linux 环境会安装 `triton`；
其他平台不会安装该条件依赖。

# API

根包 `mdc_llm_deploy` 提供 11 个公开符号。依赖 PyTorch 或 ONNX 的流程函数采用延迟导入；
读取配置、异常类型或 `__version__` 不会加载这些运行时。

## 流程函数

### `export`

```python
export(
    model: torch.nn.Module,
    example_inputs: Mapping[str, torch.Tensor],
) -> torch.fx.GraphModule
```

将处于 `eval()` 模式的模型导出为静态、函数式 ATen FX 图。`example_inputs` 必须非空，
键为字符串，值为 Tensor；当前导出入口要求二维 `input_ids`。模型参数、buffer 和输入必须位于
同一设备。示例输入 shape 会固化到图中。

成功后图处于 `FLOAT_PREFILL` 阶段。无法生成受支持的 ATen 图或无法识别所需模型结构时抛出
`UnsupportedPatternError`。

### `oneshot`

```python
oneshot(
    graph: torch.fx.GraphModule,
    config: QuantizationConfig | Mapping[str, object] | str,
    calibration_dataloader: Iterable[Mapping[str, torch.Tensor]],
) -> torch.fx.GraphModule
```

校准并 fake-quantize `FLOAT_PREFILL` 图。该函数以事务方式原地修改图，并返回同一
`GraphModule`；成功后阶段变为 `QUANTIZED_PREFILL`。失败时原图保持不变。

`config` 可传 `QuantizationConfig`、映射或 JSON 文件路径字符串。底层
`QuantizationConfig.load()` 也接受 `pathlib.Path`。校准 batch 的键必须与 FX placeholder
一致。空 modifier 链是 no-op；选择器未命中 target 会抛出 `QuantizationConfigError`。

### `convert_to_decode`

```python
convert_to_decode(
    graph: torch.fx.GraphModule,
) -> torch.fx.GraphModule
```

将静态 prefill 图原地改写为单 token decode 图，并返回同一 `GraphModule`。浮点图和量化图
均可转换；对应阶段分别变为 `FLOAT_DECODE` 和 `QUANTIZED_DECODE`。输入序列长度必须至少为
2，且图中必须存在可识别的 attention 与 K/V 边界。

如需同时导出 prefill 和 decode，应先导出 prefill，再调用此函数。decode 图不能执行
`oneshot()`，重复转换也会抛出 `GraphStateError`。

### `onnx_export`

```python
onnx_export(
    graph: torch.fx.GraphModule,
    output_path: str | pathlib.Path,
    *,
    mask_mode: Literal["masked", "maskless"],
    overwrite: bool = False,
) -> onnx.ModelProto
```

将 prefill 或 decode FX 图降低为 MDC ONNX 方言，原子写入 `.onnx` 文件，并返回
`onnx.ModelProto`。`mask_mode="masked"` 使用显式因果语义；`"maskless"` 使用全可见、
非因果语义。

目标文件默认不可覆盖，已存在时抛出 `FileExistsError`；显式传入 `overwrite=True` 才允许
替换。GPTQ 图不支持 ONNX 导出。输出包含扩展算子语义，不应假定通用 ONNX Runtime 可以执行。

## 配置对象

### `QuantizationConfig`

```python
QuantizationConfig(
    modifiers: tuple[Modifier, ...],
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
)
```

不可变、有序的量化配置对象，不依赖 PyTorch。常用接口：

- `QuantizationConfig.from_dict(value)`：严格解析 JSON 兼容映射。
- `QuantizationConfig.load(value)`：接受现有对象、映射、字符串路径或 `Path`。
- `to_dict()`：返回包含已解析默认值的字典。
- `to_json_string()`：返回键排序、缩进为 2 且以换行结尾的 JSON。
- `fingerprint`：返回规范化配置的 64 位小写 SHA-256。

## 异常

- `MdcDeployError`：库定义异常的基类。
- `GraphStateError`：当前图阶段不允许所请求操作。
- `UnsupportedPatternError`：模型或图模式无法表示、融合或改写。
- `QuantizationConfigError`：量化配置格式、类型、选择或组合无效；同时继承 `ValueError`。
- `OnnxExportError`：MDC ONNX 降低、兼容性检查或结构校验失败。

## 版本

`__version__` 返回当前包版本字符串；当前值为 `"0.1.0"`。

# 示例

以下代码只展示官方模型的加载方式和 API 调用顺序，不是开箱即用脚本。`0.1.0` 未承诺官方
Qwen3 checkpoint 的直接兼容性；实际接入需要先确认模型输出、attention 边界、静态 shape、
显存和目标导出能力。

示例统一使用 `Qwen/Qwen3-4B`。模型必须使用 eager attention、处于 `eval()` 模式，且输入与
参数位于同一设备。

## FP16 ONNX

跳过 `oneshot()`，直接导出浮点 prefill 图：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mdc_llm_deploy import export, onnx_export

model_id = "Qwen/Qwen3-4B"
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    attn_implementation="eager",
).eval().to(device)
input_ids = tokenizer("示例输入", return_tensors="pt").input_ids.to(device)
inputs = {"input_ids": input_ids}

graph = export(model, inputs)
onnx_export(
    graph,
    "artifacts/qwen3-4b-fp16-prefill.onnx",
    mask_mode="masked",
)
```

## MinMax prefill 与 decode

量化必须发生在 prefill 阶段。先导出 prefill，再原地改写同一张图并导出 decode：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mdc_llm_deploy import convert_to_decode, export, oneshot, onnx_export

model_id = "Qwen/Qwen3-4B"
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    attn_implementation="eager",
).eval().to(device)
input_ids = tokenizer("校准输入", return_tensors="pt").input_ids.to(device)
inputs = {"input_ids": input_ids}

graph = export(model, inputs)
oneshot(graph, "configs/minmax-linear-w8a8.json", [inputs])

onnx_export(
    graph,
    "artifacts/qwen3-4b-minmax-prefill.onnx",
    mask_mode="masked",
)
convert_to_decode(graph)
onnx_export(
    graph,
    "artifacts/qwen3-4b-minmax-decode.onnx",
    mask_mode="masked",
)
```

`convert_to_decode()` 使用导出输入的最后一个位置构造单步 decode，因此输入序列长度必须至少为
2。发布形状与短文本示例不同；接入时应显式构造目标静态长度的校准输入。

## GPTQ FX

GPTQ 只用于 FX 数值路径，不调用 `onnx_export()`：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mdc_llm_deploy import export, oneshot

model_id = "Qwen/Qwen3-4B"
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    attn_implementation="eager",
).eval().to(device)
input_ids = tokenizer("校准输入", return_tensors="pt").input_ids.to(device)
inputs = {"input_ids": input_ids}

graph = export(model, inputs)
oneshot(graph, "configs/gptq-linear-w4a8.json", [inputs])
outputs = graph(**inputs)
```

接入 MoE 模型时，将模型 ID 替换为 `Qwen/Qwen3-30B-A3B`。针对 MoE expert 的 MinMax 和
GPTQ 配置分别使用 `configs/minmax-moe-w8a8.json` 与
`configs/gptq-moe-w8a8.json`；attention 与普通 linear 仍使用各自配置。

# 配置文件

仓库提供五个版本化配置：

- `configs/minmax-linear-w8a8.json`：Linear 权重 int8 per-channel 对称量化；激活 int8
  static per-tensor 对称量化；支持 ONNX 导出。
- `configs/minmax-attention-a8.json`：Attention query、key、value、score 分别执行 int8
  static per-tensor 对称量化；支持 ONNX 导出。
- `configs/minmax-moe-w8a8.json`：MoE expert 权重与激活执行 int8 static per-tensor
  对称量化；支持 ONNX 导出。
- `configs/gptq-linear-w4a8.json`：Linear 权重 int4 per-channel，激活 int8 static
  per-tensor；只支持 FX 数值路径。
- `configs/gptq-moe-w8a8.json`：MoE expert 权重与激活执行 int8 static per-tensor；
  只支持 FX 数值路径。

## 结构

配置根对象包含：

- `modifiers`：按声明顺序执行的 `minmax` 或 `gptq` 操作链。
- `include`：根级 FQN 选择器；空列表表示选择全部已发现 target。
- `exclude`：根级 FQN 排除器；命中时优先于 `include`。

每个 modifier 可定义局部 `include`、`exclude`，以及 `linear`、`attention`、`moe`
target。局部选择器一旦提供，会完整替换对应根级选择器。不同 modifier 可以处理互不重叠的
target；同一 target 被重复命中时抛出 `QuantizationConfigError`。

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

## 字段与能力边界

- 权重 `bits` 支持 `4`、`8`；`granularity` 支持 `per_tensor`、`per_channel`。
- 激活 `bits` 支持 `4`、`8`；`granularity` 支持 `per_tensor`、`per_token`；
  `mode` 支持 `static`、`dynamic`。
- `symmetric` 默认值为 `true`。
- Attention 可分别配置 `query`、`key`、`value`、`score`；未配置的边保持浮点。
- GPTQ 支持 `percdamp`、`actorder`、`block_size`，不接受 attention 配置。
- 配置解析严格拒绝未知字段、重复 JSON 字段和不合法组合。
- 配置模型可表达的组合不等于当前导出后端支持的组合；不支持的组合会在规划、图校验或
  `onnx_export()` 阶段明确失败。

完整语义见[量化配置设计](docs/designs/config.md)，机器可读约束见
[JSON Schema](mdc_llm_deploy/config/schema.json)。
