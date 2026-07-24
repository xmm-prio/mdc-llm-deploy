# MDC LLM Deploy

**面向 MDC 部署的 Transformers-compatible LLM 量化与 ONNX 图转换库。**

[`量化`](docs/quantization.md) · [`ONNX 图处理`](docs/onnx.md) ·
[`导出示例`](examples/README.md) · [`算子文档`](docs/operators)

MDC LLM Deploy 为 PyTorch/Transformers 模型提供 MinMax INT8 fake-quant，
并将标准 ONNX 图转换为 MDC 可识别的量化与融合算子。当前示例覆盖 Qwen3-4B 和
Qwen3-8B，支持使用单层模型、缩小词表和静态 shape 快速验证导出链路。

主要能力：

- 原地量化 Dense `torch.nn.Linear`，支持一步式和三阶段 API；
- 导出标准 ONNX opset 21 QDQ 图；
- 将受支持的 W8A8 QDQ 子图 lowering 为 MDC 量化算子；
- 融合 Qwen3 RMSNorm、RoPE 和 FusedInferAttentionScore；
- 导出 Qwen3 FP16、W8A8 和 TP=2 静态 ONNX 图。

## 安装

项目需要 Python `>=3.11,<3.13`。推荐在 Linux 虚拟环境中安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

开发环境额外安装测试、类型检查和代码检查依赖：

```bash
python -m pip install ".[dev]"
```

> [!NOTE]
> PyTorch、Transformers、ONNX 和 ONNXScript 的已验证版本由
> [`pyproject.toml`](pyproject.toml) 固定。CUDA 不是 API 的必要条件，但完整 LLM
> 导出通常需要较大内存或显存。MDC、CANN、ATC 和 ACL 环境不包含在 Python 包中。

## 快速开始

以下流程假设已有待处理的 `torch.nn.Module`、一组模型输入和代表性校准数据。

### 1. 量化模型

```python
from mdc_llm_deploy.quantization import MinMaxConfig, quantize

config = MinMaxConfig(
    weight=True,
    activation=True,
    weight_granularity="per_channel",
    activation_granularity="per_tensor",
    weight_symmetric=True,
    activation_symmetric=True,
)

# model_inputs 与 calibration_batches 使用模型 forward 接收的参数名。
model.eval()
quantize(model, config, calibration_batches)
```

`quantize` 原地完成 prepare、calibrate 和 convert。需要控制生命周期、筛选目标 Linear
或恢复量化 checkpoint 时，参阅[量化文档](docs/quantization.md)。

### 2. 导出并转换 ONNX

```python
from pathlib import Path

import onnx
import torch

from mdc_llm_deploy.onnx import AdapterConfig, OnnxAdapter

raw_path = Path("output/model_qdq.onnx")
mdc_path = Path("output/model_mdc.onnx")
raw_path.parent.mkdir(parents=True, exist_ok=True)

program = torch.onnx.export(
    model,
    (),
    kwargs=model_inputs,
    dynamo=True,
    opset_version=21,
    optimize=False,
    external_data=True,
)
if program is None:
    raise RuntimeError("ONNX export did not return an ONNXProgram")
program.save(raw_path, external_data=True)

graph = onnx.load(raw_path, load_external_data=True)
OnnxAdapter(AdapterConfig())(graph)
onnx.save_model(
    graph,
    mdc_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=f"{mdc_path.name}.data",
    size_threshold=0,
)
```

`OnnxAdapter` 按固定顺序完成 QDQ lowering、opset 兼容处理、图规范化、算子融合、
schema 注册和检查。处理成功后原地返回同一 `ModelProto`；任一步失败时，输入模型保持
不变。完整契约见 [MDC ONNX 图处理](docs/onnx.md)。

> [!IMPORTANT]
> raw QDQ 支持范围大于 MDC lowering 支持范围。当前 lowering 面向静态 W8A8
> `MatMul`：权重必须是二维、静态、对称 INT8，激活必须是静态 INT8；不支持
> `Gemm`、动态 scale、group/block quant、INT4、FP8 或非对称权重。

## 量化能力

MinMax fake-quant 当前支持：

- 浮点 dtype：FP16、BF16、FP32；
- 权重：per-tensor 或 per-channel，对称或非对称 INT8；
- 激活：per-tensor 或 per-token，对称或非对称 INT8；
- weight-only、activation-only 和 W8A8 raw QDQ 导出；
- 量化参数随 `state_dict` 保存并严格恢复。

> [!WARNING]
> per-token 激活 scale 绑定校准时的 token 位置和长度。不同长度的 Prefill、Decode
> 需要分别校准和导出，或改用 per-tensor 激活量化。公共量化 API 当前只处理 Dense
> `nn.Linear`，不处理 Qwen3 MoE expert 的三维参数。

## ONNX 图处理

标准 ONNX 图可通过 `OnnxAdapter` 完成 MDC 转换。需要单独观察融合结果时，可直接运行
融合编排器：

```python
from mdc_llm_deploy.onnx import run_fusion_passes

report = run_fusion_passes(graph)
print(report.counts)
print(report.total_fused_count)
```

融合顺序固定为 RMSNorm、RoPE、FIA。独立编排器不是原子操作：后续 pass 失败时会保留
前序成功修改；需要失败回滚时使用 `OnnxAdapter`。

当前自定义算子 ABI：

- [ApplyRotaryPosEmb](docs/operators/ApplyRotaryPosEmb.md)
- [AscendDequant](docs/operators/AscendDequant.md)
- [AscendQuantV2](docs/operators/AscendQuantV2.md)
- [FusedInferAttentionScore](docs/operators/FusedInferAttentionScore.md)
- [RmsNorm](docs/operators/RmsNorm.md)

## 模型导出示例

### Qwen3-8B 单层 W8A8

加载首个 Transformer 层并缩小词表，完成 W8A8 量化及静态 Prefill/Decode 导出：

```bash
python examples/qwen3_8b_w8a8_export.py \
  --model /path/to/Qwen3-8B \
  --output-dir output/qwen3_8b_w8a8 \
  --vocab-size 1024
```

### Qwen3-8B 完整 FP16

导出全层、完整词表的静态分块 Attention 图；可限制层数进行快速验证：

```bash
python examples/qwen3_8b_fp16_export.py \
  --model /path/to/Qwen3-8B \
  --num-hidden-layers 2 \
  --output-dir output/qwen3_8b_fp16
```

### Qwen3-4B FP16 TP=2

串行生成两个 rank 的本地图，并注入 HCCL 风格通信节点：

```bash
python examples/qwen3_4b_fp16_tp_export.py \
  --model /path/to/Qwen3-4B \
  --num-hidden-layers 1 \
  --vocab-size 1024 \
  --chunk-size 1 \
  --kv-capacity 16 \
  --output-dir output/qwen3_4b_fp16_tp2
```

以上脚本也接受 Hugging Face 模型 ID。首次使用模型 ID 时需要下载 checkpoint。
完整参数、静态输入 ABI 和输出文件结构见[示例文档](examples/README.md)。

> [!CAUTION]
> Qwen3-30B-A3B 当前仅有微型结构测试，不代表完整 checkpoint 已完成端到端量化、
> ONNX 导出或真机部署。MC62/CANN 9.1.0 的已知 MoE 路由限制见
> [ONNX 文档](docs/onnx.md#已知验证边界)。

## 精度验证

`examples.qwen3_8b_layer_accuracy` 使用 Qwen3-8B 首个 DecoderLayer，对 FP16、
W8A8 per-token 和 W8A8 per-tensor 生成 Torch reference、raw QDQ ONNX、MDC ONNX
及精度报告：

```bash
python -m examples.qwen3_8b_layer_accuracy generate \
  --model /path/to/Qwen3-8B \
  --output-dir output/qwen3_8b_layer_accuracy
```

板端输出必须与相同量化配置的 Torch 输出比较，不能直接用 FP16 Torch 验收 W8A8。
当前 MDC 验收门槛为 cosine `>= 0.999`。比较命令和产物说明见
[示例文档](examples/README.md#qwen3-8b-单层精度验证)。

## 开发验证

```bash
python -m pytest
python -m ruff check .
python -m mypy mdc_llm_deploy
```

测试分为 `tests/unit`、`tests/integration` 和 `tests/hardware`。硬件目录中的测试主要
生成和校验部署制品，不代表本机自动执行 ATC、ACL 或真机推理。
