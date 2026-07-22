# MinMax 量化

`mdc_llm_deploy.quantization` 提供原地 MinMax INT8 fake-quant 流程。当前目标是
`torch.nn.Linear`；模型仍保留浮点权重，转换后用冻结量化参数模拟 INT8 数值，适合推理和
ONNX QDQ 导出。

## 基本流程

三阶段 API 适合显式控制生命周期：

```python
from mdc_llm_deploy.quantization import MinMaxConfig, calibrate, convert, prepare

config = MinMaxConfig(
    weight=True,
    activation=True,
    weight_granularity="per_channel",
    activation_granularity="per_tensor",
)
batches = [
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
]

prepare(model, config)
calibrate(model, batches)
convert(model)
```

一步式 API 执行相同流程：

```python
from mdc_llm_deploy.quantization import quantize

quantize(model, config, batches)
```

所有生命周期 API 都原地修改并返回同一模型。激活量化开启时，校准批次必须覆盖每个目标
Linear，否则 `convert` 严格失败。仅启用权重量化时可以传空批次。

校准默认使用 Rich 显示 batch 进度。CI、服务进程或需要自行管理终端输出时可关闭：

```python
calibrate(model, batches, show_progress=False)
quantize(model, config, batches, show_progress=False)
```

库使用 Python `logging` 记录量化和 ONNX 处理的阶段、耗时及汇总信息。首次导入
`mdc_llm_deploy` 时自动通过 Rich 启用彩色日志，默认打印 `INFO` 及以上级别。若应用、
测试框架或服务框架已配置 root handler，库会保留现有配置，不添加或替换 handler；
需要定制格式、输出位置或级别时，应在导入本库前完成应用级日志配置。

阶段边界使用 `INFO`，未验证的依赖版本使用 `WARNING`，模块名称等诊断细节使用 `DEBUG`。

## 目标筛选

`TargetSelector` 使用模块全限定名 glob。`exclude` 优先于 `include`：

```python
from mdc_llm_deploy.quantization import MinMaxConfig, TargetSelector

config = MinMaxConfig(
    targets=TargetSelector(
        include=("model.layers.*",),
        exclude=("*.self_attn.o_proj",),
    )
)
```

共享 Linear 的所有别名必须得到一致选择。根模块本身是 Linear 时不能原地替换。

## 支持矩阵

权重和激活可独立启用，但不能同时关闭。

| 维度 | 支持值 |
| --- | --- |
| 浮点 dtype | FP16、BF16、FP32 |
| 权重量化粒度 | per-tensor、per-channel |
| 激活量化粒度 | per-tensor、per-token |
| 权重量化范围 | 对称、非对称 INT8 |
| 激活量化范围 | 对称、非对称 INT8 |
| ONNX 表示 | 标准 `QuantizeLinear` / `DequantizeLinear`，opset 21 |

支持集合共 24 种配置：4 种 weight-only、4 种 activation-only、16 种 W8A8；三个
浮点 dtype 共形成 72 个 raw QDQ 导出案例。

### per-token 限制

per-token 激活量化按校准时固定 token 位置冻结 scale。转换后输入 rank 或 token 长度变化
会报错。因此该模式不支持同一模型在 generation prefill 与 decode 间切换不同 token
长度。需要多形状 generation 时，应使用 per-tensor 激活量化或分别构建固定形状模型。

## checkpoint 恢复

量化模型的 `state_dict` 包含浮点参数与持久化的 scale/zero-point。恢复时先创建相同浮点
结构，再直接重建 converted wrapper；无需重新校准：

```python
from mdc_llm_deploy.quantization import load_quantized_state_dict

restored = build_model()
load_quantized_state_dict(restored, config, checkpoint)
```

恢复执行严格 key、dtype、shape 和数值校验。配置必须与 checkpoint 的量化方式一致。
该 API 不接管 `save_pretrained` 或 `from_pretrained`。

## ONNX QDQ 导出

量化模型使用 opset 21 导出，并关闭 exporter 图优化：

```python
program = torch.onnx.export(
    model,
    (),
    kwargs=model_inputs,
    dynamo=True,
    opset_version=21,
    optimize=False,
    external_data=False,
)
```

也可将相同模型交给 Transformers `OnnxExporter`，配置
`OnnxConfig(opset_version=21, optimize=False, dynamic=False, external_data=False)`。

opset 21 用于表达对称 INT8 `QuantizeLinear`：省略 zero-point，并通过
`output_dtype=INT8` 明确输出类型。非对称量化则携带 INT8 zero-point。现有浮点导出继续
使用 opset 18，不受量化导出影响。

QDQ 实现记录其已验证的 Torch 版本。当前环境版本不同时会输出 `WARNING` 并继续运行，
不会仅因版本字符串不同而拒绝导出；若实际 API 或图契约不兼容，仍会在准确位置抛出异常。

## MDC lowering 边界

raw QDQ 导出支持上述完整矩阵。`process_onnx` 当前只 lowering 其中的 W8A8 子集：

- 权重必须对称；
- 权重可 per-tensor 或 per-channel；
- 激活可 per-tensor 或 per-token，可对称或非对称；
- 浮点 dtype 只支持 FP16、FP32，不支持 BF16；
- weight-only、activation-only 和非对称权重会明确失败，且输入模型保持不变。

满足边界时，标准 QDQ 会转换为 MDC `NPUAscendQuantV2` 与 `AscendDequant`，图中不再残留
标准 QDQ 节点。

`process_onnx` 默认显示各处理阶段的 Rich 进度。非交互环境可调用
`process_onnx(model, show_progress=False)`，阶段日志不受该开关影响。

## 非目标

当前只支持 Dense `nn.Linear`。不处理 Qwen3 MoE experts 的 3D 参数，不提供 MoE 端到端
量化，不执行硬件验证或质量基准，也不提供独立 ONNX 文件写入 API。
