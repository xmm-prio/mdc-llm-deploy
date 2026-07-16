# MDC LLM Deploy 产品需求

## 1. 目标

本项目提供面向 MDC 部署的 Transformers 兼容压缩与导出链路。当前版本交付：

- 推理与导出专用的 Qwen3 Dense、Qwen3-MOE 模型；
- 本地目录和 Hugging Face Hub checkpoint 加载；
- 单文件及分片 safetensors 加载；
- prefill FX、单 token decode FX 与 MDC ONNX 导出；
- Dense/MoE、浮点与受支持量化组合；
- 多层独立 KV Cache ABI。

模型实现参考 Transformers 5.9.0 语义，但不继承官方模型类。

## 2. 非目标

- 不提供训练或自定义算子 autograd；
- 不在运行时改变序列长度、mask 或位置编码语义；
- 不承诺 GPTQ 支持 packed `MoeExpert` 权重；
- 不在 A 端执行 ATC 或保留 B 端编译产物；
- 不引入新的 `ir/` 或 `protocol/` 分层。

## 3. 模块职责

- `models/`：Qwen3 配置、模型、checkpoint 解析与权重加载；
- `mdc_ops/`：MDC 自定义算子的 eager、meta 与 ONNX symbolic；
- `export/`：PyTorch 捕获、边界发现、量化和 decode 改写；
- `onnx_export/`：标准 ONNX 转换、MDC 算子映射、验证和写盘；
- `quantization/`：MinMax、GPTQ 及量化元数据；
- `tools/release_matrix.py`：发布组合构造与本地 ONNX 生成。

模块之间通过公开配置、FX metadata 和 operator schema 协作，不依赖模型类名判断。

## 4. 模型 API

公开模型入口：

- `ExportModelConfig`
- `Qwen3Config`
- `Qwen3MoeConfig`
- `Qwen3ForCausalLM`
- `Qwen3MoeForCausalLM`
- `AutoExportModel`

`ExportModelConfig` 固定序列长度和 `causal`/`none` mask 语义。cos/sin buffer 与可选 BOOL 因果 mask 在初始化阶段生成。模型输出顺序固定为 logits，随后按层输出 K/V。

`AutoExportModel.from_pretrained()` 读取 checkpoint 的 `config.json`，构建对应模型并加载 safetensors。MoE loader 将每层专家的 gate/up/down 权重按固定顺序打包为 expert-major rank-2 tensor。

## 5. MoeExpert ABI

输入顺序：

1. token hidden states；
2. top-k expert id；
3. top-k routing weight；
4. expert-major packed 权重；
5. 可选 quant scale；
6. 可选 quant offset。

packed 权重 shape 为 `[expert_count, packed_weights_per_expert]`。每个 expert 行按 gate、up、down 顺序存放。expert 数和 top-k 均由输入 shape 推导，不固定 shared expert。

浮点权重不得携带量化参数。INT8 权重要求每个 expert 的三个 projection 均有 scale，offset 可缺省。

## 6. FX 与 Decode

`export()` 捕获 eval 模型并固化：

- 图阶段与模型类型；
- 输入输出 ABI；
- 模块边界；
- 模型尺寸与 mask 语义；
- 量化目标。

Dense RMSNorm、RoPE 和 Attention 保持标准 PyTorch 数据流，导出阶段按边界映射为 MDC 算子。Qwen3-MOE 直接保留 `MoeExpert` custom op。

`convert_to_decode()` 原地将 prefill 图改写为单 token decode 图：

- `input_ids` shape 变为 `[1, 1]`；
- 每层新增 `past.N.key`、`past.N.value`；
- 每层输出 `present.N.key`、`present.N.value`；
- 使用 prefill 最后位置的 cos/sin；
- 每层 cache 独立拼接，不复用量化目标。

KV Cache layout 固定为 BNSD。

## 7. ONNX API

公开签名：

```python
onnx_export(
    graph,
    output_path,
    *,
    external_data=True,
)
```

ONNX 层不接收 `mask_mode` 或 `overwrite`。语义来自 FX metadata；同名产物始终原子覆盖。

`external_data=True` 时，外部数据固定写入 `<model>.onnx.data`。`external_data=False` 时权重内联，并清理同名旧 external data。失败不得留下临时模型或数据文件。

validator 只验证：

- protobuf/ONNX 基本结构；
- 输入输出名字唯一；
- domain 与 opset；
- MDC custom op 的输入、输出和 schema。

validator 不识别 Qwen 类名，不验证模型拓扑。

## 8. 量化边界

- Dense linear：支持现有 MinMax 与 GPTQ 能力；
- Attention：支持 query/key/value/score activation metadata；
- Packed MoE：MinMax 以单个 expert-major 参数为目标；
- Packed MoE GPTQ：明确拒绝，保持 GPTQ 现有矩阵输入能力边界；
- 模型可直接安装验证过的 INT8 packed 权重及量化参数。

量化失败必须保持原 FX 图不变。

## 9. 发布矩阵

发布维度包括：

- Dense / MoE；
- prefill / decode；
- 浮点 / 受支持量化配置；
- fp16 / fp32。

mask 不再是 `onnx_export` 参数或重复矩阵维度；每个模型实例通过 `ExportModelConfig` 固定语义。

至少使用两层小配置覆盖：

- prefill ONNX；
- decode ONNX；
- 每层 KV 名称、shape 与顺序；
- external data 可访问性；
- 每层 FIA/RoPE/RMSNorm 数量；
- MoE 浮点与 INT8 ABI。

## 10. 安装与打包

用户安装方式唯一：

```bash
pip install -e .
```

运行与转换依赖由 `pyproject.toml` 管理下限。`requirements.txt` 仅记录验收环境的精确快照。

wheel 不得包含 Tiny 生产模型、测试 fixture、临时 checkpoint、ONNX、external data、缓存或构建目录。

## 11. B 端验收

A 端只发送提交 SHA、验证矩阵与命令。B 端从远端拉取指定提交，在 B 端生成 ONNX 并执行 ATC，不修改或上传代码。

每个矩阵项返回：

- 提交 SHA；
- ATC 版本与完整命令；
- 退出码；
- 最短决定性日志；
- `.om`、ONNX 和 external data 检查结果。

若 ABI 明确失败，A 端修复、提交、推送后重新验证；认证、权限或环境故障作为外部阻塞单独报告。
