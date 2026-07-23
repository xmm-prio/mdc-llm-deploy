# 示例

## Qwen3-8B 单层 W8A8 导出

`qwen3_8b_w8a8_export.py` 完成以下流程：

1. 从 `Qwen/Qwen3-8B` 加载首个 Transformer 层、末尾 norm，以及默认前 1024 个
   token 对应的 embedding 和 lm head 预训练权重；
2. 用一批固定随机 token 数据执行 W8A8、权重/激活 per-tensor 对称静态量化；
3. 使用 Transformers `OnnxExporter` 导出静态 prefill 和 decode 图；
4. 使用 `mdc_llm_deploy.onnx.process_onnx` 完成 MDC lowering 与图融合；
5. 将每张 ONNX 图及其权重分开保存。

从仓库根目录运行：

```powershell
.venv\Scripts\python.exe examples\qwen3_8b_w8a8_export.py
```

也可指定本地模型目录和输出目录：

```powershell
.venv\Scripts\python.exe examples\qwen3_8b_w8a8_export.py `
  --model C:\models\Qwen3-8B `
  --output-dir output\qwen3_8b_w8a8 `
  --vocab-size 1024
```

默认输出：

```text
output/qwen3_8b_w8a8/
├── prefill.onnx
├── prefill.onnx.data
├── decode.onnx
└── decode.onnx.data
```

同名输出会被覆盖，不产生中间 ONNX 文件。prefill 输入长度为 3072。按
`export_for_generation` 的原生契约，decode 输入一个新 token 和长度 3072 的 KV cache，
attention 总长度为 3073。

脚本只实例化一层网络，并缩减词表以便快速端到端验证，但首次使用 Hugging Face 模型
ID 时仍需下载 Qwen3-8B checkpoint 分片。`--vocab-size` 可调整保留的词表行数，不能
超过源模型词表大小。

运行时自动优先选择 CUDA；没有可用 GPU 时回退到 CPU。

## Qwen3-8B 完整 FP16 分块 Attention 导出

`qwen3_8b_fp16_export.py` 导出不融合 FIA 的静态 prefill/decode 图。模型使用
`eager` Attention，图中保留 `MatMul`、`Softmax` 等小算子；RMSNorm 和 RoPE
仍执行 MDC 自定义算子融合。脚本加载标准 Qwen3-8B 的全部 Transformer 层、完整词表
和原始预训练权重，不裁剪网络。

两张图都接收固定容量为 32000 的 KV buffer，但不在图内写回。调用方把
`present_key`、`present_value` 写入 buffer：

- KV 张量包含层维度，布局为
  `[num_hidden_layers, batch, num_key_value_heads, capacity, head_dim]`；
- prefill：输入 2048 个 token、全层 32000 KV buffer、长度 34048 的 mask；输出
  logits 和所有层本次 2048 个 token 的 KV；
- decode：输入 1 个 token、全层 32000 KV buffer、长度 32001 的 mask；输出 logits
  和所有层本次 1 个 token 的 KV；
- mask 前 32000 位标记 KV buffer 有效区，末尾标记当前 token；图内据此计算
  position id。

导出完整 Qwen3-8B：

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe examples\qwen3_8b_fp16_export.py `
  --model C:\models\Qwen3-8B `
  --output-dir output\qwen3_8b_fp16_chunked
```

完整模型权重超过 ONNX 单文件限制，因此每张图使用一个外部权重文件：

```text
output/qwen3_8b_fp16_chunked/
├── prefill.onnx
├── prefill.onnx.data
├── decode.onnx
└── decode.onnx.data
```
