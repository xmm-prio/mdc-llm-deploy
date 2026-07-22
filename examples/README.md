# 示例

## Qwen3-8B 单层 W8A8 导出

`qwen3_8b_w8a8_export.py` 完成以下流程：

1. 从 `Qwen/Qwen3-8B` 加载 embedding、首个 Transformer 层、末尾 norm 和
   lm head 的预训练权重；
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
  --output-dir output\qwen3_8b_w8a8
```

默认输出：

```text
output/qwen3_8b_w8a8/
├── prefill.onnx
├── prefill.data
├── decode.onnx
└── decode.data
```

同名输出会被覆盖，不产生中间 ONNX 文件。prefill 输入长度为 3072。按
`export_for_generation` 的原生契约，decode 输入一个新 token 和长度 3072 的 KV cache，
attention 总长度为 3073。

脚本只实例化一层网络，但首次使用 Hugging Face 模型 ID 时仍需下载 Qwen3-8B checkpoint
分片。导出与 MDC 后处理也需要较大内存和磁盘空间。
