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

## Qwen3-8B 单层 FP16 分块 Attention 导出

`qwen3_8b_fp16_export.py` 导出不融合 FIA 的静态 prefill/decode 图。模型使用
`eager` Attention，图中保留 `MatMul`、`Softmax` 等小算子；RMSNorm 和 RoPE
仍执行 MDC 自定义算子融合。

两张图都接收固定容量为 32000 的 KV buffer，但不在图内写回。调用方把
`present_key`、`present_value` 写入 buffer：

- prefill：输入 2048 个 token、32000 KV buffer、长度 34048 的 mask；输出 logits
  和本次 2048 个 token 的 KV；
- decode：输入 1 个 token、32000 KV buffer、长度 32001 的 mask；输出 logits
  和本次 1 个 token 的 KV；
- mask 前 32000 位标记 KV buffer 有效区，末尾标记当前 token；图内据此计算
  position id。

导出真实单层 Qwen3-8B，并同时生成确定性输入和 PyTorch 参考：

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe examples\qwen3_8b_fp16_export.py export `
  --model C:\models\Qwen3-8B `
  --output-dir output\qwen3_8b_fp16_chunked `
  --validation-dir output\qwen3_8b_fp16_chunked\validation
```

默认保留前 1024 个词表项。输出 ONNX 内嵌权重，便于直接交给 ATC：

```text
output/qwen3_8b_fp16_chunked/
├── prefill.onnx
├── decode.onnx
├── atc_fusion_switch.json
└── validation/
    ├── manifest.json
    ├── prefill/
    │   ├── input/
    │   └── torch/
    └── decode/
        ├── input/
        └── torch/
```

`manifest.json` 记录输入、输出顺序及 dtype/shape。decode 验证输入默认把 PyTorch
prefill 产生的 2048 个 KV 写入 buffer 前部，保证两阶段输入一致。

ATC 编译时传入
`--fusion_switch_file=atc_fusion_switch.json`。该文件关闭 MC62 CANN 9.1.0 中两个
存在 ABI 问题的 vendor MatMul fusion，不改变 ONNX 语义。导出流程还会折叠静态 shape
子图；未经折叠的 Qwen3 图会触发该版本 ATC 在 `Reshape` 常量推导阶段崩溃。

MDC 推理输出拉回后，与 PyTorch 参考比较：

```powershell
.venv\Scripts\python.exe examples\qwen3_8b_fp16_export.py compare `
  --manifest output\qwen3_8b_fp16_chunked\validation\manifest.json `
  --stage prefill `
  --board-output output\mdc\prefill
```

比较覆盖 logits、`present_key`、`present_value`。默认要求输出有限、非全零且
cosine 不低于 `0.999`，同时打印最大和平均绝对误差。
