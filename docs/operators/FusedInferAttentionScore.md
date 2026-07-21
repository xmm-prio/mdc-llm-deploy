# FusedInferAttentionScore

源码：`mdc_llm_deploy/custom_ops/fused_infer_attention_score/`。包内按 Torch 契约、
执行 kernel、FakeTensor、ONNX 窄契约和插件注册分层。导入此算子包只注册 FIA
的 Torch op；创建 ONNX profile 时才注册进程本地 schema。

## Torch CPU/CUDA 支持范围

Torch schema 与 MC62 ONNX ABI 相互独立。Torch 实现覆盖 Qwen3 浮点
Prefill/Decode 子集：

- Q/K/V 均为 rank 4、同设备、同 dtype；支持 `FLOAT16`、`BFLOAT16`、
  `FLOAT32`。
- 仅支持 `BNSD`、`BSND`。Q/K/V batch 与 head dim 相同，K/V shape 必须相同。
  `num_heads` 必须等于 Q head 数；`num_key_value_heads=0` 表示使用实际 KV head
  数，否则必须与其相等；Q head 数必须可整除 KV head 数。
- 可选输入只支持 `atten_mask`、`actual_seq_lengths`、
  `actual_seq_lengths_kv`。mask 支持 BOOL/INT8/UINT8，并可广播到
  `[B,Nq,Sq,Skv]`；非零表示屏蔽。长度为 INT64，可含 1 个值或每 batch 一个值。
- 支持 `softmax_lse_flag`。false 时第二输出为 FLOAT32 `[1]` 零张量；true 时为
  `[B,Nq,Sq,1]`。
- `pre_tokens/next_tokens` 必须保持 `2147483647`；`sparse_mode`、
  `inner_precise`、`block_size`、四个量化 mode 必须为 0。
- PSE、PagedAttention、padding、Attention 量化/反量化、共享前缀、MLA RoPE、
  learnable sink 均不属于 Torch 执行范围，传入会报 `NotImplementedError`。

CPU/CUDA 均使用 FP32 score/softmax，输出转换回输入 dtype。Q/K/V 含 NaN/Inf、
长度越界、mask 不可广播、GQA 不合法或有效 query 行全部被 mask 时明确报错。
算子仅用于推理，不支持 autograd；FakeTensor 与 `torch.compile(fullgraph=True)`
复用 Torch 宽契约，不受 ONNX Decode 限制。

## ONNX ABI 与导出收窄

默认 `ai.onnx` 域、opset 18。MC62 arch38 的浮点路径只导出 Decode FIA：

- Q/K/V 必选，均为 rank 4 `BNSD`，dtype 仅允许 `FLOAT16` 或
  `BFLOAT16`，且三者必须相同；
- query 序列长度必须静态等于 1；
- 不接受 mask、实际长度、PSE、量化参数及其它任何可选输入；
- `softmax_lse_flag` 必须为 false；
- 窗口、稀疏、PagedAttention 和量化属性必须保持 Torch 默认值。

FIA 节点仅传 Q/K/V 三个输入，仅产生 `attention_out` 一个输出，仅携带
`num_heads`、`scale`、`input_layout`、`num_key_value_heads` 四个属性。
Torch API 仍返回二元组；第二个 `softmax_lse` 在 ONNX 图中由独立
`Constant` 节点提供 `[1]` FP32 零张量，不占用 FIA 的输出。

仅支持 Dynamo 导出。调用方先执行
`create_onnx_export_profile("fused_infer_attention_score")`，再把 profile 的
`custom_translation_table` 传给 `torch.onnx.export(..., dynamo=True,
opset_version=18)`。默认域 `OpSchema` 仅存在于当前进程；模型不携带该 schema，
新进程运行 `onnx.checker` 前需重新创建 profile。

Torch 完整槽位顺序保留在算子包契约元数据中：Q/K/V 位于 0-2，
四个 optional 位于 3-6，`dequant_scale1/quant_scale1/dequant_scale2`
位于 7-9，`quant_scale2/quant_offset2` 位于 10-11。浮点 Decode 不使用
量化槽，因此不填 trailing empty。

MC62 arch38 的 PFA 浮点 Prefill（query 序列长度大于 1）不可用。此类图应保留
小算子；只有 Q/K/V/score 均为 per-tensor INT8 时，才能由目标适配流程生成
fully-int8 FIA。直接通过此 Torch custom op 导出 float Prefill、动态序列或可选
mask 会明确报错，避免进入 CANN 的 arch38 PFA dummy 内核。

## B 端确定性用例

`tests.hardware.custom_ops.fused_infer_attention_score` 对齐参考 `build_fia`：
FP16 BNSD，B=1，heads=8，query seq=1，KV seq=16，head dim=64，MHA，
无 mask/有效长度。custom ONNX 中 FIA 节点为三输入、单输出；golden 与用例
manifest 只暴露最终 attention 输出。
