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

默认 `ai.onnx` 域、opset 18。schema ABI 冻结自 CANN 开源
`CANN/ops-transformer` master：

- commit：`606a5ddb67c67d93c137a7b474fa7a5edd05f7c9`；
- 源文件：`attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp`；
- 来源：
  `https://gitcode.com/cann/ops-transformer/blob/606a5ddb67c67d93c137a7b474fa7a5edd05f7c9/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp`。

冻结 schema 包含 31 个有序输入槽。前三个必选输入为 `query`、`key`、`value`；
后续可选槽依次为：

```text
pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
antiquant_scale, antiquant_offset, block_table, query_padding_size,
kv_padding_size, key_antiquant_scale, key_antiquant_offset,
value_antiquant_scale, value_antiquant_offset, key_shared_prefix,
value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope,
key_rope_antiquant_scale, dequant_scale_query, learnable_sink,
q_start_idx, kv_start_idx
```

输出固定为 `attention_out`、`softmax_lse`。属性为 `num_heads`（必填），以及
`scale`、`pre_tokens`、`next_tokens`、`input_layout`、
`num_key_value_heads`、`sparse_mode`、`inner_precise`、`block_size`、
`antiquant_mode`、`softmax_lse_flag`、`key_antiquant_mode`、
`value_antiquant_mode`、`query_quant_mode`、`pse_type`、`out_dtype`。
默认值与冻结源码一致，其中 `input_layout="BSH"`、`inner_precise=1`。

当前 custom-op Dynamo translation 仍只接受 MC62 arch38 浮点 Decode 子集：

- Q/K/V 必选，均为 rank 4 `BNSD`，dtype 仅允许 `FLOAT16` 或
  `BFLOAT16`，且三者必须相同；
- query 序列长度必须静态等于 1；
- 不接受 mask、实际长度、PSE、量化参数及其它任何可选输入；
- `softmax_lse_flag` 必须为 false；
- 窗口、稀疏、PagedAttention 和量化属性必须保持 Torch 默认值。

FIA 节点可省略未使用的尾部可选输入，因此当前 translation 只传 Q/K/V；节点按
冻结 ABI 产生两个输出。translation 显式提供 `num_heads`、`scale`、
`input_layout`、`num_key_value_heads`，并为兼容现有 Torch profile 显式保留
`inner_precise=0`；ONNXScript 会把其余 schema 默认属性实体化到节点。首输出
通过一个可被优化为 `Cast` 的 `CastLike` 补齐序列化类型元数据。
`softmax_lse_flag=false` 时 Torch API 要求第二项为 `[1]` FP32 零张量，因此
对外返回值仍由独立 `Constant` 提供；FIA 的第二输出只用于满足完整 ABI，不暴露
为模型输出。

仅支持 Dynamo 导出。调用方先执行
`create_onnx_export_profile("fused_infer_attention_score")`，再把 profile 的
`custom_translation_table` 传给 `torch.onnx.export(..., dynamo=True,
opset_version=18)`。默认域 `OpSchema` 仅存在于当前进程；模型不携带该 schema。
新进程运行 `onnx.checker` 前需重新创建 profile，或显式调用
`register_schemas("FusedInferAttentionScore")`。

MC62 arch38 的 PFA 浮点 Prefill（query 序列长度大于 1）不可用。此类图应保留
小算子；只有 Q/K/V/score 均为 per-tensor INT8 时，才能由目标适配流程生成
fully-int8 FIA。直接通过此 Torch custom op 导出 float Prefill、动态序列或可选
mask 会明确报错，避免进入 CANN 的 arch38 PFA dummy 内核。

## ONNX 图融合

`run_fusion_passes` 与 `process_onnx` 可从 Transformers 5.14.0 的 Qwen3 静态
浮点导出图识别 eager/SDPA attention，覆盖 Prefill 与真实 KV-cache Decode、
MHA/GQA。融合要求 Q/K/V 为静态 BNSD，dtype 为 FP16 或 BF16，并能完整证明
scale、KV repeat、mask 广播和子图闭包。

additive mask 仅接受精确 `{0, -inf}` 或 `{0, finfo(dtype).min}`，融合时转换为
BOOL 屏蔽语义。任意有限 bias、ALiBi/PSE、dropout、动态 shape、量化 attention
和 FP32 attention 均安全保留原图；其中 FP32 的 FIA 命中数固定为 0。融合节点按
冻结 ABI 生成 31 个输入槽和 2 个输出，未使用的 optional 输入槽为空字符串。

## B 端确定性用例

`tests.hardware.custom_ops.fused_infer_attention_score` 对齐参考 `build_fia`：
FP16 BNSD，B=1，heads=8，query seq=1，KV seq=16，head dim=64，MHA，
无 mask/有效长度。custom ONNX 中 FIA 节点为三输入、双输出；golden 与用例
manifest 只暴露最终 attention 输出。
