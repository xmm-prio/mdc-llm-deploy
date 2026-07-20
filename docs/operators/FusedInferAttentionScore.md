# FusedInferAttentionScore

源码：`mdc_llm_deploy/custom_ops/fused_infer_attention_score.py`。

## Torch CPU/CUDA 支持范围

当前 Torch 实现只覆盖 Qwen3 浮点 Prefill/Decode 子集：

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

CPU 使用 FP32 score/softmax。CUDA 使用在线 softmax Triton kernel，不回退到
PyTorch，额外要求 head dim 不超过 512 且安装 Triton。Q/K/V 含 NaN/Inf、
长度越界、mask 不可广播、GQA 不合法或有效 query 行全部被 mask 时明确报错。
算子仅用于推理，不支持 autograd。

## ONNX ABI 与导出收窄

默认 `ai.onnx` 域、opset 18。节点固定保留以下 29 个输入槽：

```text
query, key, value, pse_shift, atten_mask,
actual_seq_lengths, actual_seq_lengths_kv,
dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
antiquant_scale, antiquant_offset, block_table, query_padding_size,
kv_padding_size, key_antiquant_scale, key_antiquant_offset,
value_antiquant_scale, value_antiquant_offset, key_shared_prefix,
value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope,
key_rope_antiquant_scale, dequant_scale_query, learnable_sink
```

输出固定为 `attention_out, softmax_lse`。节点携带 14 个属性：
`num_heads`、`scale`、`pre_tokens`、`next_tokens`、`input_layout`、
`num_key_value_heads`、`sparse_mode`、`inner_precise`、`block_size`、
`antiquant_mode`、`softmax_lse_flag`、`key_antiquant_mode`、
`value_antiquant_mode`、`query_quant_mode`。

导出进一步收窄为：

- 仅 `BNSD`；
- 仅 Q/K/V、mask、两个有效长度槽可非空；
- `softmax_lse_flag` 必须为 false；
- 窗口、稀疏、PagedAttention 和量化属性保持 Torch 默认值。

任一保留槽被使用或属性越界时导出直接报错。29 槽 ABI 仍完整保留，未使用槽
序列化为空字符串。

## B 端确定性用例

`tests.hardware.custom_ops.fused_infer_attention_score` 生成 FLOAT32 BNSD GQA、
广播 mask、batch 有效长度用例，同时包含标准 ONNX、MDC ONNX、输入 bin 和
manifest。
