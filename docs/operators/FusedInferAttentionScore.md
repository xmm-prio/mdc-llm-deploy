# FusedInferAttentionScore

`FusedInferAttentionScore`（FIA）由 ONNX pipeline 的 attention fusion pass 生成。
输入必须是标准 ONNX 导出图，不依赖 Torch custom op、导出 profile 或
translation table。

## ONNX ABI

默认域、opset 18。schema ABI 冻结自 CANN `ops-transformer`：

- commit：`606a5ddb67c67d93c137a7b474fa7a5edd05f7c9`；
- 源文件：`attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp`。

schema 包含 31 个有序输入槽。前三个必选输入为 `query`、`key`、`value`，后续可选
槽依次为：

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

输出固定为 `attention_out`、`softmax_lse`。`num_heads` 为必填属性；其余属性和
默认值与冻结源码一致。schema 由 `mdc_llm_deploy.onnx.schema` 集中声明，
`OnnxAdapter` 按模型实际节点注册。

## 融合范围

FIA pass 识别 Transformers Qwen3 静态 attention 子图：

- 支持 eager 和 SDPA、Prefill 和真实 KV-cache Decode、MHA 和 GQA；
- Q/K/V 必须为静态 BNSD，dtype 为 FP16 或 BF16；
- additive mask 仅接受精确 `{0, -inf}` 或 `{0, finfo(dtype).min}`；
- 必须完整证明 scale、KV repeat、mask 广播和子图闭包；
- FP32 attention、有限 bias、ALiBi/PSE、dropout、动态 shape，以及 Q/K/V 直接为
  INT8 的量化 attention 保持原图；
- Linear projection 可使用 W8A8。projection 输出经 `AscendDequant` 恢复为 FP16
  或 BF16 后，后续 attention 子图仍可融合为 FIA。

```python
import onnx

from mdc_llm_deploy.onnx import AdapterConfig, OnnxAdapter

model = onnx.load("model.onnx")
OnnxAdapter(AdapterConfig())(model)
```

流程完成融合、按需 schema 注册和最终 ONNX checker 校验。新进程单独校验已生成模型
时，需先调用 `register_schemas("FusedInferAttentionScore")`。
