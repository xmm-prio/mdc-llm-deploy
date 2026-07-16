# MoeExpert

## 契约

```text
MoeExpert(
    x: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    expert_weights: Tensor,
    quant_scales: Tensor? = null,
    quant_offsets: Tensor? = null
) -> out: Tensor
```

- `x`：浮点 `[token_count, hidden_size]`。
- `topk_ids`：INT32/INT64 `[token_count, top_k]`。
- `topk_weight`：与 `x` 同 dtype、同 routing shape；每行非负且和为 1。
- `expert_weights`：expert-major rank-2
  `[expert_count, 3 * hidden_size * intermediate_size]`。
- 每个 expert 行内固定按 `gate_proj`、`up_proj`、`down_proj` 的 row-major 数据排列。
- `top_k` 和 `expert_count` 均由输入 shape 推导，不固定 shared expert。

浮点权重不传量化参数。INT8 权重必须传 `quant_scales`，顺序为每个 expert 的
`gate, up, down`；`quant_offsets` 可省略，省略表示对称量化。

## 语义

每个 expert 计算：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

结果按 `topk_ids` 和 `topk_weight` 加权求和。输出 shape 和 dtype 与 `x` 一致。

算子仅支持推理，不提供 autograd。routing id 越界、同 token 重复、packed width
不合法、量化参数数量不匹配或跨设备输入必须显式失败。

## ONNX

- GE/ONNX 名称：`MoeExpert`
- opset：18
- schema 真源：`mdc_llm_deploy.operators.contracts.schema.OPERATOR_SCHEMAS`
- 浮点 FX 保留上述通用契约；量化 ONNX 在映射层适配当前 ATC ABI：
  `x` 转为 INT8、`topk_ids` 转为 INT16、权重展平为一维。
- ATC 量化 scale 顺序为一个输入激活 scale，随后按 expert 依次排列
  `gate`、`up`、中间激活、`down`，总长度为 `1 + expert_count * 4`。

ATC 支持范围以 B 端针对指定提交返回的工具版本、命令、退出码和决定性日志为准。
验证结果只记录已实际执行的 dtype、expert 数和 top-k 组合。
