# ApplyRotaryPosEmb

## 名称

- GE 原名：`ApplyRotaryPosEmb`
- ONNX OP：`ApplyRotaryPosEmb`
- ONNX domain/opset：`ai.onnx::<opset>::ApplyRotaryPosEmb`

## 源码映射与验证状态

- Python reference：`mdc_llm_deploy.mdc_ops.operators.apply_rotary_pos_emb`
- 集中 schema 键：`OPERATOR_SCHEMAS["ApplyRotaryPosEmb"]`
- 源码函数使用 snake_case，GE 名称和 ONNX `op_type` 使用上方固定名称；三者不得互换。
- 本文定义待验收契约，不代表 GPU、NPU、parser、ATC 或真机已验证。B 端 parser/ATC 状态以 `docs/validation/b-side.md` 文本记录为准。

## ONNX OP 原型

```text
ApplyRotaryPosEmb(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    layout: int = 1,
    rotary_mode: string = "half"
) -> (
    query_out: Tensor,
    key_out: Tensor
)
```

GE `ApplyRotaryPosEmb` 对 `query` 和 `key` 采用输入输出复用语义；ONNX
节点使用显式的 `query_out`、`key_out` 表达更新结果。

## 输入

| 名称 | 必选 | 支持类型 | 格式 | Shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| `query` | 是 | `BFLOAT16`、`FLOAT16`、`FLOAT32` | `ND` | BSND 时为 `(B,S,Nq,D)`；TND 时为 `(T,Nq,D)` | Attention 的 Query；对最后一维应用旋转位置编码 |
| `key` | 是 | `BFLOAT16`、`FLOAT16`、`FLOAT32` | `ND` | BSND 时为 `(B,S,Nk,D)`；TND 时为 `(T,Nk,D)` | Attention 的 Key；与 `query` 使用相同位置编码 |
| `cos` | 是 | `BFLOAT16`、`FLOAT16`、`FLOAT32` | `ND` | N 维为 1，并可在 B 维广播 | 各 token、各旋转维度的余弦系数 |
| `sin` | 是 | `BFLOAT16`、`FLOAT16`、`FLOAT32` | `ND` | 与 `cos` 相同 | 各 token、各旋转维度的正弦系数 |

四个输入的数据类型必须一致。Atlas 推理系列不支持 `BFLOAT16`。

## 输出

| 名称 | 支持类型 | 格式 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `query_out` | 与 `query` 相同 | `ND` | 与 `query` 相同 | 应用 RoPE 后的 Query |
| `key_out` | 与 `key` 相同 | `ND` | 与 `key` 相同 | 应用 RoPE 后的 Key |

## 属性

| 名称 | ONNX 类型 | 默认值 | 支持值 | 说明 |
| --- | --- | --- | --- | --- |
| `layout` | `INT` | `1` | `1=BSND`、`2=SBND`、`3=BNSD`、`4=TND` | 指定 B、S/T、N、D 各维在输入中的排列方式 |
| `rotary_mode` | `STRING` | `"half"` | `"half"`、`"interleave"`、`"quarter"` | 指定最后一维的分组旋转方式：前后半区、相邻元素交错或四分区旋转 |

在 CANN 9.0 的 Atlas A2/A3 路径上通常仅支持 `BSND`/`TND` 和
`rotary_mode="half"`；其他布局和旋转模式取决于目标产品。

`0.1.0` 发布模型固定使用 `layout=1`（BSND）和 `rotary_mode="half"`；`onnx_export` 在 RoPE 后显式转置为 FusedInferAttentionScore 和 KV Cache 使用的 BNSD。首次发布候选必须先用目标 SoC/OPP 的最小 ATC 样例确认该组合可解析；不支持时发布验收失败，不允许静默切换 layout。

## 前置条件与错误

- `query`、`key`、`cos`、`sin` 必须位于同一设备并具有相同 dtype。
- `query` 与 `key` 的 head dim 必须相同且为偶数；`cos`、`sin` 必须能按声明 layout 广播到 token 和旋转维。
- 输入包含 NaN/Inf、layout 与 rank 不匹配或广播失败时必须显式报错。
- 自定义算子不支持 autograd；反向传播必须抛出明确错误。

## 数值验收

CPU 与 GPU 分别和独立 FP32 PyTorch reference 对照。随机输入使用 seed `20260714`、均匀分布 `[-2, 2]`，并覆盖全零、最小 shape、非连续输入和 3072-token 发布 shape。

| 输出 dtype | `atol` | `rtol` |
| --- | ---: | ---: |
| `FLOAT32` | `1e-5` | `1e-5` |
| `FLOAT16` | `1e-3` | `1e-3` |
| `BFLOAT16` | `2e-2` | `2e-2` |

`query_out`、`key_out` 必须分别满足阈值；shape、stride 无承诺，但 dtype 和逻辑值必须一致。
