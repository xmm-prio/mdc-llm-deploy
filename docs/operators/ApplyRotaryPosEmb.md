# ApplyRotaryPosEmb

## 名称

- ONNX OP：`ApplyRotaryPosEmb`
- ONNX domain/opset：`ai.onnx::<opset>::ApplyRotaryPosEmb`

## 源码映射

- Python reference：`mdc_llm_deploy.operators.apply_rotary_pos_emb`
- 集中 schema 键：`OPERATOR_SCHEMAS["ApplyRotaryPosEmb"]`
- 源码函数使用 snake_case，ONNX `op_type` 使用上方固定名称。

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

## 输入

| 名称 | 必选 | 支持类型 | 格式 | Shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| `query` | 是 | `BFLOAT16`、`FLOAT16`、`FLOAT32` | `ND` | BSND 时为 `(B,S,Nq,D)`；TND 时为 `(T,Nq,D)` | Attention 的 Query；对最后一维应用旋转位置编码 |
| `key` | 是 | `BFLOAT16`、`FLOAT16`、`FLOAT32` | `ND` | BSND 时为 `(B,S,Nk,D)`；TND 时为 `(T,Nk,D)` | Attention 的 Key；与 `query` 使用相同位置编码 |
| `cos` | 是 | `BFLOAT16`、`FLOAT16`、`FLOAT32` | `ND` | N 维为 1，并可在 B 维广播 | 各 token、各旋转维度的余弦系数 |
| `sin` | 是 | `BFLOAT16`、`FLOAT16`、`FLOAT32` | `ND` | 与 `cos` 相同 | 各 token、各旋转维度的正弦系数 |

四个输入的数据类型必须一致。

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

## 前置条件与错误

- `query`、`key`、`cos`、`sin` 必须位于同一设备并具有相同 dtype。
- `query` 与 `key` 的 head dim 必须相同且为偶数；`cos`、`sin` 必须能按声明 layout 广播到 token 和旋转维。
- 输入包含 NaN/Inf、layout 与 rank 不匹配或广播失败时必须显式报错。
- 自定义算子不支持 autograd；反向传播必须抛出明确错误。
