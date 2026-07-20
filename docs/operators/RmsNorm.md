# RmsNorm

## 名称

- ONNX OP：`NPURmsNorm`
- ONNX domain/opset：`ai.onnx::<opset>::NPURmsNorm`

## 源码映射

- Python reference：`mdc_llm_deploy.operators.rms_norm`
- 集中 schema 键：`OPERATOR_SCHEMAS["RmsNorm"]`
- 源码函数使用 snake_case，ONNX `op_type` 固定为 `NPURmsNorm`。

## ONNX OP 原型

```text
NPURmsNorm(
    x: Tensor,
    gamma: Tensor,
    epsilon: float = 1e-6
) -> (
    y: Tensor,
    rstd: Tensor
)
```

## 输入


| 名称      | 必选  | 支持类型                           | 格式   | Shape        | 说明                             |
| ------- | --- | ------------------------------ | ---- | ------------ | ------------------------------ |
| `x`     | 是   | `FLOAT32`、`FLOAT16`、`BFLOAT16` | `ND` | 1～8 维        | 待归一化张量；在 `gamma` 对应的尾部维度上计算均方根 |
| `gamma` | 是   | `FLOAT32`、`FLOAT16`、`BFLOAT16` | `ND` | `x` 的一个或多个尾维 | 归一化后的逐元素缩放权重                   |


`x` 与 `gamma` 的数据类型应一致，并满足
`gamma_shape = x_shape[n:]`。

## 输出


| 名称     | 支持类型                           | 格式   | Shape               | 说明                                          |
| ------ | ------------------------------ | ---- | ------------------- | ------------------------------------------- |
| `y`    | `FLOAT32`、`FLOAT16`、`BFLOAT16` | `ND` | 与 `x` 相同            | `x * rstd * gamma` 的归一化结果                   |
| `rstd` | `FLOAT32`                      | `ND` | 保留 `x` 中未参与归一化的前置维度 | `1 / sqrt(mean(x²) + epsilon)`，供反向计算或后续融合使用 |

## 属性


| 名称        | ONNX 类型 | 默认值    | 说明                                |
| --------- | ------- | ------ | --------------------------------- |
| `epsilon` | `FLOAT` | `1e-6` | 在开平方前加到 `mean(x²)`，用于避免除零并提高数值稳定性 |

## 前置条件与错误

- `x`、`gamma` 必须位于同一设备、dtype 相同，且 `gamma_shape = x_shape[n:]`。
- `epsilon` 必须是有限正数。
- 输入或 `gamma` 包含 NaN/Inf 时必须显式报错。
- 自定义算子不支持 autograd；反向传播必须抛出明确错误。