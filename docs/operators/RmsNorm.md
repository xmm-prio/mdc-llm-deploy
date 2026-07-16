# RmsNorm

## 名称

- GE 原名：`RmsNorm`
- ONNX OP：`NPURmsNorm`
- ONNX domain/opset：`ai.onnx::<opset>::NPURmsNorm`

## 源码映射与验证状态

- Python reference：`mdc_llm_deploy.operators.rms_norm`
- 集中 schema 键：`OPERATOR_SCHEMAS["RmsNorm"]`
- 源码函数使用 snake_case；GE 名称为 `RmsNorm`，ONNX `op_type` 固定为 `NPURmsNorm`。
- 本文定义待验收契约，不代表 GPU、NPU、parser、ATC 或真机已验证。B 端 parser/ATC 状态以 `docs/validation/b-side.md` 文本记录为准。

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


Atlas 推理系列不支持 `BFLOAT16`。

## 属性


| 名称        | ONNX 类型 | 默认值    | 说明                                |
| --------- | ------- | ------ | --------------------------------- |
| `epsilon` | `FLOAT` | `1e-6` | 在开平方前加到 `mean(x²)`，用于避免除零并提高数值稳定性 |


CANN 9.0 官方 ONNX parser 将 `NPURmsNorm` 映射到 GE `RmsNorm`，并将
`epsilon` 设为 `1e-6`。

`0.1.0` 只导出 `epsilon=1e-6`；模型配置不是该值时必须在导出前报错，不能静默改写。

## 前置条件与错误

- `x`、`gamma` 必须位于同一设备、dtype 相同，且 `gamma_shape = x_shape[n:]`。
- `epsilon` 必须是有限正数。
- 输入或 `gamma` 包含 NaN/Inf 时必须显式报错。
- 自定义算子不支持 autograd；反向传播必须抛出明确错误。

## 数值验收

CPU 与 GPU 分别和使用 FP32 累加的独立 PyTorch reference 对照。随机输入使用 seed `20260714`、正态分布 `N(0, 1)`，并覆盖全零、极小方差、最小合法 shape、非连续输入和 hidden size 64 的发布 shape。

| 输出 | dtype | `atol` | `rtol` |
| --- | --- | ---: | ---: |
| `y` | `FLOAT32` | `1e-5` | `1e-5` |
| `y` | `FLOAT16` | `2e-3` | `2e-3` |
| `y` | `BFLOAT16` | `2e-2` | `2e-2` |
| `rstd` | `FLOAT32` | `1e-5` | `1e-5` |

同时断言 `rstd` 使用 FP32、shape 等于所有未归一化前置维。