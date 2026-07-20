# AscendQuantV2

## 名称

- ONNX OP：`NPUAscendQuantV2`
- ONNX domain/opset：`ai.onnx::<opset>::NPUAscendQuantV2`

## 源码映射

- Python reference：`mdc_llm_deploy.operators.ascend_quant_v2`
- 集中 schema 键：`OPERATOR_SCHEMAS["AscendQuantV2"]`
- 源码函数使用 snake_case，ONNX `op_type` 固定为 `NPUAscendQuantV2`。

## ONNX OP 原型

```text
NPUAscendQuantV2(
    x: Tensor,
    scale: Tensor,
    offset: Tensor?,
    axis: int = -1,
    dtype: int = 2
) -> y: Tensor
```

## 输入

| 名称 | 必选 | 支持类型 | 格式 | 说明 |
| --- | --- | --- | --- | --- |
| `x` | 是 | `FLOAT16`、`FLOAT32`、`BFLOAT16` | `ND` | 待量化数据 |
| `scale` | 是 | `FLOAT16`、`FLOAT32`、`BFLOAT16` | `ND` | 量化缩放因子 |
| `offset` | 否 | `FLOAT16`、`FLOAT32`、`BFLOAT16` | `ND` | 量化偏移；省略时按零处理 |

`x`、`scale`、`offset` 的类型应保持一致。`scale` 和 `offset` 在 `axis`
对应维度上的长度应等于 `x` 对应维度或为 1。

## 输出

| 名称 | 支持类型 | 格式 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `y` | `INT8`、`INT4`、`HIFLOAT8`、`FLOAT8_E5M2`、`FLOAT8_E4M3FN` | `ND` | 与 `x` 相同 | 对 `x` 应用 `scale`、`offset`、舍入和类型转换后的量化结果 |

## 属性

| 名称 | ONNX 类型 | 默认值 | 支持值/说明 |
| --- | --- | --- | --- |
| `axis` | `INT` | `-1` | 指定按元素应用 `scale`、`offset` 的维度；负值从尾维开始计数，`-1` 表示最后一维 |
| `dtype` | `INT` | `2` | 指定 `y` 的目标类型；`2` 表示 `INT8`，`29` 表示 `INT4` |

`dtype=2`（INT8）时，算子按
`y = clamp(round(x * scale) + offset, -128, 127)` 计算，其中 `round`
为 ties-to-even；这里的乘法 `scale` 是仿射量化 scale 的倒数，
`offset` 是浮点表示的整数 zero point。

## 前置条件与错误

- `x`、`scale`、`offset` 必须位于同一设备并具有相同浮点 dtype；`offset` 省略时按同 dtype 的零处理。
- `scale` 必须有限且严格大于零；输入、scale 或 offset 包含 NaN/Inf 时必须显式报错。
- `axis` 规范化后必须位于输入 rank 内，scale/offset shape 必须满足 axis 广播规则。
- 自定义算子不支持 autograd；反向传播必须抛出明确错误。
