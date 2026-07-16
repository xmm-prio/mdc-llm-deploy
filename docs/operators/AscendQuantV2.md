# AscendQuantV2

## 名称

- GE 原名：`AscendQuantV2`
- ONNX OP：`NPUAscendQuantV2`
- ONNX domain/opset：`ai.onnx::<opset>::NPUAscendQuantV2`

## 源码映射与验证状态

- Python reference：`mdc_llm_deploy.operators.ascend_quant_v2`
- 集中 schema 键：`OPERATOR_SCHEMAS["AscendQuantV2"]`
- 源码函数使用 snake_case；GE 名称为 `AscendQuantV2`，ONNX `op_type` 固定为 `NPUAscendQuantV2`。
- 本文定义待验收契约，不代表 GPU、NPU、parser、ATC 或真机已验证。B 端 parser/ATC 状态以 `docs/validation/b-side.md` 文本记录为准。

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

具体输出类型取决于 `dtype` 和目标产品。Atlas 推理系列仅支持 `INT8`；
Atlas A2/A3 支持 `INT8`、`INT4`；FP8/HIFLOAT8 仅在支持这些类型的产品上可用。

## 属性

| 名称 | ONNX 类型 | 默认值 | 支持值/说明 |
| --- | --- | --- | --- |
| `axis` | `INT` | `-1` | 指定按元素应用 `scale`、`offset` 的维度；负值从尾维开始计数，`-1` 表示最后一维 |
| `dtype` | `INT` | `2` | 指定 `y` 的目标类型。使用 GE 数据类型编码；`2` 表示 `INT8`，`29` 表示 `INT4`，其他编码以目标 CANN 9.0 OPP 为准 |

CANN 9.0 官方 ONNX parser 只从 `NPUAscendQuantV2` 节点解析 `axis` 和
`dtype`，并映射到 GE `AscendQuantV2`。

`0.1.0` 的 MDC ONNX 发布路径只使用 `dtype=2`（INT8）。算子按
`y = clamp(round(x * scale) + offset, -128, 127)` 计算，其中 `round`
为 ties-to-even；这里的乘法 `scale` 是 PRD 仿射量化 scale 的倒数，
`offset` 是浮点表示的整数 zero point。

## 前置条件与错误

- `x`、`scale`、`offset` 必须位于同一设备并具有相同浮点 dtype；`offset` 省略时按同 dtype 的零处理。
- `scale` 必须有限且严格大于零；输入、scale 或 offset 包含 NaN/Inf 时必须显式报错。
- `axis` 规范化后必须位于输入 rank 内，scale/offset shape 必须满足 axis 广播规则。
- 自定义算子不支持 autograd；反向传播必须抛出明确错误。

## 数值验收

CPU 和 GPU 的 INT8 输出必须与独立 FP32 reference **逐元素完全一致**，不使用浮点容差。测试使用 seed `20260714`，并覆盖：

- ties-to-even 边界 `±0.5`、`±1.5`；
- `-128`、`127` 两侧的饱和输入；
- 标量、per-tensor、per-token `axis=-2`；
- 全零、最小合法 shape、非连续输入和 `[1, 3072, 64]` 发布 shape。

Fake/Meta 必须保持输入 shape 并将输出 dtype 推导为 INT8。
