# AscendDequant

## 名称

- ONNX OP：`AscendDequant`
- ONNX domain/opset：支持标准导入形式 `ai.onnx::<opset>::AscendDequant`

## 源码映射

- lowering：`mdc_llm_deploy.onnx.quant_lowering`
- ONNX schema：`mdc_llm_deploy.onnx.schemas`
- 源码函数使用 snake_case，ONNX `op_type` 使用上方固定名称。

## ONNX OP 原型

```text
AscendDequant(
    x: Tensor,
    deq_scale: Tensor,
    sqrt_mode: bool = false,
    relu_flag: bool = false,
    dtype: int = 0
) -> y: Tensor
```

## 输入


| 名称          | 必选  | 支持类型     | 格式   | 说明              |
| ----------- | --- | -------- | ---- | --------------- |
| `x`         | 是   | `INT32`  | `ND` | 量化累加结果          |
| `deq_scale` | 是   | `UINT64` | `ND` | 反量化因子；可为标量或通道向量 |


`deq_scale` 的低 32 位保存完整 IEEE-754 FP32 bit pattern，高 32 位必须清零。

## 输出


| 名称  | 支持类型                | 格式   | Shape    | 说明 |
| --- | ------------------- | ---- | -------- | --- |
| `y` | `FLOAT16`、`FLOAT32` | `ND` | 与 `x` 对应 | `x` 乘以 `deq_scale`，并按属性完成类型转换和可选 ReLU 后的反量化结果 |




## 属性


| 名称          | ONNX 类型      | 默认值     | 说明                         |
| ----------- | ------------ | ------- | -------------------------- |
| `sqrt_mode` | `INT`/`BOOL` | `false` | `false` 时直接使用 `deq_scale`；`true` 时启用 scale 平方根分解，用于降低大 scale 的低精度表示损失 |
| `relu_flag` | `INT`/`BOOL` | `false` | `true` 时对反量化结果执行 `max(y, 0)`；`false` 时不融合激活 |
| `dtype`     | `INT`        | `0`     | 指定 `y` 的目标类型；`0=FLOAT32`、`1=FLOAT16` |

导出器必须始终显式写入 `dtype`，不得依赖默认值；只允许 `0` 和 `1`。

## 前置条件与错误

- `x` 必须为 INT32，`deq_scale` 必须为 UINT64 标量或与输出通道对应的一维 Tensor。
- `deq_scale` 的高 32 位非零、低 32 位解码为 NaN/Inf 或 shape 不匹配时必须显式报错。
- 自定义算子不支持 autograd；反向传播必须抛出明确错误。
