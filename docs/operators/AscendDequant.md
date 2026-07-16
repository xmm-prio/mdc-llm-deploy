# AscendDequant

## 名称

- GE 原名：`AscendDequant`
- ONNX OP：`AscendDequant`
- ONNX domain/opset：支持标准导入形式 `ai.onnx::<opset>::AscendDequant`

## 源码映射与验证状态

- Python reference：`mdc_llm_deploy.operators.ascend_dequant`
- 集中 schema 键：`OPERATOR_SCHEMAS["AscendDequant"]`
- 源码函数使用 snake_case，GE 名称和 ONNX `op_type` 使用上方固定名称。
- 本文定义待验收契约，不代表 GPU、NPU、parser、ATC 或真机已验证。B 端 parser/ATC 状态以 `docs/validation/b-side.md` 文本记录为准。

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


通用 GE 接口允许 `UINT64` 高 32 位承载控制信息；`0.1.0` 只使用受限子集：低 32 位保存完整 IEEE-754 FP32 bit pattern，高 32 位必须清零，不执行 s19 尾数截断或其他控制编码。

## 输出


| 名称  | 支持类型                | 格式   | Shape    | 说明 |
| --- | ------------------- | ---- | -------- | --- |
| `y` | `FLOAT16`、`FLOAT32` | `ND` | 与 `x` 对应 | `x` 乘以 `deq_scale`，并按属性完成类型转换和可选 ReLU 后的反量化结果 |




## 属性


| 名称          | ONNX 类型      | 默认值     | 说明                         |
| ----------- | ------------ | ------- | -------------------------- |
| `sqrt_mode` | `INT`/`BOOL` | `false` | `false` 时直接使用 `deq_scale`；`true` 时启用 scale 平方根分解，用于降低大 scale 的低精度表示损失 |
| `relu_flag` | `INT`/`BOOL` | `false` | `true` 时对反量化结果执行 `max(y, 0)`；`false` 时不融合激活 |
| `dtype`     | `INT`        | `0`     | 指定 `y` 的目标类型，使用 GE 数据类型编码；`0=FLOAT32`、`1=FLOAT16` |

`0.1.0` 导出器必须始终显式写入 `dtype`，不得依赖默认值；只允许 `0` 和 `1`。

## 前置条件与错误

- `x` 必须为 INT32，`deq_scale` 必须为 UINT64 标量或与输出通道对应的一维 Tensor。
- `deq_scale` 的高 32 位非零、低 32 位解码为 NaN/Inf 或 shape 不匹配时必须显式报错。
- `sqrt_mode` 和 `relu_flag` 在 `0.1.0` 发布模型中固定为 `false`；其他值可由独立算子测试覆盖，但不得由模型导出器生成。
- 自定义算子不支持 autograd；反向传播必须抛出明确错误。

## 数值验收

reference 先按上述 bit pattern 解码 FP32 scale，以 FP32 计算 `x * scale`，最后转换到目标 dtype。随机输入使用 seed `20260714`，并覆盖 INT32 零值、正负极值附近、标量/通道 scale、最小合法 shape和发布 shape。

| 输出 dtype | `atol` | `rtol` |
| --- | ---: | ---: |
| `FLOAT32` | `1e-5` | `1e-5` |
| `FLOAT16` | `1e-3` | `1e-3` |

CPU 与 GPU 均须满足阈值；Fake/Meta 必须保持逻辑 shape，并按显式 `dtype` 推导输出类型。


