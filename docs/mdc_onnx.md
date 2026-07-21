# MDC ONNX 图处理

`mdc_llm_deploy.mdc_onnx` 提供面向 MC62 部署的 ONNX 图处理 API。所有 API
接收已经加载到内存的 `onnx.ModelProto`，原地修改并返回同一对象，不负责文件读写。

## 总入口

```python
from mdc_llm_deploy.mdc_onnx import process_onnx

processed = process_onnx(model)
assert processed is model
```

`process_onnx` 按固定顺序执行：

1. 将受支持的 static W8A8 MatMul QDQ 子图 lowering 为
   `NPUAscendQuantV2 + INT8 MatMul + AscendDequant`；
2. 验证剩余标准算子可由 opset 18 表达，并把默认 domain opset 降至 18；
3. 运行最终 ONNX 校验，确认主图无残余 QDQ。

整个流程具有事务语义。任一步失败时抛出异常，传入模型保持不变。

## 支持范围

- 锚点仅支持 `MatMul`，不支持 `Gemm`。
- 激活仅支持 static INT8：
  - per-tensor；
  - Qwen token 轴 `-2` 的 per-token；
  - zero point 可为非零，转换为 `NPUAscendQuantV2.offset`。
- 权重仅支持 static、对称 INT8：
  - per-tensor；
  - per-output-channel；
  - 二维常量权重；
  - 直接 QDQ，或 QDQ 后单个二维交换 `Transpose`。
- 输出仅支持 FP16 和 FP32。
- `AscendDequant.deq_scale` 使用完整 FP32 bit pattern，存放于 UINT64
  低 32 位，高 32 位为零。

动态 scale、非对称权重、UINT8、INT4、FP8、blocked/group quant、半量化、
共享 QDQ 路径及其它图形均会严格失败。

## Schema 生命周期

导入 `mdc_llm_deploy.mdc_onnx` 时，会在当前 Python 进程的 ONNX registry 中幂等注册
`NPUAscendQuantV2` 和 `AscendDequant` 的默认 domain opset 18 schema。
schema 不会写入序列化模型；其它进程使用 `onnx.checker` 前也需要导入本包。

## 已知验证边界

非对称激活的 offset 修正依赖 ATC 的 MatmulQuantToFixpipeFusion。当前自动验收覆盖
图结构、参数转换和 MC62CM12AA ATC 编译；ATC 编译不能证明非对称路径的最终数值精度，
该项需要后续真机精度验证。
