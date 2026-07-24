# RmsNorm

`NPURmsNorm` 由 ONNX pipeline 的 RMSNorm fusion pass 生成。输入必须是标准 ONNX
导出图，不依赖 Torch custom op、导出 profile 或 translation table。

## ONNX ABI

默认域、opset 18：

```text
NPURmsNorm(x, gamma, epsilon: FLOAT = 1e-6) -> (y, rstd)
```

节点固定包含 2 个输入、2 个输出，属性名为 `epsilon`。schema 由
`mdc_llm_deploy.onnx.schema` 集中声明，`OnnxAdapter` 按模型实际节点注册。

## 融合范围

RMSNorm pass 识别 Qwen3 的 FP32 累加分解：

- 支持 FP16、BF16、FP32 输入；
- 要求静态可证明的尾维归一化结构；
- `epsilon` 必须是有限正数；
- 不匹配或无法证明等价的子图保持不变。

```python
import onnx

from mdc_llm_deploy.onnx import AdapterConfig, OnnxAdapter

model = onnx.load("model.onnx")
OnnxAdapter(AdapterConfig())(model)
```

流程完成融合、按需 schema 注册和最终 ONNX checker 校验。新进程单独校验已生成模型
时，需先调用 `register_schemas("NPURmsNorm")`。
