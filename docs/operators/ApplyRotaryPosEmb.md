# ApplyRotaryPosEmb

`ApplyRotaryPosEmb` 由 ONNX pipeline 的 RoPE fusion pass 生成。输入必须是标准 ONNX
导出图，不依赖 Torch custom op、导出 profile 或 translation table。

## ONNX ABI

默认域、opset 18：

```text
ApplyRotaryPosEmb(
    query, key, cos, sin,
    layout: INT = 1,
    rotary_mode: STRING = "half"
) -> (query_out, key_out)
```

节点固定包含 4 个输入、2 个输出。当前 Qwen3 融合固定生成 `layout=3`（BNSD）和
`rotary_mode="half"`。schema 由 `mdc_llm_deploy.onnx.schema` 集中声明，
`OnnxAdapter` 按模型实际节点注册。

## 融合范围

RoPE pass 识别 Qwen3 的 half-rotation Q/K 子图：

- 支持 FP16、BF16、FP32；
- 要求静态 BNSD shape；
- 支持 MHA 和 GQA 的不同 Q/KV head 数；
- 必须完整证明 cos/sin 广播、旋转维和子图闭包；
- 不匹配或无法证明等价的子图保持不变。

```python
import onnx

from mdc_llm_deploy.onnx import AdapterConfig, OnnxAdapter

model = onnx.load("model.onnx")
OnnxAdapter(AdapterConfig())(model)
```

流程完成融合、按需 schema 注册和最终 ONNX checker 校验。新进程单独校验已生成模型
时，需先调用 `register_schemas("ApplyRotaryPosEmb")`。
