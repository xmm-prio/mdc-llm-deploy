# MoeExpert

源码：`mdc_llm_deploy/custom_ops/moe_expert.py`。

## 两套独立契约

类内保留两套明确隔离的契约：

- 通用 Torch 契约用于 CPU/CUDA 推理，支持浮点 `x`、浮点或 INT8 权重。
- MDC ONNX 契约用于板端全量化部署，严格匹配 `MoeExpert::OpDef`。

通用浮点 Torch 调用不得导出为 MDC ONNX；导出时会明确报错，不做隐式布局或
dtype 转换。

## 通用 Torch CPU/CUDA 契约

```text
MoeExpert(
    x, topk_ids, topk_weight, expert_weights,
    quant_scales=null, quant_offsets=null
) -> out
```

- `x`：浮点 `[T,H]`。CUDA 支持 `FLOAT16`、`BFLOAT16`、`FLOAT32`。
- `topk_ids`：INT32/INT64 `[T,K]`；`topk_weight` 与其同 shape，并与 `x`
  dtype 相同。
- `expert_weights`：expert-major
  `[E,3*H*I]`，每个 expert 依次保存 `gate [I,H]`、`up [I,H]`、
  `down [H,I]`。
- 浮点权重与 `x` dtype 相同，不接受量化参数。
- INT8 权重接受通用 per-channel `quant_scales [E,2I+H]`，可选同 shape
  浮点 `quant_offsets`。
- 输出 shape/dtype 与 `x` 相同。CPU 使用 FP32 中间计算；CUDA 使用 Triton。

Routing expert id 必须合法且同 token 内不重复；权重必须非负、有限，每行和为 1。

## MDC ONNX 六槽 ABI

默认 `ai.onnx` 域、opset 18，固定六个输入槽：

```text
MoeExpert(
    x, topk_ids, topk_weight, expert_weights,
    quant_scales, quant_offsets?
) -> out
```

- `x`：INT8 `[T,H]`。
- `topk_ids`：INT16 `[T,K]`。
- `topk_weight`：FP16 `[T,K]`，按 `T*K` 连续顺序读取。
- `expert_weights`：INT8 `[3*E*I,H]`。每个 expert 连续保存
  `gate [I,H]`、`up [I,H]`、`down [I,H]`。
- `quant_scales`：必填 FP32 `[1+4E]`。第 0 项为 `tokenScale`；之后每个
  expert 依次为 `gateWScale`、`upWScale`、`activationScale`、
  `downWScale`；运行时要求所有 scale 有限且为正数。
- `quant_offsets`：当前板端不支持且内核不读取，必须保留为空槽。
- `E=(len(quant_scales)-1)/4`，
  `I=expert_weights.shape[0]/(3E)`。
- 输出：FP16 `[T,H]`。

当前 tiling 约束：`H` 为 256 的正整数倍，`I` 为 128 的正整数倍。

全量化参考语义：

1. INT8 token 分别与 gate/up INT8 权重做整数点积，再乘
   `tokenScale*gateWScale` 或 `tokenScale*upWScale`。
2. 计算 `silu(gate)*up`，除以 `activationScale` 后 round，并截断到
   INT8 `[-128,127]`。
3. INT8 activation 与 down INT8 权重做整数点积，乘
   `activationScale*downWScale`。
4. 按 FP16 `topk_weight` 加权累加，输出转 FP16。

CPU golden/Fake 与 CUDA Triton 均接受此全量化契约。

## B 端确定性用例

`tests.hardware.custom_ops.moe_expert` 只生成一个真实 MDC INT8 用例：
`T=1,H=256,I=256,E=4,K=2`。用例包含标准 ONNX golden、六槽 MDC ONNX、
输入 bin 和 manifest；`quant_offsets` 序列化为空槽。
