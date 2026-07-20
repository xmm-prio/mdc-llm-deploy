# MoeExpert

源码：`mdc_llm_deploy/custom_ops/moe_expert.py`。

## Torch CPU/CUDA 支持范围

```text
MoeExpert(
    x, topk_ids, topk_weight, expert_weights,
    quant_scales=null, quant_offsets=null
) -> out
```

- `x` 为浮点 `[token_count, hidden_size]`；CPU 接受 PyTorch 浮点 dtype，CUDA
  明确只接受 `FLOAT16`、`BFLOAT16`、`FLOAT32`。
- `topk_ids` 为 INT32/INT64 `[token_count, top_k]`；`topk_weight` 同 shape、
  同 `x` dtype。`top_k > 0`，每个 token 的 expert id 不重复、不越界；routing
  权重非负、有限，每行在 `rtol=1e-4, atol=1e-5` 下和为 1。
- `expert_weights` 为 expert-major rank 2：
  `[expert_count, 3 * hidden_size * intermediate_size]`。每行依次存放 row-major
  `gate_proj [I,H]`、`up_proj [I,H]`、`down_proj [H,I]`。
- 浮点权重必须与 `x` dtype 相同，且不得传量化参数。
- INT8 权重必须传浮点 `quant_scales [E,2I+H]`；`quant_offsets` 可省略，省略
  表示零 offset，否则 shape/dtype 规则相同。参数顺序为 gate 的 I 项、up 的
  I 项、down 的 H 项，均按输出 channel 反量化。
- 所有输入必须同设备。输出 shape/dtype 与 `x` 相同；各 expert 执行
  `down(silu(gate(x)) * up(x))` 后按 routing 权重求和。

CPU 使用 FP32 中间计算。CUDA 使用分阶段 Triton kernel，不回退到 PyTorch；
内部会转连续，要求 hidden size 与 intermediate size 的下一次幂不超过 65536，
且环境安装 Triton。非法 packed width、routing、量化组合或设备会明确报错。
算子仅用于推理，不支持 autograd。

## ONNX ABI 与导出边界

默认 `ai.onnx` 域、opset 18：

```text
MoeExpert(
    x, topk_ids, topk_weight, expert_weights,
    quant_scales?, quant_offsets?
) -> out
```

节点固定保留 6 个输入槽；浮点用例后两个槽为空，INT8 用例填入量化参数。当前
symbolic 只负责保持 ABI，不额外读取 ONNX 类型/shape 元数据；Torch 前向负责
执行时约束，ATC/板端负责最终 ABI 能力校验。

## B 端确定性用例

`tests.hardware.custom_ops.moe_expert` 分别生成浮点 packed 权重用例和带非零
offset 的 INT8 per-channel 反量化用例。两者均包含标准 ONNX、MDC ONNX、合法
routing 输入 bin 和 manifest。
