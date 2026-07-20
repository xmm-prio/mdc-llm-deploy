# RmsNorm

源码：`mdc_llm_deploy/custom_ops/rms_norm.py`。

## Torch CPU/CUDA 支持范围

- 输入：`x, gamma, epsilon=1e-6`；输出：`y, rstd`。
- `x` 为 1～8 维；`gamma` 为 1～`x.ndim` 维，且必须精确匹配 `x` 的一个或
  多个尾维。`gamma` 各维不得为空。
- `x/gamma` 支持 `FLOAT16`、`BFLOAT16`、`FLOAT32`，dtype 与设备必须一致。
- `epsilon` 必须是有限正实数，bool 不视为实数参数。
- `y` shape/dtype 与 `x` 相同；`rstd` shape 为去除归一化尾维后的前缀，
  dtype 固定 `FLOAT32`。
- CPU 使用 FP32 累加。CUDA 使用 Triton，不回退到 PyTorch；额外要求输入连续，
  `gamma.numel() <= 65536`，且环境安装 Triton。

Torch 会拒绝 shape、rank、dtype、设备、epsilon、NaN/Inf 不合法的输入。算子
仅用于推理，不支持 autograd。

## ONNX ABI 与导出收窄

默认 `ai.onnx` 域、opset 18：

```text
NPURmsNorm(x, gamma, epsilon: FLOAT = 1e-6) -> (y, rstd)
```

节点固定有 2 个输入、2 个输出，属性名固定为 `epsilon`。导出要求：

- `x/gamma` rank 和所有维度静态已知；
- rank、尾维 shape、dtype 满足 Torch 范围；
- 两者 dtype 相同，且为 `FLOAT16`、`BFLOAT16`、`FLOAT32`；
- `epsilon` 有限且大于 0。

缺少类型元数据、动态维度或任一约束不满足时，导出直接报错。

## B 端确定性用例

`tests.hardware.custom_ops.rms_norm` 生成 FLOAT32、末维归一化用例，同时包含
标准 ONNX、MDC ONNX、输入 bin 和 manifest。