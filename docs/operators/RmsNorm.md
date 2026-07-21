# RmsNorm

源码位于 `mdc_llm_deploy/custom_ops/rms_norm/`，按职责拆分：

- `contract.py`：Torch 宽契约；
- `kernels.py`：CPU/CUDA kernel；
- `fake.py`：FakeTensor 元数据推导；
- `onnx.py`：ONNX 窄契约、本地 schema 与 Dynamo translation；
- `registration.py`：不可变插件描述及 Torch 注册。

导入 `mdc_llm_deploy.custom_ops.rms_norm` 只注册 RmsNorm 的 Torch 算子。
调用 `create_onnx_export_profile("rms_norm")` 后，才在当前进程注册默认域
`NPURmsNorm` schema，并返回 Dynamo `custom_translation_table`。

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

Torch 会拒绝 shape、rank、dtype、设备、epsilon、NaN/Inf 不合法的输入。宽契约
支持单尾维及多尾维归一化，并用于 eager、FakeTensor、`torch.compile` 和
`torch.export`。算子仅用于推理，不支持 autograd。

## ONNX ABI 与导出收窄

默认域、opset 18，本地 `OpSchema`：

```text
NPURmsNorm(x, gamma, epsilon: FLOAT = 1e-6) -> (y, rstd)
```

节点固定有 2 个输入、2 个输出，属性名固定为 `epsilon`。导出要求：

- `x/gamma` rank 和所有维度静态已知；
- rank、尾维 shape、dtype 满足 Torch 范围；
- 两者 dtype 相同，且为 `FLOAT16`、`BFLOAT16`、`FLOAT32`；
- `epsilon` 有限且大于 0。

缺少类型元数据、动态维度或任一约束不满足时，Dynamo translation 直接报错。
因此，Torch 中合法的动态 shape 不属于当前 ONNX 直出子集。ONNX 校验不得反向
限制 eager、FakeTensor 或 compile。

导出仅支持 `torch.onnx.export(..., dynamo=True, opset_version=18)`。不提供
legacy symbolic。schema 只存在于当前 Python 进程，不写入 ONNX 模型；新进程
运行 `onnx.checker(..., full_check=True)` 前必须重新创建 profile。

## B 端确定性用例

`tests.hardware.custom_ops.rms_norm` 定义 FLOAT32、末维归一化确定性用例。硬件
公共生成流程切换到 Dynamo 后，可由该定义生成标准 ONNX、MDC ONNX、输入 bin
和 manifest。本任务只做 A 端本地契约验收，不代表 ATC 或真机通过。