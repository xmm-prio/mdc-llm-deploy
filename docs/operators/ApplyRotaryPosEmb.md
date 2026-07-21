# ApplyRotaryPosEmb

源码：`mdc_llm_deploy/custom_ops/apply_rotary_pos_emb/`。导入该算子包只注册
`mdc_llm_deploy::apply_rotary_pos_emb`，不会注册其他算子；创建包含
`apply_rotary_pos_emb` 的 ONNX export profile 时才安装进程内本地 schema。

## Torch CPU/CUDA 支持范围

- 输入顺序：`query, key, cos, sin, layout=1, rotary_mode="half"`；输出为
  `query_out, key_out`。
- 支持 `FLOAT16`、`BFLOAT16`、`FLOAT32`，四个输入 dtype、设备必须一致。
- 布局：`1=BSND`、`2=SBND`、`3=BNSD`、`4=TND`；前三种要求 rank 4，
  TND 要求 rank 3。
- `query` 与 `key` 仅允许 head 数不同，head dim 必须相同。`cos/sin` shape
  相同，head 轴必须为 1，其他非末维可取 1 或对应 token 维。
- 旋转维 `R=cos.shape[-1]` 满足 `0 < R <= D`。`half`、`interleave` 要求
  `R` 被 2 整除；`quarter` 要求被 4 整除。`R < D` 时尾部原样保留。
- CPU 使用 FP32 中间计算后转回输入 dtype。CUDA 使用 Triton，不回退到
  PyTorch；CUDA 额外要求四个输入连续，且环境安装 Triton。

Torch 会拒绝非法 layout/mode、rank、广播、dtype、设备、旋转维，以及含
NaN/Inf 的输入。算子仅用于推理，不支持 autograd。

## Dynamo ONNX ABI 与导出收窄

默认 `ai.onnx` 域、opset 18：

```text
ApplyRotaryPosEmb(
    query, key, cos, sin,
    layout: INT = 1,
    rotary_mode: STRING = "half"
) -> (query_out, key_out)
```

节点固定有 4 个输入、2 个输出，属性名为 `layout`、`rotary_mode`。仅支持
`torch.onnx.export(..., dynamo=True, opset_version=18)`，调用方必须传入
`create_onnx_export_profile("apply_rotary_pos_emb")` 返回的
`custom_translation_table`。不提供 legacy symbolic。

ONNX 直出边界独立收窄：

- 四个输入必须具有静态 shape、相同 dtype 和相同 rank；dtype 限于
  `FLOAT16`、`BFLOAT16`、`FLOAT`。
- layout、rank、query/key shape、cos/sin 广播关系与 Torch 契约相同。
- query/key head dim 相同且不超过 1024；旋转维仍须满足 mode 对应整除规则。

这些限制不进入 Torch kernel 或 Fake 路径。例如 head dim 大于 1024 仍可执行
eager、Fake、`torch.compile` 和 `torch.export`，但 Dynamo ONNX 导出会拒绝。
本地 `OpSchema` 只存在于当前进程；新进程调用 checker 前必须重新创建 profile。

## B 端确定性用例

`tests.hardware.custom_ops.apply_rotary_pos_emb` 定义 FLOAT32、BSND、half、
GQA head 数确定性用例。硬件 artifact 的统一 Dynamo 生成流程由后续跨算子集成
波次接入；本任务不代表 ATC 或真机验证通过。
