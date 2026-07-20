# ApplyRotaryPosEmb

源码：`mdc_llm_deploy/custom_ops/apply_rotary_pos_emb.py`。

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

## ONNX ABI 与导出收窄

默认 `ai.onnx` 域、opset 18：

```text
ApplyRotaryPosEmb(
    query, key, cos, sin,
    layout: INT = 1,
    rotary_mode: STRING = "half"
) -> (query_out, key_out)
```

节点固定有 4 个输入、2 个输出，属性名为 `layout`、`rotary_mode`。导出仅检查
`layout ∈ {1,2,3,4}` 和三种 mode；具体 dtype/shape 边界由 Torch 前向及板端
编译共同约束。属性不在上述集合时导出直接报错。

## B 端确定性用例

`tests.hardware.custom_ops.apply_rotary_pos_emb` 生成 FLOAT32、BSND、half、
GQA head 数用例，同时包含标准 ONNX、MDC ONNX、输入 bin 和 manifest。
