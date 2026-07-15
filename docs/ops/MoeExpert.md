# MoeExpert

## 名称

- GE 原名：`MoeExpert`
- ONNX OP：`MoeExpert`
- ONNX domain/opset：`ai.onnx::<opset>::MoeExpert`

## 源码映射与验证状态

- Python reference：`mdc_llm_deploy.mdc_ops.operators.moe_expert`
- 集中 schema 键：`OPERATOR_SCHEMAS["MoeExpert"]`
- `OperatorSchema.inputs` 记录五个必选输入；`quant_offsets` 是下方原型定义的第六个可选输入，省略时按对称量化处理。
- CANN 9.1.0、SoC `MC62CM12AA` 已完成部分 ATC 编译探针；真机推理仍未验证。完整状态以 `docs/validation/b-side.md` 为准。

## ONNX OP 原型

```text
MoeExpert(
    x: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    expert_weights: Tensor,
    quant_scales: Tensor,
    quant_offsets: Tensor? = null
) -> out: Tensor
```

## 输入

| 序号 | 名称 | 必选 | 支持类型 | 格式 | Shape | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `x` | 是 | `INT8` | `ND` | `[tokenNum, hiddenSize]` | Token 隐状态（激活值量化输入）。已经过量化的 Token 特征矩阵 |
| 1 | `topk_ids` | 是 | `INT16` | `ND` | `[tokenNum, topkExpertNum]` | 专家索引。每个 Token 激活并路由指向的前 K 个专家 ID |
| 2 | `topk_weight` | 是 | `FLOAT16` | `ND` | `[tokenNum, topkExpertNum]` | 专家路由权重。每个 Token 对应激活专家的归一化权重，通常由 Softmax 计算得出 |
| 3 | `expert_weights` | 是 | `INT8` | `ND`（一维） | `[expertWeightsNum]` | 量化专家权重（Packed）。所有专家网络权重经 Flatten 后紧凑打包存储，内部包含所有专家的权重参数 |
| 4 | `quant_scales` | 是 | `FLOAT32` | `ND` | `[1 + expertNum × 4]` | 反量化 Scale 参数。包含 1 个全局 Token 激活值 Scale，以及每个专家的 4 个特定 Scale，例如不同投影层或门控层使用的 Scale |
| 5 | `quant_offsets` | 否 | `INT32` | `ND` | 与 `quant_scales` 相同 | 反量化偏移量（Zero Point），顺序与 `quant_scales` 完全一致；省略时使用对称反量化，Zero Point 为 0 |

## 输出

| 序号 | 名称 | 必选 | 支持类型 | 格式 | Shape | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `out` | 是 | `FLOAT16` | `ND` | `[tokenNum, hiddenSize]` | 根据 `topk_ids` 执行专家网络计算，并以 `topk_weight` 加权合并后得到的 Token 特征矩阵 |

## 0.1.0 参数顺序与 shared expert

Tiny Qwen3-MoE 使用 4 个 routed expert 和 1 个 shared expert，因此 `expertNum=5`。expert id `0..3` 是 routed expert，id `4` 是 shared expert。

每个 token 先选择 routed top-2 并把两项权重归一化为和 1，再追加 `(topk_id=4, topk_weight=1.0)`，因此传给本算子的 `topkExpertNum=3`。算子结果等价于“两个 routed expert 的加权和 + shared expert 输出”。

`quant_scales` 长度固定为 `21`，顺序为：

1. 全局输入激活 scale；
2. 按 expert id `0..4`，每个 expert 依次存放 `gate_proj` 权重、`up_proj` 权重、中间激活、`down_proj` 权重 scale。

`quant_offsets` 如存在，使用完全相同的顺序。`expert_weights` 按 expert id 和上述三个权重投影顺序打包；每个矩阵内部使用 row-major。各段起始 offset 和长度必须写入图元数据并由结构测试断言，不能依赖 Python 对象遍历顺序。

## 前置条件与错误

- 当前 ATC 发布规格要求 `hiddenSize` 按 256 对齐、`expertInterDim` 按 128 对齐。已验证 `hiddenSize=256`、`expertInterDim=128/256`、`tokenNum=8/3072`；`hiddenSize=64/128` 会在 tiling 阶段失败。
- `topk_ids` 值域必须为 `[0, expertNum)`；每行 routed id 不得重复，shared id 必须且只能出现一次。
- 每行前两个 routed weight 必须有限、非负且在 FP32 下和为 1；shared weight 必须为 1。
- `x`、weights、scale/offset 必须位于同一设备，shape 和打包元数据必须一致。
- 输入、权重或量化参数含 NaN/Inf、expert id 越界或重复时必须显式报错。
- 自定义算子不支持 autograd；反向传播必须抛出明确错误。

## 数值验收

独立 reference 先按 scale/offset 反量化输入和权重，以 FP32 执行 SiLU gated MLP：
`down_proj(silu(gate_proj(x)) * up_proj(x))`，再按 routed/shared 规则合并，最后转换为 FP16。

随机输入使用 seed `20260714`，并覆盖全零、单 token、所有 expert 均被命中、饱和 INT8、非法 id/重复 id 和发布 shape。CPU 与 GPU 的 `out` 均须满足 `atol=5e-3`、`rtol=5e-3`，且展平余弦相似度不低于 `0.999`。若两侧均全零，余弦相似度定义为 1；仅一侧全零则为 0。

Fake/Meta 必须输出 `[tokenNum, hiddenSize]` 的 FP16 Tensor。
