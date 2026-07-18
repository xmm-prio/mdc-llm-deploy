# FusedInferAttentionScore

## 名称

- GE 原名：`FusedInferAttentionScore`
- ONNX OP：`FusedInferAttentionScore`
- ONNX domain/opset：`ai.onnx::<opset>::FusedInferAttentionScore`
- CANN 9.0 执行接口版本：`FusedInferAttentionScoreV4`

## 源码映射与验证状态

- Python reference：`mdc_llm_deploy.operators.fused_infer_attention_score`
- 集中 schema 键：`OPERATOR_SCHEMAS["FusedInferAttentionScore"]`
- `OperatorSchema.inputs` 记录当前源码胶囊的三个必选张量；下方 parser 原型定义完整可选槽位及固定顺序。导出器不得压缩中间空槽位。
- 本文定义待验收契约，不代表 GPU、NPU、parser、ATC 或真机已验证。B 端 parser/ATC 状态以 `docs/validation/b-side.md` 文本记录为准。

## ONNX OP 原型

```text
FusedInferAttentionScore(
    query: Tensor,
    key: Tensor[],
    value: Tensor[],
    pse_shift: Tensor? = null,
    atten_mask: Tensor? = null,
    actual_seq_lengths: Tensor? = null,
    actual_seq_lengths_kv: Tensor? = null,
    dequant_scale1: Tensor? = null,
    quant_scale1: Tensor? = null,
    dequant_scale2: Tensor? = null,
    quant_scale2: Tensor? = null,
    quant_offset2: Tensor? = null,
    antiquant_scale: Tensor? = null,
    antiquant_offset: Tensor? = null,
    block_table: Tensor? = null,
    query_padding_size: Tensor? = null,
    kv_padding_size: Tensor? = null,
    key_antiquant_scale: Tensor? = null,
    key_antiquant_offset: Tensor? = null,
    value_antiquant_scale: Tensor? = null,
    value_antiquant_offset: Tensor? = null,
    key_shared_prefix: Tensor? = null,
    value_shared_prefix: Tensor? = null,
    actual_shared_prefix_len: Tensor? = null,
    query_rope: Tensor? = null,
    key_rope: Tensor? = null,
    key_rope_antiquant_scale: Tensor? = null,
    dequant_scale_query: Tensor? = null,
    learnable_sink: Tensor? = null,
    num_heads: int = 1,
    scale: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    input_layout: string = "BSH",
    num_key_value_heads: int = 0,
    sparse_mode: int = 0,
    inner_precise: int = 0,
    block_size: int = 0,
    antiquant_mode: int = 0,
    softmax_lse_flag: bool = false,
    key_antiquant_mode: int = 0,
    value_antiquant_mode: int = 0,
    query_quant_mode: int = 0
) -> (
    attention_out: Tensor,
    softmax_lse: Tensor
)
```

## 输入

| 名称 | 必选 | 支持类型 | 格式 | 说明 |
| --- | --- | --- | --- | --- |
| `query` | 是 | `FLOAT16`、`BFLOAT16`、`INT8` | `ND` | Attention 的 Q 输入 |
| `key` | 是 | `FLOAT16`、`BFLOAT16`、`INT8`、`INT4(INT32)` | `ND` | Attention 的 K 输入或 KV Cache 中的 Key；数组形式可表示各 Batch 的非连续 Tensor |
| `value` | 是 | `FLOAT16`、`BFLOAT16`、`INT8`、`INT4(INT32)` | `ND` | Attention 的 V 输入或 KV Cache 中的 Value；元素数量和对应 shape 应与 `key` 一致 |
| `pse_shift` | 否 | `FLOAT16`、`BFLOAT16` | `ND` | 加到 QK 分数上的位置编码偏置，如 PSE/ALiBi |
| `atten_mask` | 否 | `BOOL`、`INT8`、`UINT8` | `ND` | 标识不可参与 Attention 的 Q-K 位置；非零值表示屏蔽 |
| `actual_seq_lengths` | 否 | `INT64` | `ND` | 各 Batch 中 Query 的有效序列长度；TND 布局下通常使用累加长度 |
| `actual_seq_lengths_kv` | 否 | `INT64` | `ND` | 各 Batch 中 Key/Value 的有效序列长度；PagedAttention 场景必须提供 |
| `dequant_scale1` | 否 | `UINT64`、`FLOAT32` | `ND` | 第一次矩阵乘 QK 后的反量化因子，支持 per-tensor |
| `quant_scale1` | 否 | `FLOAT32` | `ND` | Softmax 结果 P 在第二次矩阵乘前的量化因子 |
| `dequant_scale2` | 否 | `UINT64`、`FLOAT32` | `ND` | 第二次矩阵乘 PV 后的反量化因子，支持 per-tensor |
| `quant_scale2` | 否 | `FLOAT32`、`BFLOAT16` | `ND` | `attention_out` 的后量化因子，支持 per-tensor/per-channel |
| `quant_offset2` | 否 | 与 `quant_scale2` 相同 | `ND` | `attention_out` 后量化的零点；提供时使用非对称量化 |
| `antiquant_scale` | 否 | Decode：`FLOAT16`、`BFLOAT16`、`FLOAT32`；Prefill：`FLOAT16` | `ND` | Key/Value 共用的反量化缩放因子 |
| `antiquant_offset` | 否 | 与 `antiquant_scale` 相同 | `ND` | Key/Value 共用的反量化零点；提供时使用非对称反量化 |
| `block_table` | 否 | `INT32` | `ND` | PagedAttention 中逻辑 KV block 到物理 block 的索引映射表 |
| `query_padding_size` | 否 | `INT64` | `ND` | Query 每个 Batch 左侧 padding 的元素数量，与有效序列长度共同确定搬运区间 |
| `kv_padding_size` | 否 | `INT64` | `ND` | Key/Value 每个 Batch 左侧 padding 的元素数量 |
| `key_antiquant_scale` | 否 | `FLOAT16`、`BFLOAT16`、`FLOAT32` | `ND` | Key 独立的反量化缩放因子；用于替代共用 `antiquant_scale` |
| `key_antiquant_offset` | 否 | 与 `key_antiquant_scale` 相同 | `ND` | Key 独立的反量化零点 |
| `value_antiquant_scale` | 否 | `FLOAT16`、`BFLOAT16`、`FLOAT32` | `ND` | Value 独立的反量化缩放因子 |
| `value_antiquant_offset` | 否 | 与 `value_antiquant_scale` 相同 | `ND` | Value 独立的反量化零点 |
| `key_shared_prefix` | 否 | `FLOAT16`、`BFLOAT16`、`INT8` | `ND` | 所有 Batch 共用的系统前缀 Key |
| `value_shared_prefix` | 否 | `FLOAT16`、`BFLOAT16`、`INT8` | `ND` | 所有 Batch 共用的系统前缀 Value，需与 `key_shared_prefix` 配套提供 |
| `actual_shared_prefix_len` | 否 | `INT64` | `ND` | 共享 Key/Value 前缀的实际有效长度，通常为 shape `(1)` |
| `query_rope` | 否 | `FLOAT16`、`BFLOAT16` | `ND` | MLA 场景中与 Query 非 RoPE 部分分离传入的 RoPE 特征 |
| `key_rope` | 否 | `FLOAT16`、`BFLOAT16` | `ND` | MLA 场景中与 Key 非 RoPE 部分分离传入的 RoPE 特征；需与 `query_rope` 配套提供 |
| `key_rope_antiquant_scale` | 否 | 保留参数，CANN 9.0 不生效 | - | 为 Key RoPE 部分预留的反量化因子 |
| `dequant_scale_query` | 否 | `FLOAT32` | `ND` | 全量化场景中 INT8 Query 的反量化因子，支持 per-token 叠加 per-head |
| `learnable_sink` | 否 | `FLOAT16`、`BFLOAT16` | `ND` | 每个 Query head 的可学习 Sink Token 分数，用于吸收无关 Attention 概率 |

## 输出

| 名称 | 支持类型 | 格式 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `attention_out` | `FLOAT16`、`BFLOAT16`、`INT8` | `ND` | D 维与 `value` 相同，其余维度与 `query` 对应 | `Softmax(scale * QKᵀ + mask/pse) * V` 的结果 |
| `softmax_lse` | `FLOAT32` | `ND` | 通常为 `(B,N,Q_S,1)`；TND/NTD_TND 为 `(T,N,1)` | 每行 Attention 分数的 log-sum-exp，用于 Ring Attention 等后续合并 |

`softmax_lse_flag=false` 时，`softmax_lse` 为 shape `[1]` 的零 Tensor。

## 属性

| 名称 | ONNX 类型 | 默认值 | 支持值/说明 |
| --- | --- | --- | --- |
| `num_heads` | `INT` | `1` | Query head 数，即 `Q_N`；应与 `query` 布局中的 N 维一致 |
| `scale` | `FLOAT` | `1.0` | 乘到 QK 分数上的缩放系数，通常设置为 `1 / sqrt(D)` |
| `pre_tokens` | `INT` | `2147483647` | 每个 Query 最多向前关联的 token 数；最大值表示不限制 |
| `next_tokens` | `INT` | `2147483647` | 每个 Query 最多向后关联的 token 数；最大值表示不限制 |
| `input_layout` | `STRING` | `"BSH"` | 指定 Q/K/V 的维度排列；带下划线时表示“输入布局_输出布局”，如 `BNSD_BSND` |
| `num_key_value_heads` | `INT` | `0` | Key/Value head 数，即 `KV_N`；0 表示与 Query 相同，用于配置 MQA/GQA |
| `sparse_mode` | `INT` | `0` | Mask 模式：`0=default`、`1=all mask`、`2=左上 causal`、`3=右下 causal`、`4=band`、`9=tree mask` |
| `inner_precise` | `INT` | `0` | bit0 选择精度/性能，bit1 控制无效行修正：`0=高精度`、`1=高性能`、`2=高精度+修正`、`3=高性能+修正` |
| `block_size` | `INT` | `0` | PagedAttention 每个 KV block 可保存的最大 token 数；未使用 PagedAttention 时为 0 |
| `antiquant_mode` | `INT` | `0` | Key/Value 共用反量化参数的粒度；`0=per-channel（含 per-tensor）`，`1=per-token` |
| `softmax_lse_flag` | `INT`/`BOOL` | `false` | `true` 时计算并返回有效 `softmax_lse`；`false` 时第二输出为占位零 Tensor |
| `key_antiquant_mode` | `INT` | `0` | Key 独立反量化参数的粒度，应与 `key_antiquant_scale` 的 shape 对应 |
| `value_antiquant_mode` | `INT` | `0` | Value 独立反量化参数的粒度，应与 `value_antiquant_scale` 的 shape 对应 |
| `query_quant_mode` | `INT` | `0` | Query 全量化/反量化参数的粒度；CANN 9.0 V4 当前有效值 `3` 表示 per-token 叠加 per-head |

## 0.1.0 导出约束

- 输入 layout 固定为 `BNSD`，`num_heads=4`、`num_key_value_heads=2`、`scale=0.25`。
- batch 固定为 1，因此 `key` 和 `value` 的 Tensor 数组长度固定为 1；ONNX 符号直接在对应变长输入槽位传入单个 Tensor，不生成 ONNX Sequence。
- 设导出配置的静态序列长度为 `S`：prefill Q/K/V 的序列维均为 `S`；decode Q 的序列维为 1，输入 past K/V 的序列维为 `S-1`，算子接收拼接当前 token 后序列维为 `S` 的 K/V。发布配置 `S=3072` 时，prefill Q shape 为 `[1, 4, 3072, 16]`、K/V 为 `[1, 2, 3072, 16]`；decode Q 为 `[1, 4, 1, 16]`、拼接后 K/V 为 `[1, 2, 3072, 16]`。
- masked 模式显式提供 BOOL `atten_mask`，并使用 `sparse_mode=0`；maskless 模式省略 `atten_mask`，使用 `sparse_mode=0`、`pre_tokens=next_tokens=2147483647`。
- `softmax_lse_flag=false`，第二输出必须是 shape `[1]` 的 FP32 零 Tensor。
- K Cache 是 RoPE 后的 K，V Cache 是 value projection 后、进入 Attention 前的 V。
- Q/K/V 的 INT8 量化参数分别映射到 query dequant、`key_antiquant_*` 和 `value_antiquant_*`；softmax score 的 INT8 量化映射到 `quant_scale1`。
- 全 per-tensor INT8 场景必须同时提供 `dequant_scale1=query_scale×key_scale`、`quant_scale1=1/score_scale`、`dequant_scale2=score_scale×value_scale`。缺少任一项会导致 ATC tiling 失败。
- K/V 参数完全相同时允许合并为 `antiquant_*`。
- Q 和 softmax score 只允许对称量化，因为 `0.1.0` 接口没有对应 zero-point 输入；非对称 Q/score 必须在 `onnx_export` 阶段拒绝。K/V 非对称量化分别使用 `key_antiquant_offset` 和 `value_antiquant_offset`。
- `0.1.0` 的模型 ONNX 只允许浮点或 INT8 K/V；`INT4(INT32)` 仅保留为算子接口能力，不进入发布模型。

## 模型 ONNX KV Cache I/O

`ExportModelConfig.save_kv_cache` 默认是 `True`。标准 ONNX 与 MDC ONNX 使用同一公开名称、
顺序和静态 BNSD shape：

- prefill 输入仅有 `input_ids: [1, S]`；输出为 `logits`，随后按数字层序排列
  `present.N.key`、`present.N.value`，每个 cache shape 为 `[1, Nkv, S, D]`。
- decode 输入为 `input_ids: [1, 1]`，随后按相同层序排列 `past.N.key/value`，shape 为
  `[1, Nkv, S-1, D]`；输出为 `logits` 和 `[1, Nkv, S, D]` 的
  `present.N.key/value`。
- 显式设置 `save_kv_cache=False` 只隐藏两阶段 ONNX 的公开 `present` 输出。decode 的
  `past` 输入、FX 内部逐层 `present` 输出和 Attention lowering 不变。

浮点路径的 cache 沿用图 dtype，生产模型通常为 `FLOAT16`。Attention key/value 激活量化
命中时，MDC ONNX 中送入 `FusedInferAttentionScore` 并公开的对应 cache 为 `INT8`；
量化 decode 的 `past` 和 `present` ABI 同为 `INT8`。标准 ONNX 保留 FX 图声明的 cache
dtype，不套用 MDC lowering 的 dtype 覆盖。

## 前置条件与错误

- Q/K/V 必须位于同一设备，head dim 必须一致，K/V 的 batch、sequence、KV head 必须一致。
- mask 必须能广播到 `[B, Nq, Sq, Skv]`，非零值表示屏蔽。
- 量化输入必须提供与 dtype、粒度和 shape 匹配的 scale/offset；K/V 参数不同时禁止误用共用 `antiquant_*`。
- 输入、scale 或 offset 包含 NaN/Inf，head 数不匹配，或可见行被完全 mask 时必须显式报错。
- 自定义算子不支持 autograd；反向传播必须抛出明确错误。

## 数值验收

独立 reference 使用 FP32 计算 `Softmax(scale * QKᵀ + mask) * V`，最后转换到目标输出 dtype。随机输入使用 seed `20260714`、正态分布 `N(0, 0.5)`；同时覆盖全零、单 token、GQA、causal/maskless、量化 K/V、非连续输入和 3072-token 发布 shape。

余弦相似度在展平后的 `attention_out` 上计算；若 reference 与结果均为全零，则定义为 1，仅一方全零则定义为 0。

| 路径/输出 dtype | `atol` | `rtol` | 最低余弦相似度 |
| --- | ---: | ---: | ---: |
| 浮点 `FLOAT32` | `1e-4` | `1e-4` | `0.99999` |
| 浮点 `FLOAT16` | `3e-3` | `3e-3` | `0.999` |
| 浮点 `BFLOAT16` | `2e-2` | `2e-2` | `0.995` |
| INT8 Q/K/V 或 score，输出 `FLOAT16` | `1e-2` | `1e-2` | `0.99` |

CPU 和 GPU 都必须满足绝对/相对误差与余弦阈值；三者任一失败即失败。Fake/Meta 必须精确匹配两个输出的 shape 和 dtype。
