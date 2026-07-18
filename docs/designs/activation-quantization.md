# 激活边界量化共享

## 目标

激活量化以物理 tensor value 为边界，而不是以 Linear、Attention 或 MoE
的逻辑 target FQN 为边界。同一个 value 被多个 target 消费时，只采集一份
校准样本；导出时，同一个 ONNX value 的等价量化请求只生成一个
`NPUAscendQuantV2`。

该设计不识别 Q/K/V 名称。Q、K、V 投影共享 Quant，只是通用 fan-out
共享规则的一个结果。

## FX 校准边界

一次 `oneshot` 事务内，校准规划把 target 映射到真实 FX `Node`：

- Linear 和 GPTQ 使用算子输入 value。
- Attention query、key、value 和 score 使用对应输出 value。
- MoE 使用 `MoeExpert` 的激活输入 value。

校准计划是阶段准入的唯一依据。计划不需要任何产物时，校准阶段直接返回空工件，不枚举
`calibration_dataloader`、不解析物理边界，也不执行 FX 图；计划非空时才逐 batch 校验并观察。
GPTQ 和 packed MoE 即使没有 activation 配置也需要完整样本，因此不会进入空计划短路。

同一边界在每个 calibration batch 中观察一次。校准计划按算法声明产物：
MinMax 流式合并极值并生成 qparams，不保留跨 batch 原始 activation；
GPTQ 与 packed MoE 保留完整、顺序稳定的 activation matrix。多个 target
可共享同一物理边界的采集状态。

物理边界共享不代表量化契约合并。bits、granularity、mode 或 symmetric
不同的 target 分别生成 qparams；边界与激活契约都相同时，极值状态和
qparams 结果复用。`per_tensor` 极值使用常量空间，`per_token` 仅保留每行
极值，均不保留隐藏维原始 activation。

参数 alias group 聚合 GPTQ 或 MoE 样本时，也按物理边界去重，避免同一
样本因多个 FQN 被重复拼接。

## ONNX Quant 等价

一次 MDC ONNX lowering 只创建一个 `OnnxLoweringContext`。Linear、
Attention 和 MoE 通过该 context 请求激活 Quant，不直接创建
`NPUAscendQuantV2`。

两个请求仅在以下内容全部相同时等价：

- source ONNX value 名称；
- 实际发射的 inverse scale 内容、dtype 和 shape；
- 实际发射的 offset 内容、dtype 和 shape；
- `axis`；
- Quant 输出 dtype。

等价请求复用第一个 Quant 输出。source 不同或任一发射契约不同，均保留
独立 Quant。不跨 Reshape、Transpose、Cast 推断数值等价。

Attention key/value cache 先把浮点 producer 重绑到内部稳定名称，再请求
以原 graph output 名称为输出的 Quant，因此 graph ABI 不变。共享 context
同时维护名称、类型和 initializer 索引；任一 lowering pass 替换 initializer
集合后必须同步重建索引。

## KV Cache 公开 ABI

`ExportModelConfig.save_kv_cache` 默认是 `True`，标准 ONNX 与 MDC ONNX 均公开
`logits` 和按数字层序排列的 `present.N.key/value`。显式设置为 `False` 时，两类 ONNX
均只公开 `logits`；该开关不删除 FX 内部 cache，也不删除 decode 的 `past.N.key/value`
输入。

设静态导出长度为 `S`、KV head 数为 `Nkv`、head dim 为 `D`。prefill 的 present cache
使用 BNSD shape `[1, Nkv, S, D]`。decode 的 past cache 固定为
`[1, Nkv, S-1, D]`，拼接当前 token 后的 present cache 固定为
`[1, Nkv, S, D]`；此契约只描述一次静态 `S-1` 到 `S` 转换，不提供动态 cache 生命周期。

未命中 Attention key/value 量化目标时，cache 沿用图的浮点 dtype，生产 FP16 图通常为
`FLOAT16`。命中时，MDC lowering 把对应 cache producer 重绑为 `INT8` Quant 输出；
量化 decode 的 past/present ABI 同为 `INT8`。标准 ONNX 继续使用 FX metadata 声明的
dtype，不复用 MDC 专属 dtype 覆盖。无论是否公开 present 输出，内部 producer、量化请求
和 decode 转换都保持不变。

## metadata v1 兼容

`GRAPH_SCHEMA_VERSION` 保持 `1`。`GraphMetadata`、`QuantizedTarget` 和
公开 API 不增加必填字段。

每个逻辑 target 继续保留独立的：

- `quantized_targets` 项；
- `properties["activation_qparams"][fqn]` 项。

共享只存在于运行期校准边界和 ONNX lowering context。旧消费者仍按 FQN
读取相同结构，不需要理解共享关系。

## 导出验证

普通 `onnx_export` 与 release validation 共用量化拓扑验证：

- 等价 `NPUAscendQuantV2` 不得重复；
- Quant 输出必须到达合法的 Linear、Attention、MoE 消费端口或 graph output；
- 每个 Linear target 仍需独立 MatMul 和 AscendDequant 覆盖，但 Quant 数量
  可以小于 Linear target 数量；
- decode 的 INT8 cache graph input 可以直接进入 Attention；
- custom operator 必须可达，图必须满足 SSA 和拓扑顺序；
- metadata 声明的 target family 必须与图中结构一致。

相同图、配置和校准数据重复导出时，节点、边、initializer 名称与顺序以及
Quant 共享关系必须一致。
