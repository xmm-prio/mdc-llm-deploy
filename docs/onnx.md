# MDC ONNX 图处理

`mdc_llm_deploy.onnx` 提供面向 MC62 部署的 ONNX 图处理 API。所有 API
接收已经加载到内存的 `onnx.ModelProto`，原地修改并返回同一对象，不负责文件读写。

## 总入口

```python
from mdc_llm_deploy.onnx import process_onnx

processed = process_onnx(model)
assert processed is model
```

`process_onnx` 按固定顺序执行：

1. 将受支持的 static W8A8 MatMul QDQ 子图 lowering 为
   `NPUAscendQuantV2 + INT8 MatMul + AscendDequant`；
2. 执行 MDC parser compatibility lowering：把静态 `Split.num_outputs` 精确转换为
   opset 18 的 `split` 常量输入；
3. 验证剩余标准算子可由 opset 18 表达，并把默认 domain opset 降至 18；
4. 规范化透明 `Identity`、常量 Cast 和无损浮点 Cast 往返；
5. 固定按 RMSNorm、RoPE、FIA 顺序运行融合；
6. 扫描主图及子图，只注册实际出现的 MDC custom-op schema；
7. 运行最终 ONNX checker，并确认主图无残余 QDQ。

lowering 产生的 custom op 会在 opset 检查前按需注册；融合新产生的 schema 会在最终
checker 前再次按需注册。导入包本身仍无 registry 副作用。整个流程先处理模型副本，
全部成功后才写回；任一步失败时抛出异常，传入模型字节保持不变，返回值始终是原
`ModelProto` 对象。

`Split.num_outputs` lowering 保留 ONNX 对非整除轴的定义：前段使用向上取整的块长，
最后一段接收剩余元素。输入轴必须是已知静态维度；动态维度无法在不引入运行时 shape
子图的前提下精确转换，因此会明确报错，不生成语义近似图。

## 融合编排器

```python
from mdc_llm_deploy.onnx import run_fusion_passes

report = run_fusion_passes(model)
assert tuple(report.counts) == (
    "rms_norm",
    "apply_rotary_pos_emb",
    "fused_infer_attention_score",
)
```

`run_fusion_passes` 直接原地逐 pass 修改模型并返回不可变 `FusionReport`。
`counts` 记录每个 pass 命中数，`total_fused_count` 记录总命中数。不匹配图是安全
no-op，三个计数均为 0。后续 pass 抛错时不会撤销前序 pass 的成功修改；这是独立
runner 的明确契约，不等同于 `process_onnx` 的全流程原子契约。

当前融合范围：

- RMSNorm：Qwen3 FP32 累加分解，支持 FP16、BF16、FP32；
- RoPE：Qwen3 BNSD half-rotation Q/K 对，支持 FP16、BF16、FP32；
- FIA：静态 BNSD eager/SDPA、Prefill/真实 KV-cache Decode、MHA/GQA，仅支持
  FP16、BF16；
- FP32 attention、动态 shape、有限 attention bias、ALiBi/PSE、dropout，以及 Q/K/V
  直接为 INT8 的量化 attention 均保持小算子图，FIA 命中数为 0；W8A8 projection
  经 `AscendDequant` 恢复为 FP16/BF16 后仍可融合。

## 支持范围

- 锚点仅支持 `MatMul`，不支持 `Gemm`。
- 激活仅支持 static INT8：
  - per-tensor；
  - Qwen token 轴 `-2` 的 per-token；
  - zero point 可为非零，转换为 `NPUAscendQuantV2.offset`。
- 权重仅支持 static、对称 INT8：
  - per-tensor；
  - per-output-channel；
  - 二维常量权重；
  - 直接 QDQ，或 QDQ 后单个二维交换 `Transpose`。
- 输出仅支持 FP16 和 FP32。
- `AscendDequant.deq_scale` 使用完整 FP32 bit pattern，存放于 UINT64
  低 32 位，高 32 位为零。

动态 scale、非对称权重、UINT8、INT4、FP8、blocked/group quant、半量化、
共享 QDQ 路径及其它图形均会严格失败。

## Schema 生命周期

全部自定义 schema 集中在 `mdc_llm_deploy.onnx.schemas`，包括量化、
`NPURmsNorm`、`ApplyRotaryPosEmb` 和 `FusedInferAttentionScore`。导入
`mdc_llm_deploy.onnx` 或 schema 包都不会修改 ONNX 进程内 registry。
调用方按节点名显式注册：

```python
from mdc_llm_deploy.onnx.schemas import register_schemas

register_schemas("NPURmsNorm", "ApplyRotaryPosEmb")
```

不传名称时注册中心内全部 schema。重复名称会去重；重复调用幂等；未知名称和同
domain、同 opset、同名称但 ABI 不同的已有 schema 会在写入前报错。schema 不会
写入序列化模型；其它进程使用 `onnx.checker` 前必须重新显式注册模型所需 schema。

选中 schema 会先完成整批 ABI 预检，全部通过后才开始注册。预检失败时本次调用不会
新增 schema。ONNX registry 不支持多 schema 事务；实际注册中途失败时，已成功注册的
兼容前缀会保留，且不会通过注销模拟回滚。项目内注册由同一锁串行化，但第三方直接调用
ONNX registry 不受该锁约束，仍可在预检和写入期间产生竞态。需要隔离此类外部修改时，
应在独立 Python 进程中完成处理。

## 已知验证边界

非对称激活的 offset 修正依赖 ATC 的 MatmulQuantToFixpipeFusion。当前自动验收覆盖
图结构、参数转换和 MC62CM12AA ATC 编译；ATC 编译不能证明非对称路径的最终数值精度，
该项需要后续真机精度验证。

Qwen3 FIA 硬件 bundle 同时包含：

- `models/`：完整模型，结果用于全图验收；
- `fia_slices/`：从每个完整模型的真实融合结果裁出的单节点 FIA ABI 编译切片，仅用于
  隔离 FIA schema、输入槽位和属性能否被 ATC 接受，不能替代完整模型验收；
- `atc_fusion_switch.json`：关闭
  `VenBatchMatMulActEltwiseFusionPassManager` 和
  `VenBatchMatMulEltwiseFusionPassManager`。

CANN 9.1.0 的上述 vendor fusion 会把 Qwen3 MLP MatMul 子图替换成
`VenFusedBatchMatMulV3`。MC62 安装中的该算子源码调用 `SetFixShiftValue`，但配套
AscendC `MatmulImpl` 不提供该成员，属于 CANN 组件内部 ABI 不一致。ATC 的
`--custom_fusion=off` 不会关闭这两个 vendor pass；编译完整模型时还需传入
`--fusion_switch_file=atc_fusion_switch.json`。该开关只禁用有缺陷的图融合，不改
ONNX 语义。

CANN 9.1.0 的 MC62 算子库还不支持 Qwen3 MoE 路由子图中的 INT64 `RealDiv`：
`TopK` 产生的 INT64 索引经过整除后供 `GatherND` 使用，而该环境的 `RealDiv` 仅接收
INT32 等类型。当前不通过插入窄化转换改变模型语义；因此 MoE 完整图仍记为环境阻塞，
FIA slice 编译成功也不能覆盖该失败。
