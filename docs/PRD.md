# mdc_llm_deploy 产品需求文档

## 1. 文档信息

- 产品名称及 Python 顶层包名：`mdc_llm_deploy`
- 当前版本：`0.1.0`
- 目标平台：华为 MDC 车载平台
- 当前验证环境：A 端负责开发，B 端提供 GPU 运行环境和 ATC 转换工具链；每次验收必须记录实际环境版本；MDC 真机验证不属于 `0.1.0` 范围

## 2. 产品目标

`mdc_llm_deploy` 是一个兼容 Transformers 模型的 LLM 部署转换库。它将模型转换为可量化的 ATen FX 图，以 fake quant 表达 PTQ 结果，完成 MDC 算子融合，并导出面向 MDC 的静态 ONNX 模型。

`0.1.0` 的目标是建立从模型导出、量化、算子融合、ONNX 导出到 ATC 转换的完整链路。GPU 算子运行和 ATC 转换在 B 端执行发布阻塞验证；在完成 MDC 真机验证前，不声明生成的 OM 模型已可在 MDC 上部署运行。

## 3. 0.1.0 交付范围

### 3.1 必须交付

- 单层、小词表的 Tiny Qwen3 Dense 参考模型。
- 单层、小词表的 Tiny Qwen3-MoE 参考模型。
- 从 Transformers/PyTorch 模型到 ATen FX 图的导出能力。
- 基于 ATen FX 图的 MinMax 量化，包括 linear、attention 和 moe 三类独立配置。
- 基于 ATen FX 图的 GPTQ 量化，包括 linear W4A8 和 moe W8A8；只验收 FX 数值，不支持 ONNX 导出。
- MDC 自定义算子的 ONNX 导出期图融合、参考计算实现、Fake/Meta 实现和 ONNX 符号注册。
- masked causal 和 maskless non-causal 两种模式下的 prefill/decode 静态 ONNX 导出。
- 将长序列 FX 图原地转换为 decode FX 图的能力。
- ONNX 结构检查和 PyTorch 参考计算验证。
- B 端 GPU 自定义算子运行验证。
- B 端 ATC 编译和 OM 模型生成验证。

### 3.2 不在 0.1.0 承诺范围内

- 官方 Qwen3 或 Qwen3-MoE checkpoint 的兼容性。
- 真实数据集上的困惑度、任务准确率或量化精度损失指标。
- 动态 batch、动态序列长度或超过约定长度的输入。
- MDC 真机运行。
- GPTQ/W4 ONNX 导出和 ATC 转换。
- 自定义算子的反向传播。
- SpinQuant 等离群值抑制算法。

## 4. 技术与依赖约束

- Python 3.11 和 3.12 均受 `0.1.0` 支持；A 端发布门禁固定使用 Python 3.12，B 端 GPU 与 ATC 验证允许使用实际部署环境的 Python 3.11。
- 核心依赖包括 PyTorch、Transformers、Accelerate、ONNX、ONNXScript、Datasets、Evaluate 和 Triton。
- 核心依赖采用“经过验证的版本范围”策略，开发环境同时提供带哈希的锁定版本。依赖上下界、锁文件和 B 端环境摘要必须在首次发布候选验收前固化。
- ONNX opset 固定为 18。
- 最终文件是供目标 ATC parser 使用的“MDC ONNX 方言”，仅使用 `ai.onnx` domain；ATC parser 所需的非标准算子也按工具链要求放在该 domain 下。该文件不声明可被通用 ONNX Runtime 执行，也不声明完全符合标准 ONNX 算子 schema。
- 发布验收模型的参数和浮点输入使用 FP16。库不负责加载模型，因此不提供“模型加载默认 dtype”行为。
- 模型 attention 实现必须设置为 `eager`，保证导出的图保留可供融合的小算子模式。
- 模型参数和输入数据必须位于同一设备。

## 5. 包结构

- `models`：Tiny Qwen3 Dense 和 Tiny Qwen3-MoE 的导出友好模型定义。
- `export`：将模型转换为 ATen dialect 的 `torch.fx.GraphModule`，执行图规范化和 ATen 等价替换，并标记可融合区域；不在 FX 图中引入 MDC ONNX 算子。
- `quantization`：MinMax、GPTQ、校准、量化规划和图改写。
- `onnx_export`：根据 FX 图阶段元数据和显式 mask 模式，将量化后的 FX 图导出为 MDC 适配的 ONNX。
- `config`：不依赖 torch 的量化配置定义、严格校验、JSON Schema、规范化序列化和配置指纹。
- `configs`：发布验收使用的量化配置文件。
- `mdc_ops`：MDC 自定义算子的 schema、设备实现、Fake/Meta 实现和 ONNX 符号。
- `utils`：跨模块通用工具。

配置模块的详细设计见 [designs/config.md](designs/config.md)。

## 6. 中间表示与阶段边界

### 6.1 标准交换对象

项目内部和公开 API 之间的标准图交换对象为 ATen dialect 的 `torch.fx.GraphModule`，下文简称“FX 图”。模型参数由 FX 图中的 `get_attr` 节点或子模块携带。

FX 图必须携带私有阶段元数据，阶段只允许为：

- `FLOAT_PREFILL`
- `QUANTIZED_PREFILL`
- `FLOAT_DECODE`
- `QUANTIZED_DECODE`

私有元数据使用 `mdc_llm_deploy.graph.GraphMetadata` 不可变 dataclass，挂载键固定为 `mdc_llm_deploy`，当前 `schema_version=1`。字段契约如下：

- `stage`、`model_kind`、`sequence_length` 和 decode 的 `absolute_position` 描述生命周期；`model_kind` 只允许 `dense`、`moe`。
- `input_abi`、`output_abi` 是有序 `TensorAbi` 元组，名称唯一，dtype 必须受支持，shape 必须为全正整数静态 shape。
- `boundaries` 是 `FusionBoundary` 元组；kind 只允许 `linear`、`attention`、`moe`、`rms_norm`、`rope`，节点不能被多个边界重复占有。
- `quantized_targets` 是 `QuantizedTarget` 元组，记录 FQN、target、algorithm、位宽、粒度、对称性、scale、zero point 和 GPTQ 限定回退原因；FQN 必须唯一。
- 量化 stage 必须同时具有非空 target 元数据和 64 位小写 SHA-256 配置指纹；浮点 stage 两者都不得携带。
- `properties` 仅容纳有限、JSON 兼容扩展值。新增字段或改变语义必须升级 schema 版本；未知版本立即拒绝，不做猜测迁移。

完整 validator 必须同时检查 dataclass 内部约束、FX placeholder/output 与 ABI 数量、阶段/绝对位置、模型/算法/target 能力、量化参数以及融合边界所有权。跨模块调用不得绕过该 validator 自行解释字典。

合法状态迁移为 `export → FLOAT_PREFILL`、`oneshot: FLOAT_PREFILL → QUANTIZED_PREFILL`，以及 `convert_to_decode: *_PREFILL → 对应的 *_DECODE`。重复量化、重复 decode、对 decode 图执行 `oneshot` 或把 decode 图导出为 prefill 时必须抛出 `GraphStateError`。

`oneshot` 和 `convert_to_decode` 对外保持原地修改语义，但必须先在内部候选图上完成全部检查，再一次性提交修改；失败时传入对象的图、参数和阶段元数据必须保持不变。

### 6.2 prefill 与 decode

- `export` 使用 3072 token 长序列输入生成静态 FX 图，并记录输入 ABI、可融合区域和 Q/K/V 边界元数据。
- `oneshot` 使用同形状的 prefill 数据对该 FX 图执行 fake quant。
- 调用方先对量化 FX 图执行 `onnx_export`，得到 prefill ONNX。
- 调用方随后执行 `convert_to_decode`，将同一 FX 图原地改写为 decode 图，再执行 `onnx_export` 得到 decode ONNX。
- `convert_to_decode` 在已识别或融合的 Attention Q/K/V 边界切分张量：前 3071 token 变为显式 KV Cache 输入，最后一个 token 计算当前 Q/K/V。
- decode Attention 使用 cached K/V 与当前 K/V，并输出更新后的 K/V Cache。
- `convert_to_decode` 不感知量化，浮点图和量化图均可转换；`0.1.0` 标准流程在 `oneshot` 后调用。
- prefill 校准得到的权重及激活量化参数必须复用于 decode，不允许 decode 独立校准。
- 无法完整识别 Q/K/V、建立 cache 输入输出或映射量化状态时必须显式报错。

`0.1.0` 的 decode 是仅针对绝对位置 3071 的单步等价验证图，不支持连续自回归生成。prefill ONNX 的 3072 长度 cache 不作为 decode ONNX 的输入；decode 验收 cache 由同一参考模型对输入前 3071 个 token 执行前向并在已定义的 K/V 边界提取得到。decode 输出长度为 3072 的更新 cache，只用于结果对照，不允许再次送入当前 decode 图。连续生成、运行时 `past_len`、滑窗和满 cache 策略属于后续版本。

### 6.3 标准中间 ONNX 生命周期

标准中间 ONNX 只用于隔离 PyTorch/ATen 导出问题和 MDC 方言下沉问题，不是公开 API 产物：

1. 从已通过完整 graph validator 的候选 FX 图写入同目录唯一临时文件。
2. 使用固定 ONNX 版本执行 protobuf 读取、`onnx.checker.check_model` 和 shape inference；任一步失败立即终止。
3. 仅从通过检查的内存模型下沉到 opset 18 MDC ONNX 方言；禁止从用户提供或上次运行残留的中间文件继续。
4. MDC 方言在独立临时文件完成结构检查后，以原子替换提交到目标路径。
5. 成功、失败、超时均删除标准中间 ONNX 和 MDC 临时文件。标准中间 ONNX 不入库、不发布、不传往 B 端。

标准中间 ONNX 仍是 `ai.onnx` 标准模型，必须通过标准 checker 和 shape inference；最终 MDC ONNX 含 parser 扩展语义，只执行本项目方言 validator，不声称通用 ONNX Runtime 可执行。

### 6.4 集中能力矩阵

`mdc_llm_deploy.capabilities.CAPABILITY_MATRIX` 是模型、算法、target、prefill/decode、masked/maskless 和产物层级的唯一能力真源：

- FP16 不使用量化 target；Dense/MoE 均覆盖 prefill/decode 与 masked/maskless，可进入 FX、ONNX、ATC，共 8 项发布矩阵。
- MinMax：Dense 支持 linear/attention，MoE 支持 linear/attention/moe；均覆盖 prefill/decode 与 masked/maskless，可进入 FX、ONNX、ATC，共 20 项发布矩阵。
- GPTQ：Dense 支持 linear，MoE 支持 linear/moe；只允许 FX 数值路径。即使请求组合含 phase 和 mask mode，也不得获得 ONNX 或 ATC 能力。
- Dense+moe target、GPTQ+attention、FP16+非空 target 以及矩阵外组合必须在统一能力检查处拒绝，不允许调用点散落特判。

## 7. 公开 API

### 7.1 `export`

签名：

```python
export(
    model: torch.nn.Module,
    example_inputs: Mapping[str, torch.Tensor],
) -> torch.fx.GraphModule
```

职责：

- 接收调用方已经加载的 Transformers/PyTorch 模型和显式示例输入。
- 不负责下载模型、加载 tokenizer 或构造业务数据。
- 使用长序列示例输入将模型转换为静态 ATen FX 图。
- 执行图规范化和 ATen 等价替换，识别 RmsNorm、RoPE、Attention、MoE 等模式并记录融合元数据；真正的 MDC 自定义节点只在 `onnx_export` 中生成。

返回值：`torch.fx.GraphModule`。

### 7.2 `oneshot`

签名：

```python
oneshot(
    graph: torch.fx.GraphModule,
    config: QuantizationConfig | Mapping[str, object] | str | Path,
    calibration_dataloader: Iterable[Mapping[str, torch.Tensor]],
) -> torch.fx.GraphModule
```

职责：

- 接收 FX 图、量化配置和校准 dataloader。
- 使用 prefill 数据执行 MinMax 或 GPTQ。
- 原地修改传入的 FX 图，并返回同一个 `GraphModule` 对象。

调用方需要保留浮点基线时必须重新调用 `export`。`0.1.0` 不提供或承诺通用 FX 图复制能力。

校准 dataloader 的每个 batch 必须是 `Mapping[str, Tensor]`，键与 FX placeholder 名称一致。发布验收只使用一个 batch：`input_ids` 为 shape `[1, 3072]` 的 int64 Tensor，使用 NumPy PCG64、seed `20260714`，在 `[0, 128)` 上均匀采样；fixture 保存为规范 little-endian 字节并记录 SHA-256。不得包含 padding。模型必须处于 `eval()` 状态。

### 7.3 `convert_to_decode`

签名：

```python
convert_to_decode(
    graph: torch.fx.GraphModule,
) -> torch.fx.GraphModule
```

职责：

- 接收 prefill FX 图，在融合后的 Q/K/V 边界构造 decode 输入、当前 token 计算和 KV Cache 输出。
- 原地修改传入的 FX 图，并返回同一个 `GraphModule` 对象。
- 根据 Attention 的最终量化状态推导 K/V Cache dtype 和量化参数，不引入独立 Cache 配置模型。

返回值：传入的 `torch.fx.GraphModule`。

### 7.4 `onnx_export`

签名：

```python
onnx_export(
    graph: torch.fx.GraphModule,
    output_path: str | Path,
    *,
    mask_mode: Literal["masked", "maskless"],
    overwrite: bool = False,
) -> onnx.ModelProto
```

职责：

- 接收 prefill 或已转换的 decode FX 图、输出路径和显式 mask 模式；stage 从图阶段元数据推导。
- 直接将 fake quant 语义转换为 MDC 量化算子，不在最终 ONNX 中生成或保留 `QuantizeLinear`/`DequantizeLinear`。
- 将 position ids、cache position、RoPE cos/sin 和可选 attention mask 等静态张量折叠为 initializer。
- 写出 `.onnx` 文件并返回 `onnx.ModelProto`。
- 在临时路径完成写入和全部检查后原子替换目标文件。目标已存在且 `overwrite=False` 时抛出 `FileExistsError`；任何失败不得留下不完整目标文件。
- 最终 MDC ONNX 方言执行 protobuf/IR 完整性、SSA、拓扑、静态 shape 元数据、domain 白名单、扩展 `MatMul`、MDC 自定义算子结构和残留 QDQ 检查。由于非标准节点位于 `ai.onnx` domain，不要求最终文件通过通用 `onnx.checker.check_model` 的标准算子 schema 检查；标准 ATen 中间图必须通过固定 ONNX 版本的 checker 和 shape inference。

该 API 产出的模型只能描述为“面向 MDC 的 ONNX”。仅在对应模型通过 B 端 ATC 转换后，才能描述为“已通过 ATC 编译”；在完成 MDC 真机验证前，不得描述为“可直接部署至 MDC”。

公开异常类型固定为：

- `MdcDeployError`：库异常基类。
- `GraphStateError`：图阶段或重复调用非法。
- `UnsupportedPatternError`：模型模式、融合边界或 cache 构造失败。
- `QuantizationConfigError`：配置格式、类型或算法内在冲突。
- `OnnxExportError`：MDC 方言生成或结构检查失败。

## 8. MDC 自定义算子

`0.1.0` 必须覆盖以下全部 GE 算子，并使用 [ops](ops) 中定义的 ATC parser ONNX 名称：

- `ApplyRotaryPosEmb`
- `FusedInferAttentionScore`
- `RmsNorm`
- `AscendQuantV2`
- `AscendDequant`
- `MoeExpert`

算子的输入、输出、属性和 MDC ONNX 方言 schema 以 [ops](ops) 中的文档为准。GE 名称与 ONNX `op_type` 不同时，结构检查以文档中的 ONNX `op_type` 为准。

每个算子必须提供：

- PyTorch CPU 前向参考实现，作为发布阻塞验证项。
- GPU Triton 实现及设备分派。
- 基于 Ascend Triton 的 NPU 实现及设备分派。
- Fake/Meta 实现，支持 FX 图捕获、形状和 dtype 推导。
- opset 18 对应的 MDC ONNX 符号。
- 独立参考实现对照的前向数值测试。

GPU/NPU Triton kernel 必须交付实现。GPU kernel 必须在 B 端完成运行验证，不允许跳过；NPU kernel 不属于 `0.1.0` 运行验收范围，其实现保持未验证状态，但必须通过模块导入、设备分派注册、Fake/Meta 对照和源码静态检查，禁止以 `pass`、无条件 `NotImplementedError` 或直接调用 CPU reference 充当实现。

各算子的 CPU/GPU 数值阈值必须在对应 `docs/ops` 文档中按 dtype 独立定义；Attention 类算子同时定义余弦相似度阈值。测试使用 seed `20260714`，同时覆盖零值、饱和值、最小合法 shape 和发布验收 shape；输入出现 NaN/Inf 时必须显式报错，不能静默传播。

自定义算子不提供 autograd 实现。对其执行反向传播时必须给出明确错误。

## 9. 量化设计

### 9.1 通用要求

- 量化类型为训练后量化（PTQ）。
- FX/eager 阶段统一使用 fake quant，不直接引入 ONNX 量化算子或 MDC 量化下沉算子。MDC 自定义节点统一在 `onnx_export` 阶段生成。
- Attention 和 MoE 的 ATen 子图必须保留可识别的量化边界；`onnx_export` 分别下沉到 `FusedInferAttentionScore` 和 `MoeExpert` 的量化接口。
- 首版只验收量化实现正确性，不以 Tiny 模型的任务精度作为验收依据。
- 量化整数范围为有符号 int8 `[-128, 127]` 和有符号 int4 `[-8, 7]`。
- 量化公式为 `q = clamp(round(x / scale) + zero_point, qmin, qmax)`，反量化公式为 `x_hat = (q - zero_point) * scale`。
- 舍入采用 ties-to-even 语义，CPU、GPU 和 ONNX 导出适配必须一致。
- scale 和 zero point 的统计与计算统一使用 FP32；遇到 NaN/Inf 立即报错。
- 对称量化使用 `zero_point = 0` 和 `scale = max(abs(xmin), abs(xmax)) / qmax`。全零范围固定使用 `scale = 1.0`、`zero_point = 0`，量化结果为零。
- 非对称量化先令 `xmin = min(xmin, 0)`、`xmax = max(xmax, 0)`，再计算 `scale = (xmax - xmin) / (qmax - qmin)` 和 `zero_point = clamp(round(qmin - xmin / scale), qmin, qmax)`。包含零后的退化范围只可能是全零，按上一条处理。
- 所有 `round` 均在 FP32 中执行 ties-to-even；zero point 逻辑存储类型为 int32。下沉到要求浮点 offset 的 MDC 算子时才转换为该算子要求的浮点 dtype。

### 9.2 配置模型

- 配置对象为不依赖 torch 的不可变 dataclass，所有层级严格拒绝未知字段。
- `QuantizationConfig` 包含有序 modifier 链；支持 `minmax` 和 `gptq`。
- 根级和 modifier 级均支持 FQN `include`/`exclude`；局部选择器完整替换根级选择器。
- 每个 modifier 可分别配置 `linear`、`attention` 和 `moe`。
- `linear` 包含 `weight` 和 `activation`；`attention` 包含 `query`、`key`、`value` 和 `score`；`moe` 包含 `weight` 和 `activation`。
- GPTQ modifier 配置 `attention` 时必须立即报错；GPTQ 可配置 `linear` 和 `moe`。
- config 只拒绝格式、类型和算法内在冲突；后端不支持的组合在 ONNX 导出或 ATC 阶段报错。
- `QuantizationConfig` 不提供预设构造方法。仓库根目录 `configs` 提供验收所需的 `minmax-linear-w8a8.json`、`minmax-attention-a8.json`、`minmax-moe-w8a8.json`、`gptq-linear-w4a8.json` 和 `gptq-moe-w8a8.json`。
- 配置模块提供完整 SHA-256 指纹：对解析后的完整配置使用 UTF-8、键排序和紧凑分隔符生成规范 JSON 后计算。
- 包内发布 `mdc_llm_deploy/config/schema.json`，测试必须验证该文件与代码生成的 Schema 一致。
- 同一个 modifier 链允许 MinMax 和 GPTQ 作用于互不重叠的 target；同一 target 被多个 modifier 命中时立即报错。禁止对已经执行过 `oneshot` 的图再次叠加算法。

### 9.3 MinMax

- linear、attention 和 moe 使用相互独立的验收配置，不要求组合量化配置通过发布验收。
- linear 权重验收配置为 int8、per-channel、对称量化；激活配置由验收 JSON 明确给出。
- attention 的 Q、K、V 和 softmax score 四条激活边独立配置；缺失的边保持浮点。
- `0.1.0` 的 MDC ONNX 方言只支持 Q 和 softmax score 的对称量化；非对称 Q/score 可完成 FX fake-quant，但调用 `onnx_export` 时必须抛出 `OnnxExportError`。K/V 可使用独立 offset 表达非对称量化。
- moe 验收配置使用 int8 静态 per-tensor 输入和 int8 per-tensor 专家权重。Tiny MoE 将 4 个 routed expert 和 1 个 shared expert 都计入 `expert_num=5`；shared expert 以固定权重 1.0 追加到每个 token 的 routed top-2 列表。`quant_scales` shape 为 `[1 + expert_num × 4]`，顺序为全局输入激活 scale，随后按 expert id 依次存放 `gate_proj`、`up_proj`、中间激活和 `down_proj` scale；非对称量化时 `quant_offsets` 使用完全相同的顺序和 int32 dtype。
- 静态 per-token 参数按绝对 token 位置存储，校准 batch 对应 shape 必须一致，多个 batch 按位置聚合 min/max。

### 9.4 GPTQ

- linear GPTQ 验收配置为 int4、per-channel、对称权重量化和 int8 激活量化。
- moe GPTQ 验收配置使用 int8 per-tensor 对称权重量化和 int8 静态 per-tensor 激活量化。
- GPTQ 保留 `percdamp=0.01`、`actorder=true`、`block_size=128`。clip ratio 为包含两端点的 20 个 FP32 值：`0.5 + i × 0.5 / 19`，`i=0..19`。
- Hessian 加入 `percdamp × mean(diag(H))` 后执行 Cholesky；矩阵含非有限值或 Cholesky 失败时，该 FQN 回退为同位宽、同粒度的对称 MinMax 权重量化，并在结果元数据中记录 FQN 和原因。其他未预期异常不得静默回退。
- GPTQ 不支持 attention；配置时必须报错。
- GPTQ-linear W4 和 GPTQ-moe 只执行 FX 数值验收。调用 `onnx_export` 时必须明确报错，不进入 ATC 验收矩阵。

### 9.5 ONNX 量化下沉

- W8 linear 导出结构固定为：`NPUAscendQuantV2` 将浮点激活转换为 int8；权重在导出期固化为 int8 initializer；目标 ATC parser 按扩展语义将 `ai.onnx::MatMul` 的 int8×int8 结果解释为 int32 累加；`AscendDequant` 恢复到原激活 dtype；bias 在反量化后执行浮点 `Add`。该 `MatMul` 是 MDC ONNX 方言扩展，不满足标准 ONNX `MatMul` 类型约束，必须通过 B 端最小 parser 样例和 ATC 验收，不得交给通用 ONNX 执行器。
- `NPUAscendQuantV2` 使用乘法 scale，即仿射量化 scale 的倒数；offset 使用与输入激活相同 dtype 的浮点 zero point。
- FP16 激活的仿射 scale 不得小于 `1 / 65504`，避免倒数转换为 FP16 后溢出。
- per-token linear 的 `NPUAscendQuantV2.axis` 为 `-2`。
- per-tensor linear 将激活 scale 与权重 scale 的乘积编码给 `AscendDequant`；per-token linear 只编码权重 scale，并在 `AscendDequant` 后使用 `Mul` 应用激活 scale。
- `AscendDequant.deq_scale` 使用 uint64：完整 FP32 bit pattern 放在低 32 位，高 32 位清零，不执行 s19 尾数截断。

### 9.6 KV Cache

- K Cache dtype 由最终 `attention.key` 量化状态决定，V Cache dtype由最终 `attention.value` 量化状态决定，允许二者不同。
- `attention.key` 或 `attention.value` 未配置量化时，对应 Cache 保持原浮点 dtype；配置量化时使用对应整数 dtype。
- `0.1.0` 的量化 KV Cache 只使用静态参数，scale/zero point 固化为 ONNX initializer。
- decode 转换时，普通当前-token激活使用位置 3071 的量化参数；量化 K/V Cache 保留前 3071 个位置的参数，位置 3071 用于当前 K/V 和更新后的 Cache。
- ApplyRotaryPosEmb 使用 BSND，随后显式转置为 BNSD。Cache 存储边界固定为转置后的 RoPE K，以及 value projection 后转置为 BNSD、进入 Attention 前的 V；Q 不进入 cache。
- Cache layout 固定为 `BNSD`。Tiny 模型每层 K/V 的 prefill 输出 shape 为 `[1, 2, 3072, 16]`，decode 输入 shape 为 `[1, 2, 3071, 16]`，decode 输出 shape 为 `[1, 2, 3072, 16]`。
- per-tensor K/V 参数为标量；静态 per-token 参数 shape 为 `[1, 1, 3072, 1]`，按绝对位置索引并在 head 与 head-dim 维广播。K/V 参数不同时使用 `key_antiquant_*` 和 `value_antiquant_*`，相同时允许使用共用 `antiquant_*`。
- `0.1.0` 的 MDC ONNX decode 只支持浮点或 int8 KV Cache。int4 key/value 可在 FX fake-quant 阶段表达，但调用 `onnx_export` 导出 decode 时必须明确报错。

### 9.7 后续能力

以下能力可以由配置模型表达或在后续版本实现，但不属于 `0.1.0` 发布阻塞项：

- 其他位宽组合。
- 非对称权重量化。
- SpinQuant 等离群值抑制算法。
- W4 ONNX 表示和 ATC 转换。

## 10. 静态形状与验证输入

- batch 固定为 1。
- 最大总序列长度固定为 3072。
- prefill 验收输入为 3072 token。
- decode 验收输入为 1 个新 token 和长度为 3071 的 KV cache。
- decode 新 token 的绝对位置为 3071。
- position ids、cache position、RoPE cos/sin 和 attention mask 均不是 ONNX 运行时输入，而是静态 initializer。
- masked 模式使用 BOOL causal attention mask initializer，非零位置表示屏蔽；FusedInferAttentionScore 使用 `sparse_mode=0`，不再叠加隐式 sparse mask。prefill mask 为 shape `[1, 1, 3072, 3072]` 的严格上三角，decode mask 为 shape `[1, 1, 1, 3072]` 的全零 Tensor。maskless 模式不提供 attention mask，`sparse_mode=0`、`pre_tokens=next_tokens=2147483647`，执行全可见 non-causal Attention。
- prefill ONNX 的运行时输入仅为 `input_ids`；decode ONNX 的运行时输入为 `input_ids` 和各层 past K/V。
- prefill/decode 输出均包含 logits 和各层更新后的 K/V。
- 不承诺动态 shape。

ONNX I/O ABI 固定为：

- `input_ids`：int64，prefill `[1, 3072]`，decode `[1, 1]`，值域 `[0, 128)`。
- prefill 输出 `logits`：与模型浮点 dtype 相同，shape `[1, 3072, 128]`。
- decode 输出 `logits`：与模型浮点 dtype 相同，shape `[1, 1, 128]`。
- 第 `i` 层 cache 输入依次命名为 `past_key_values.{i}.key`、`past_key_values.{i}.value`；输出依次命名为 `present.{i}.key`、`present.{i}.value`。
- 输出顺序固定为 `logits`，随后按层号递增排列每层 key、value。

Tiny Qwen3 Dense 固定为：1 层、vocab size 128、hidden size 64、intermediate size 128、4 个 Query head、2 个 KV head、head dim 16、max position embeddings 3072、`rms_norm_eps=1e-6`、`rope_theta=1_000_000`、`hidden_act="silu"`、所有线性层无 bias、attention/embedding dropout 为 0、`tie_word_embeddings=false`、`use_cache=true`、initializer range 0.02。权重使用 PyTorch CPU generator、seed `20260714` 初始化后再转换到验收 dtype，模型处于 `eval()` 状态。

Tiny Qwen3-MoE 沿用 Dense 主干，包含 4 个 routed expert、top-k 2、每专家 intermediate size 64 和 1 个 shared expert。Router 对 4 个 routed expert 执行 softmax，选择 top-2 后将两项权重重新归一化为和 1；shared expert 输出以权重 1.0 无条件相加，不使用独立 shared gate。Router 归入 linear 配置；所有进入 `MoeExpert` 的 routed/shared expert 模块归入 moe 配置。

## 11. 验收标准

### 11.1 模型级验收

Tiny Qwen3 Dense 和 Tiny Qwen3-MoE 均必须完成：

1. 3072 token 模型导出为静态 FX 图。
2. 各验收配置独立执行原地量化。
3. FP16 基线图和量化图分别导出 prefill ONNX。
4. FP16 基线图和量化图分别执行 `convert_to_decode` 并导出 decode ONNX。
5. 标准 ATen 中间图通过 ONNX checker/shape inference；最终 MDC ONNX 方言通过 IR、静态 shape、domain、扩展 `MatMul`、残留 QDQ 和自定义算子结构检查。
6. GPU 自定义算子数值验证。
7. FP16 和属于 MinMax ATC 矩阵的量化模型均纳入 ONNX 导出、B 端 ATC 编译和 OM 生成验证。

MinMax 与 GPTQ 的发布验收配置使用各自独立导出的 FX 图，不能对同一张已执行 `oneshot` 的图再次叠加。单次 `oneshot` 的有序 modifier 链只允许不同算法命中互不重叠的 target。

FP16 与 MinMax 量化模型的 ONNX/ATC 验收矩阵固定为：

- FP16 基线：Tiny Dense 和 Tiny Qwen3-MoE 分别导出 prefill/decode 和 masked/maskless，共 `2 × 2 × 2 = 8` 份 ONNX。
- Tiny Dense：linear、attention 两套配置。
- Tiny Qwen3-MoE：linear、attention、moe 三套配置。
- 每套 MinMax 配置分别导出 prefill/decode 和 masked/maskless，共 `5 × 2 × 2 = 20` 份量化 ONNX。
- 8 份 FP16 ONNX 和 20 份量化 ONNX 均必须完成导出、ATC 编译和 OM 生成。单项失败应标记 `BLOCKED`，继续执行其余项目以收集证据，但在修复并转为 `PASS` 前阻塞发布；不得以 `WAIVED` 或“跳过”替代发布矩阵门禁。
- 文件名为 `{model}-{algorithm}-{target}-{mask-mode}-{stage}-{config_hash8}-{commit8}.onnx/.om`；完整配置指纹、commit SHA、工具链和产物 SHA-256 记录在 B 端文本摘要中。

GPTQ-linear 和 GPTQ-moe 仅执行 FX 验收，不生成 ONNX/OM。

### 11.2 量化正确性验收

- scale、zero point、量化整数范围和反量化结果与独立参考实现一致。
- MinMax 和 GPTQ 的 target 选择符合配置。
- MoE 的 routed/shared expert、router 和各 expert 的 `gate_proj` 不得被错误遗漏或归入错误配置组。
- prefill 的位置量化参数能够被 decode 按绝对位置正确复用。
- 原始模型与浮点 FX 图对比 `logits` 和全部 K/V：FP32 `atol=rtol=1e-5`，FP16 `atol=rtol=1e-3`。
- 量化 FX 与独立 fake-quant reference 对比：量化整数必须完全一致，FP16 反量化结果 `atol=rtol=1e-3`。
- 使用同一 3072-token fixture：长序列 prefill 的最后 token 与“前 3071 token 参考 cache + 最后 1 token”decode 的 logits 和 position 3071 K/V 使用相同 dtype 阈值。
- 不要求 Tiny 模型量化前后 logits、困惑度或任务指标满足质量阈值。

### 11.3 自定义算子验收

- CPU 前向结果与独立 PyTorch 参考实现满足约定误差。
- GPU 前向结果在 B 端与独立 PyTorch 参考实现满足约定误差。
- Fake/Meta 输出的 shape 和 dtype 与真实前向一致。
- ONNX 符号生成的 domain、op_type、输入输出和属性符合 `ops`。
- NPU 实现和设备分派必须满足第 8 节的最低静态验收，但不记为已通过运行验证。

### 11.4 B 端验收流程

- A 端负责代码开发并推送到远端仓库。
- B 端只负责拉取 A 端已推送的指定 commit 并执行验证；不得修改代码、创建或上传提交、推送分支或向 A 端回传任何产物。
- GPU 自定义算子测试、ATC 编译和 OM 模型生成必须在 B 端执行。
- B 端实际安装环境是本次验收基准。每次必须在文本摘要中记录 commit SHA、工作区是否干净、配置 SHA-256、Python 和锁文件 hash、GPU 型号/驱动、PyTorch/Triton、ATC/CANN/OPP 版本、目标 SoC、完整 ATC 参数、每项退出码以及 ONNX/OM SHA-256。
- B 端产生的任何文件不得传回 A 端。验证结束后按 B 端保留策略处理必要 ONNX、OM 和本地日志，并清理其他中间文件；A 端只接收纯文本摘要。
- A 端只能接收 B 端返回的文本摘要，不接收 B 端文件。摘要必须包含上一条规定的结构化字段和 28 项矩阵结果；A 端将可复用的环境摘要、发现、根因和解决方案整理到 `docs/validation/b-side.md`，原始日志不得复制入库。
- ONNX 导出、ATC 编译或 OM 模型生成失败时必须先排查并尝试修复。摘要必须记录失败阶段、完整命令、退出码、关键错误、已尝试方案和根因判断，并将该项标记为 `BLOCKED`；可继续验收其他项目，但不得据此发布。
- ATC 成功固定定义为：命令在规定超时内以退出码 0 结束、没有“不支持算子/回退到未批准实现”类告警、生成非空 OM，且日志确认所有 MDC 自定义节点映射到预期 GE 算子。精确命令模板、SoC、输入 shape、precision mode、OPP 路径和超时在首次发布候选前固化到 `docs/validation/b-side.md`。

### 11.5 结果状态与不可豁免门禁

- `PASS`：该项全部必需命令退出码为 0，产物和日志检查均满足契约，证据字段完整。
- `BLOCKED`：外部环境、工具链、依赖或待修复缺陷使必需证据无法取得。`BLOCKED` 不是失败豁免，也不是“尚未执行”的美化；任何发布阻塞项处于该状态时不得发布。
- `WAIVED`：仅适用于 PRD 明确列为非发布范围或信息性附加检查的项目，必须记录范围、原因、影响、批准人和批准日期。不得用于下列不可豁免门禁。

不可豁免门禁包括：A 端 Python 3.12 全量限定测试；B 端 Python 3.11 或 3.12 GPU/ATC 验证；候选图完整验证与事务失败不变性；标准中间 ONNX checker/shape inference；最终 MDC 方言结构检查；GPTQ ONNX 明确拒绝；六个自定义算子 CPU reference；B 端六算子 GPU reference 对照；六个自定义节点和扩展 INT8 `MatMul` parser 探针；28 项发布矩阵的 ATC 成功定义。任一未 `PASS`，发布状态必须是 `BLOCKED`。

### 11.6 不在当前范围内的验收

- MDC 真机运行、性能和内存指标。

这些项目在后续版本单独定义兼容性矩阵。

### 11.7 发布措辞边界

- 仅完成 A 端导出和结构检查：只允许描述为“面向 MDC 的 ONNX”。
- 对应矩阵项满足 B 端 ATC 成功定义：允许描述为“已通过 ATC 编译”。
- 未完成 MDC 真机验证前，禁止使用“可部署到 MDC”“可在 MDC 运行”“真机已验证”或同义措辞。

## 12. 明确不承诺的兼容性

- FX 图只作为进程内交换对象，`0.1.0` 不承诺复制、保存、加载或跨版本兼容。
- 包元数据允许 Python `>=3.11,<3.13`；A 端在 Python 3.12 执行发布门禁，B 端在 Python 3.11 或 3.12 执行 GPU/ATC 门禁。
- B 端工具链升级后，同一提交的 ATC 结果可能变化；`0.1.0` 不承诺跨工具链版本结果相同，但每次结果必须能在其记录的环境版本上复现。

