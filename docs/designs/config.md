# config

`config` 是不依赖 torch 的量化配置单一真源。它定义不可变 Python 对象、校验 JSON 兼容输入，并序列化已解析的默认值。图发现、数值校准和 QDQ 插入不属于该模块。

## 配置模型

`QuantizationConfig` 包含：

- `modifiers`：由 `minmax` 或 `gptq` 操作组成的有序链。
- `include`：根级 FQN 模式；空列表选中所有已发现 target。
- `exclude`：从选中 target 中排除的根级 FQN 模式。

空 modifier 链是显式 no-op。

每个 modifier 可定义：

- `include` 和 `exclude`：可选的局部选择器。
- `linear`：线性 target 的权重和激活量化配置。
- `attention`：query、key、value 和 score 边的激活量化配置。
- `moe`：进入 `MoeExpert` 的 routed/shared expert 权重和激活量化配置；router 不属于 `moe`，归入 `linear`。

modifier 局部选择器一旦提供，会完整替换对应根级字段，而不是与其合并。字段为 `null` 或缺失时继承根级值；空 `include` 显式匹配全部 target，空 `exclude` 显式不排除任何 target。

modifier 按声明顺序执行。每个 modifier 基于前一个 modifier 产出的图重新发现 target。引擎累积 target 状态，并在整条链执行完成后统一物化 QDQ 节点。

不同 modifier 可以使用不同算法，但同一 FQN target 只能被一个 modifier 命中。规划阶段发现重叠时必须抛出 `QuantizationConfigError`，不得采用“后者覆盖前者”。该约束依赖实际 target 发现，因此由 planner 校验，不由纯配置解析器猜测。

## JSON 结构

```json
{
  "include": [],
  "exclude": ["lm_head"],
  "modifiers": [
    {
      "type": "minmax",
      "include": ["lm_head"],
      "exclude": [],
      "linear": {
        "weight": {
          "bits": 8,
          "granularity": "per_channel",
          "symmetric": true
        },
        "activation": {
          "bits": 8,
          "granularity": "per_tensor",
          "mode": "static",
          "symmetric": true
        }
      }
    },
    {
      "type": "gptq",
      "include": ["model.layers.0"],
      "linear": {
        "weight": {
          "bits": 4,
          "granularity": "per_channel"
        },
        "activation": {
          "bits": 8,
          "granularity": "per_tensor",
          "mode": "static",
          "symmetric": true
        }
      },
      "percdamp": 0.01,
      "actorder": true,
      "block_size": 128
    }
  ]
}
```

所有层级都会拒绝未知字段。必填叶子字段不可缺失。`symmetric` 缺失时解析为 `true`。target 或 tensor 叶子缺失或为 `null`，表示该范围没有量化配置。

## 字段约束

### 权重

- `bits`：`4` 或 `8`。
- `granularity`：`per_tensor` 或 `per_channel`。
- `symmetric`：布尔值，默认 `true`。

### 激活

- `bits`：`4` 或 `8`。
- `granularity`：`per_tensor` 或 `per_token`。
- `mode`：`static` 或 `dynamic`。
- `symmetric`：布尔值，默认 `true`。

### 注意力

`attention.query`、`attention.key`、`attention.value` 和 `attention.score` 分别接受独立的激活配置。缺失的边保持浮点。

配置模型允许四条边表达对称或非对称量化。`0.1.0` 的 MDC ONNX 方言只支持 Q 和 score 的对称量化；非对称 Q/score 配置可完成 FX fake-quant，但 `onnx_export` 必须抛出 `OnnxExportError`。K/V 非对称量化通过各自的 antiquant offset 表达。

### MoE

`moe.weight` 接受权重配置，`moe.activation` 接受激活配置。Qwen3-MoE 的 routed
expert 由 `moe` 选择；router 继续由 `linear` 选择。expert 数量与 top-k 从模型配置
和 routing shape 推导，不注入 shared expert。

MinMax-MoE 发布配置固定为 int8 静态 per-tensor 激活和 int8 per-tensor 对称权重。GPTQ-MoE 发布配置固定为 int8 静态 per-tensor 激活和 int8 per-tensor 对称权重。其他 schema 可表达但后端未支持的组合，由 `onnx_export` 给出明确错误。

### GPTQ

- `percdamp`：非负数，默认 `0.01`。
- `actorder`：布尔值，默认 `true`。
- `block_size`：正整数，默认 `128`。
- `linear.weight.granularity`：GPTQ 只接受 `per_channel`。
- `moe.weight.granularity`：GPTQ-MoE 只接受 `per_tensor`。
- GPTQ modifier 不允许出现非 `null` 的 `attention`；`from_dict` 或 `load` 必须立即抛出 `QuantizationConfigError`。
- GPTQ 的 `linear` 或 `moe` 至少提供一个，且对应 target 必须提供 `weight`；activation 可选，但 W4A8/W8A8 发布验收配置必须显式提供静态 int8 activation。

GPTQ 字段只允许用于 `type` 为 `gptq` 的 modifier。

## 选择器语义

选择器作用于模块完全限定名（FQN）。模式匹配由点分隔的连续分量序列，不匹配任意字符串子串。

- `layers.1` 匹配 `model.layers.1.self_attn`。
- `layer` 不匹配 `model.layers.1`。
- `include` 和 `exclude` 同时命中时，`exclude` 优先。
- 空字符串模式永不匹配。

## 构造与序列化

`QuantizationConfig.load` 接受：

- 已有的 `QuantizationConfig`；
- 字典；
- 指向 UTF-8 JSON 文件的字符串或 `Path`。

`from_dict` 校验字典输入。`to_dict` 输出已解析的全部默认值。`to_json_string` 输出 UTF-8、键排序、缩进为 2 空格并以单个换行结束的可读 JSON。

`QuantizationConfig` 不提供预设构造方法。版本化验收配置只来自仓库根目录
`configs/quantization/*.json`，避免代码默认值和发布配置分叉。

## JSON Schema

- schema 使用 JSON Schema Draft 2020-12。
- 代码中的不可变 dataclass 是 schema 的生成源；发布文件固定为 `mdc_llm_deploy/config/schema.json`。
- 所有 object 都生成 `additionalProperties: false`；必填叶子进入 `required`，可选 target 使用显式 `null` union。
- `type` 使用 `minmax`、`gptq` 的判别 union，并在 GPTQ 分支中禁止非 `null` 的 `attention`。
- 测试必须重新生成 schema 并与包内文件逐字节比较。

## 配置指纹

`fingerprint` 是只读属性，返回 64 个小写十六进制字符。计算步骤固定为：

1. 调用 `to_dict()` 得到包含全部解析默认值且不含运行时状态的对象。
2. 使用 `json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)`。
3. 将结果编码为 UTF-8，不追加换行。
4. 对字节串计算 SHA-256。

输入 JSON 的空白、键顺序和显式/隐式默认值不得改变指纹。

## 模块边界

- `config/specs.py`：tensor 与 target 配置。
- `config/modifiers.py`：modifier 配置类型。
- `config/config.py`：严格解析、序列化、JSON Schema 和配置指纹。
- `quantization/selectors.py`：FQN 匹配和选择器继承。
- `quantization/planner.py`：把选择器编译为 target 工作集。
- `quantization/engine.py`：有序执行和最终物化。
- `capabilities.py`：模型、算法、target、phase、mask mode 和产物层级的集中能力矩阵。
- `graph.py`：版本化 `GraphMetadata` 和跨模块完整 validator；只接受 planner/engine 物化后的强类型 target 元数据。

配置包必须保持无 torch 依赖，使工具和未加载模型运行时的部署环境也能执行配置校验。

## 配置校验与能力校验边界

- `config` 只判断 JSON 结构、字段类型和算法内在冲突，不读取模型，不根据后端能力删改配置。
- planner 完成 FQN 发现、选择器应用和 modifier 重叠检查。
- engine 在候选图上物化 `QuantizedTarget`，写入 algorithm、target、位宽、粒度、scale、zero point、配置指纹和限定回退原因。
- `graph.validate_metadata` 对物化结果执行跨模块约束，并调用集中能力矩阵检查 model/algorithm/target/phase；矩阵外组合立即失败。
- mask mode 和请求产物在导出入口通过同一能力矩阵检查。GPTQ 无论 linear 或 moe 都只具有 FX 能力，不得进入 ONNX/ATC。
- 只有候选图通过 FX lint、重编译、metadata、ABI、能力和量化参数全部检查后，事务才以一次状态替换提交。任一步失败，原图代码、参数和 metadata 均保持不变。
