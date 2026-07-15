# B 端验证记录

## 当前状态

- 状态：`BLOCKED`
- 原因：已完成部分 ATC 探针，但尚未以修复后的指定 commit 重跑完整八项门禁和 28 项矩阵。
- 已验证结论：CANN 9.1.0、SoC `MC62CM12AA` 已接受下方 Attention 与 MoE 探针并生成 OM；不代表 MDC 真机已验证。
- 解锁条件：A 端提交并推送修复后，B 端拉取指定 commit，重跑完整门禁和发布矩阵。

## 2026-07-15 ATC 探针事实

- 量化 `FusedInferAttentionScore`：原导出缺少 slot 7 `dequant_scale1` 和 slot 9 `dequant_scale2`，ATC 报 `deqScale1, quantScale1 or deqScale2 is nullptr`。补齐两个乘积 scale 后编译成功。
- `MoeExpert` 失败规格：`hiddenSize=64/128`，均在 tiling 阶段失败；序长缩短、删除 offset、调整 TopK 均不能消除失败。
- `MoeExpert` 通过规格：`hiddenSize=256`、`expertInterDim=128/256`、`topkExpertNum=3`、五专家、offset 存在；`tokenNum=8/3072` 均编译成功并生成 OM。
- 结论：发布 MoE 测试模型固定使用 `hiddenSize=256`、`expertInterDim=128`；`hiddenSize=64` 的 Tiny 默认配置只用于本地功能测试，不作为 ATC 发布规格。
- ATC 对 `FusedInferAttentionScore` 持续输出 `output index 1 shape:[0] not valid`。将 `softmax_lse_flag` 改为 true 或省略第二输出后告警仍存在，判定为当前 OPP parser 的固定行为；完整门禁结论继续记为 `BLOCKED`，不得据此宣称无告警通过。
- 本轮只完成 ATC 编译，没有执行 MDC 真机推理。

## 权限与数据边界

1. A 端负责开发、提交和推送。B 端只拉取 A 端指定 commit。
2. B 端不得修改代码、创建或上传提交、推送分支，工作区必须在验证前后保持干净。
3. B 端生成的 ONNX、OM、日志、缓存和输入全部留在 B 端，不向 A 端回传文件。
4. A 端只接收下方格式的纯文本摘要。原始日志不得粘贴入库，只摘录定位所需的最短错误。
5. 阶段 0 不执行 MDC 真机验证，不产生“可部署”或“真机已验证”结论。

## 阶段 0 最小 parser 探针

所有探针固定使用 ONNX opset 18、`ai.onnx` domain、静态 shape。每条 ATC 命令超时固定为 1800 秒；超时后终止该命令，记录退出状态和最短关键错误，不复用半成品。

必须分别生成并编译七个最小模型，不能把多个待测节点合并后只报告一次：

1. `NPURmsNorm`：显式 `epsilon=1e-6`，确认映射到 GE `RmsNorm`。
2. `ApplyRotaryPosEmb`：`layout=1`、`rotary_mode="half"`、BSND，确认映射到同名 GE 算子。
3. `FusedInferAttentionScore`：BNSD、`num_heads=4`、`num_key_value_heads=2`、`scale=0.25`、`sparse_mode=0`，确认映射到同名 GE 算子。
4. `NPUAscendQuantV2`：`axis=-1`、`dtype=2`，确认映射到 GE `AscendQuantV2`。
5. `AscendDequant`：INT32 输入、UINT64 scale、显式 `dtype`，确认映射到同名 GE 算子。
6. `MoeExpert`：五个必选输入和可选 `quant_offsets` 分别探测，确认映射到同名 GE 算子。
7. 扩展 `MatMul`：INT8×INT8 输入、INT32 累加结果，确认目标 parser 接受 MDC 扩展语义，不按标准 ONNX 类型约束拒绝。

每个模型必须先记录 protobuf 可读、domain/op_type、opset、输入输出 dtype/shape 和属性，再调用 ATC。ATC 成功必须同时满足：1800 秒内退出码 0；产生非空 OM；无不支持算子或未批准回退告警；日志明确显示预期 GE 映射。缺少任一证据即为 `BLOCKED`，不得标记 `PASS`。

## B 端执行前环境记录模板

```text
validation_id: <YYYYMMDD-HHMMSS>
status: <PASS|BLOCKED>
commit_sha: <40位SHA>
commit_requested_by_a: <40位SHA>
branch: <分支名>
workspace_clean_before: <true|false>
workspace_clean_after: <true|false>
python: <必须为3.11.x或3.12.x>
dependency_lock_sha256: <SHA-256或BLOCKED:锁文件不存在>
config_sha256: <SHA-256；阶段0无配置时填N/A>
os: <名称和版本>
gpu: <型号>
gpu_driver: <版本>
pytorch: <版本>
triton: <版本>
atc: <版本>
cann: <版本>
opp: <版本与绝对路径>
soc_version: <ATC接受的精确值>
precision_mode: <精确值>
timeout_seconds: 1800
artifact_returned_to_a: false
code_changed_on_b: false
```

`commit_sha` 必须等于 `commit_requested_by_a`；否则整体 `BLOCKED`。Python 非 3.11/3.12、工作区不干净、SoC/CANN/OPP/ATC 任一未知时，不运行探针，先返回环境阻塞摘要。ATC/CANN/OPP 事实通过 mailbox MCP 编译结果采集，不以远端交互 shell 的 `PATH` 判断可用性。

## 单项探针记录模板

```text
probe: <NPURmsNorm|ApplyRotaryPosEmb|FusedInferAttentionScore|NPUAscendQuantV2|AscendDequant|MoeExpert-required|MoeExpert-offset|MatMul-int8>
status: <PASS|BLOCKED>
onnx_opset: 18
onnx_domain: ai.onnx
onnx_op_type: <精确名称>
input_signature: <name:dtype:shape;...>
output_signature: <name:dtype:shape;...>
attributes: <key=value;...>
command: <完整ATC命令，路径可替换为B端固定占位符>
timeout_seconds: 1800
exit_code: <整数|TIMEOUT|NOT_RUN>
om_nonempty: <true|false|NOT_RUN>
expected_ge_op: <精确名称>
observed_ge_op: <精确名称|UNKNOWN>
unsupported_or_fallback_warning: <none|最短关键文本>
onnx_sha256: <SHA-256|NOT_CREATED>
om_sha256: <SHA-256|NOT_CREATED>
root_cause: <PASS时填N/A；否则填写判断>
attempted_remediation: <PASS时填N/A；否则填写>
```

## 结果汇总模板

```text
overall_status: <PASS|BLOCKED>
passed_probes: <数量>/8
blocked_probes: <逗号分隔名称或none>
non_waivable_gate_passed: <true|false>
files_returned_to_a: false
code_or_commits_created_on_b: false
summary_sha256: <本纯文本摘要SHA-256>
```

阶段 0 parser 门禁不可 `WAIVED`。可继续执行其他探针收集事实，但八项全部 `PASS` 前，阶段 0 B 端退出条件和后续发布结论保持 `BLOCKED`。

## 后续 28 项发布矩阵记录要求

阶段 7 使用同一环境头和单项模板，另增加 model、algorithm、target、phase、mask mode、config 指纹、ONNX/OM 文件名与 SHA-256。8 项 FP16 和 20 项 MinMax 必须逐项记录；GPTQ 只记录 FX 数值结果，不生成 ONNX/OM。矩阵项失败只能记为 `BLOCKED`，不能用 `WAIVED` 代替。

发布摘要措辞只能根据证据选择：

- 仅 A 端导出和结构检查完成：“面向 MDC 的 ONNX”。
- 对应项满足 ATC 成功定义：“已通过 ATC 编译”。
- 未做 MDC 真机验证：不得写“可部署到 MDC”“可在 MDC 运行”或同义表述。
