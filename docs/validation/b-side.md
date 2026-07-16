# B 端验证记录

状态：`BLOCKED`

本记录严格区分“已取得的历史事实”和“尚待执行的候选提交验证计划”。当前没有足够证据将任何发布门禁标记为 `PASS`，也不得用 `WAIVED` 替代。

## 2026-07-15 ATC 探针事实

- 历史最小探针曾确认：`MoeExpert` 的 `hiddenSize=64/128` 会发生 tiling 失败。
- 历史最小探针曾确认：`MoeExpert` 在 `hiddenSize=256`、`expertInterDim=128/256`、`tokenNum=8/3072` 的部分组合上可完成 ATC 编译。
- 历史探针来自修复过程中的临时模型，不等于候选 commit 通过；候选 commit 的 parser 探针、GPU 数值验证和 28 项发布矩阵均须重新执行。
- `FusedInferAttentionScore` 历史探针曾出现 `output index 1 shape:[0] not valid` 告警。Attention LSE 告警未定性：尚未确认其来自 ONNX parser、GE shape inference 还是 OPP 支持检查，也未取得 OPP 维护方的无害性结论。
- 没有执行 MDC 真机推理。因此不得声明模型“可部署到 MDC”“可在 MDC 运行”或“真机已验证”。
- B 端没有向 A 端返回 ONNX、OM 或原始日志，也没有在 B 端修改代码。

```yaml
artifact_returned_to_a: false
code_changed_on_b: false
timeout_seconds: 1800
```

### 七个算子/节点的历史证据边界

以下名称是后续独立 parser/ATC 探针必须覆盖的对象，不表示候选 commit 已通过：

- `NPURmsNorm`：候选 commit 未验证，状态为 `BLOCKED`。
- `ApplyRotaryPosEmb`：候选 commit 未验证，状态为 `BLOCKED`。
- `FusedInferAttentionScore`：存在未定性的 Attention LSE 告警，状态为 `BLOCKED`。
- `NPUAscendQuantV2`：候选 commit 未验证，状态为 `BLOCKED`。
- `AscendDequant`：候选 commit 未验证，状态为 `BLOCKED`。
- `MoeExpert`：只有历史最小探针事实，候选 commit 未验证，状态为 `BLOCKED`。
- `MatMul`：INT8×INT8、INT32 累加扩展语义的候选 commit 探针未验证，状态为 `BLOCKED`。

## 候选 commit 验证计划

以下内容均是计划，不是已完成事实：

1. A 端提供已推送的候选 commit SHA；B 端拉取该 SHA，确认 `HEAD` 一致、验证前工作区干净，并记录 Python、锁文件 hash、GPU、驱动、PyTorch、Triton、ATC、CANN、OPP、目标 SoC 和 precision mode。
2. B 端执行完整测试、Ruff、Mypy 和构建，并记录命令、退出码与关键错误。
3. B 端对六个自定义算子执行 GPU 实现与独立 CPU reference 的数值对照。
4. B 端为上述七个算子/节点执行八项独立 parser/ATC 探针，其中 `FusedInferAttentionScore` 分浮点和全 per-tensor INT8 两项。
5. 对 Attention LSE 告警分别验证 `softmax_lse_flag=false`、`softmax_lse_flag=true` 和省略可选第二输出，并取得 OPP 维护方结论；结论取得前保持 `BLOCKED`。
6. 使用候选 commit 重新生成并验证 8 项 FP16 与 20 项 MinMax 发布矩阵，不复用历史 ONNX；每条 ATC 命令超时为 1800 秒。
7. B 端只返回纯文本摘要；ONNX、OM 和原始日志留在 B 端并按保留策略处理。

## 解除阻塞条件

仅当候选 commit 的基础门禁、六算子 GPU 数值对照、八项 parser/ATC 探针和 28 项发布矩阵全部满足 PRD 的通过定义，且 Attention LSE 告警已修复或取得正式无害性结论时，状态才可改为 `PASS`。在此之前保持 `BLOCKED`。
