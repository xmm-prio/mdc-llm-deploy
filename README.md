# MDC LLM Deploy

`mdc_llm_deploy` 是面向华为 MDC 的 LLM 压缩与 ONNX 转换库。当前版本提供确定性 Tiny Qwen3 Dense/MoE、ATen FX 导出、MinMax/GPTQ PTQ、decode/KV Cache 改写、六个 MDC 算子胶囊及 MDC ONNX 方言下沉。

## 环境

- Python 3.12
- PyTorch 2.6 至 3.0
- ONNX 1.17 至 2.0

## 安装

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 配置

`configs/` 中五份 JSON 是发布验收配置。配置解析器严格拒绝未知字段和错误类型，并将已解析默认值纳入规范 JSON 与 SHA-256 指纹。

```python
from mdc_llm_deploy import QuantizationConfig

config = QuantizationConfig.load("configs/minmax-linear-w8a8.json")
print(config.fingerprint)
print(config.to_json_string())
```

配置模块保持纯 Python，不导入 PyTorch。发布 Schema 位于 `mdc_llm_deploy/config/schema.json`。

## 确定性基线

Tiny Dense、Tiny MoE 和 3072-token 输入均使用 seed `20260714`。fixture 使用 NumPy PCG64 在 `[0, 128)` 采样，并按 little-endian int64、C 连续顺序编码。

```python
from mdc_llm_deploy.models import TinyQwen3Dense
from mdc_llm_deploy.utils import fixture_sha256, release_input_ids

model = TinyQwen3Dense()
input_ids = release_input_ids()
print(input_ids.shape, fixture_sha256())
```

Tiny 模型默认使用 FP16 参数并处于 `eval()` 状态。固定 prefill ABI：

- 输入 `input_ids`：int64，shape `[1, 3072]`
- 输出 `logits`：FP16，shape `[1, 3072, 128]`
- 输出 key/value cache：FP16，shape `[1, 2, 3072, 16]`

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy mdc_llm_deploy
.\.venv\Scripts\python.exe -m build
```

测试套件会生成并结构验证 FP16/MinMax 的 28 项本地 ONNX 矩阵，临时产物由 pytest 清理。GPTQ 仅验收 FX 数值路径，并明确拒绝 ONNX 导出。

B 端 parser、GPU 和 ATC 门禁见 `docs/validation/b-side.md`。A 端提交并推送前，这些门禁保持 `BLOCKED`。生成文件只能描述为“面向 MDC 的 ONNX”；对应项通过 B 端 ATC 编译后，才可描述为“已通过 ATC 编译”。MDC 真机运行不在当前版本承诺范围内。
