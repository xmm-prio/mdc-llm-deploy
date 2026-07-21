# MDC LLM Deploy

MDC LLM Deploy 是面向 MDC 部署的 Transformers-compatible LLM 压缩与 ONNX
导出库。项目支持 Python `>=3.11,<3.13`。

## 安装

从仓库根目录通过标准 Python 构建接口安装：

```powershell
.venv\Scripts\python.exe -m pip install .
```

## MDC ONNX 图处理

`process_onnx` 是 MDC ONNX 图处理的公开总入口：

```python
import onnx

from mdc_llm_deploy.onnx import process_onnx

model = onnx.load("model.onnx")
processed = process_onnx(model)
assert processed is model
```

该流程原地修改并返回同一模型。任一步失败时抛出异常，输入模型保持不变。支持范围、
处理顺序、schema 生命周期及硬件准确度验证边界见
[MDC ONNX 图处理](docs/onnx.md)。

需要单独观察融合结果时，可直接运行融合编排器：

```python
from mdc_llm_deploy.onnx import run_fusion_passes

report = run_fusion_passes(model)
print(report.counts)
```

独立编排器固定按 RMSNorm、RoPE、FIA 顺序原地执行。后续 pass 失败时，前序成功结果
保留；需要全流程失败回滚时应使用 `process_onnx`。

## ONNX 算子

标准 ONNX 导出图经 `process_onnx` 完成 lowering、融合、schema 注册和校验。
当前算子 ABI 与融合限制由对应文档维护：

- [ApplyRotaryPosEmb](docs/operators/ApplyRotaryPosEmb.md)
- [AscendDequant](docs/operators/AscendDequant.md)
- [AscendQuantV2](docs/operators/AscendQuantV2.md)
- [FusedInferAttentionScore](docs/operators/FusedInferAttentionScore.md)
- [RmsNorm](docs/operators/RmsNorm.md)

## 验证

仓库内 Windows 验证统一使用项目虚拟环境：

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy mdc_llm_deploy
```
