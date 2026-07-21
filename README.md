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

from mdc_llm_deploy.mdc_onnx import process_onnx

model = onnx.load("model.onnx")
processed = process_onnx(model)
assert processed is model
```

该流程原地修改并返回同一模型。任一步失败时抛出异常，输入模型保持不变。支持范围、
处理顺序、schema 生命周期及硬件准确度验证边界见
[MDC ONNX 图处理](docs/mdc_onnx.md)。

## Custom operators

各算子的 Torch 契约、ONNX ABI 与导出限制由对应文档维护：

- [ApplyRotaryPosEmb](docs/operators/ApplyRotaryPosEmb.md)
- [AscendDequant](docs/operators/AscendDequant.md)
- [AscendQuantV2](docs/operators/AscendQuantV2.md)
- [FusedInferAttentionScore](docs/operators/FusedInferAttentionScore.md)
- [MoeExpert](docs/operators/MoeExpert.md)
- [RmsNorm](docs/operators/RmsNorm.md)

创建导出 profile 时，选中算子的 schema 会先完成整批 ABI 预检，再开始写入当前进程的
ONNX registry。预检冲突不会由本次调用新增 schema。ONNX registry 不提供多 schema
事务：实际写入中途失败时，已经成功写入的兼容前缀会保留，不做回滚。项目锁只串行化
经本库进入的注册；第三方直接修改 registry 仍可能产生竞态，需要强隔离时应使用独立
进程。

## 验证

仓库内 Windows 验证统一使用项目虚拟环境：

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy mdc_llm_deploy
```
