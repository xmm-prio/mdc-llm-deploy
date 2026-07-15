# 测试目录

- `unit/`：隔离验证配置、模型、图契约和算子行为。
- `integration/`：验证导出、量化、ONNX 和发布矩阵流程。
- `contracts/`：冻结文档与发布能力矩阵声明。
- `conftest.py`：提供每个测试独立创建的输入和图工厂。

常用命令：

```powershell
python -m pytest tests/unit
python -m pytest tests/integration
python -m pytest tests/contracts
python -m pytest -m "not slow"
```
