# MDC_LLM_DEPLOY

A Transformers-compatible library for applying compression algorithms to LLMs for optimized deployment with MDC.

# GOAL

- Qwen3-4B和Qwen3-30B的浮点和量化模型的ONNX导出
- Qwen3-4B和Qwen3-30B的量化
- 使用单层模型和小词表快速端到端验证

# Python 环境

- Windows 下执行 Python、pytest、mypy、ruff 等命令时，统一使用 `.venv\Scripts\python.exe -m <module>`。
- 不使用 `py` 或系统 `python`，避免绕过项目虚拟环境。

