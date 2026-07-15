# MDC_LLM_DEPLOY

A Transformers-compatible library for applying compression algorithms to LLMs for optimized deployment with MDC.

## Cursor Cloud specific instructions

- 工程使用 **Python 3.12**；Cloud 中通过 `python3` 调用解释器。
- 依赖声明位于 `pyproject.toml`，带哈希锁位于 `requirements.lock`。优先创建 venv 后安装，避免污染系统环境。
- 源码位于 `mdc_llm_deploy/`，测试位于 `tests/`；公开接口只有 Python API，`tools/` 仅供内部验收。
- 标准门禁：`python3 -m pytest`、`python3 -m ruff check .`、`python3 -m mypy mdc_llm_deploy`、`python3 -m build`。
- Python 3.12 依赖可用 `python3 -m pip install --require-hashes -r requirements.lock` 复现；更新依赖后运行 `python3 tools/generate_lock.py` 重建锁文件。
- `tools/release_matrix.py` 是内部验收 runner，不属于公开产品 CLI。
