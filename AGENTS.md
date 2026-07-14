# MDC_LLM_DEPLOY

A Transformers-compatible library for applying compression algorithms to LLMs for optimized deployment with MDC.

## Cursor Cloud specific instructions

- This repository is currently a **skeleton**: it contains only `README.md`, `LICENSE`, and `.gitignore`. There is no source code, no dependency manifest (`requirements.txt` / `pyproject.toml` / `setup.py`), no tests, and no runnable application yet. There is nothing to build, lint, test, or run until code is added.
- Runtime is **Python 3.12** with system `pip` (invoke as `python3` / `pip`; there is no bare `python` alias). `pip` has network access.
- `python3 -m venv` **is** available if you want an isolated environment; otherwise system or `pip install --user` installs work fine.
- The startup update script auto-installs dependencies **only if** a manifest exists: `requirements.txt`, `requirements-dev.txt`, or an installable `pyproject.toml`/`setup.py` (editable install). When you add the first dependency manifest it is picked up automatically on the next session — no update-script change is required for those common cases.
- Once real code lands, add the standard lint/test/build/run commands here. The `.gitignore` follows the standard Python template (references `ruff`, `pytest`, `mypy`), so those are the expected tools; install them when a manifest declares them.
