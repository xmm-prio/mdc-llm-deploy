# MDC_LLM_DEPLOY

A Transformers-compatible library for applying compression algorithms to LLMs for optimized deployment with MDC.

## Cursor Cloud specific instructions

- This repository is currently a **skeleton**: it contains only `README.md`, `LICENSE`, and `.gitignore`. There is no source code, no dependency manifest (`requirements.txt` / `pyproject.toml` / `setup.py`), no tests, and no runnable application yet. There is nothing to build or run until code is added.
- Runtime is **Python 3.12** with system `pip` (invoke as `python3` / `pip`; there is no bare `python` alias). `pip` has network access.
- `venv` is **not** available out of the box (the `python3.12-venv` system package is missing). Use system or `pip install --user` installs, or install `python3.12-venv` via apt if isolation is needed.
- The startup update script auto-installs dependencies **only if** a manifest exists: `requirements.txt`, `requirements-dev.txt`, or an installable `pyproject.toml`/`setup.py` (editable install). When you add the first dependency manifest, it will be picked up automatically on the next session — no update-script change is required for the common cases above.
- Once real code lands, add the standard lint/test/build/run commands here (this project follows Python conventions per `.gitignore`, e.g. `ruff`/`pytest`).
