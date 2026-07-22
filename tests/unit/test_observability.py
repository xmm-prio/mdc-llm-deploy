"""Tests for shared logging configuration."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path


def run_python(script: str) -> dict[str, object]:
    """Run a script in a clean interpreter and return its JSON result."""
    project_root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_package_import_configures_one_colored_info_handler() -> None:
    result = run_python(
        """
import json
import importlib
import logging
import mdc_llm_deploy
from rich.logging import RichHandler

root_logger = logging.getLogger()
configured_handlers = root_logger.handlers[:]
importlib.reload(mdc_llm_deploy)
print(json.dumps({
    "level": root_logger.level,
    "handlers": [isinstance(handler, RichHandler) for handler in root_logger.handlers],
    "idempotent": root_logger.handlers == configured_handlers,
}))
"""
    )

    assert result == {
        "level": logging.INFO,
        "handlers": [True],
        "idempotent": True,
    }


def test_package_import_preserves_existing_application_configuration() -> None:
    result = run_python(
        """
import json
import logging

root_logger = logging.getLogger()
application_handler = logging.NullHandler()
root_logger.addHandler(application_handler)
root_logger.setLevel(logging.WARNING)

import mdc_llm_deploy

print(json.dumps({
    "level": root_logger.level,
    "handler_count": len(root_logger.handlers),
    "preserved": root_logger.handlers[0] is application_handler,
}))
"""
    )

    assert result == {
        "level": logging.WARNING,
        "handler_count": 1,
        "preserved": True,
    }
