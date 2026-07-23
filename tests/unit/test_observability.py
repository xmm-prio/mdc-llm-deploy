"""Tests for shared logging configuration."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path


def run_python(script: str) -> tuple[dict[str, object], str]:
    """Run a script in a clean interpreter and return its JSON result and log output."""
    project_root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout_lines = result.stdout.splitlines()
    payload = json.loads(stdout_lines[-1])
    log_output = "\n".join(stdout_lines[:-1]) + result.stderr
    return payload, log_output


def test_package_import_configures_isolated_colored_info_handler() -> None:
    result, log_output = run_python(
        """
import json
import importlib
import logging
import mdc_llm_deploy
from rich.logging import RichHandler

root_logger = logging.getLogger()
package_logger = logging.getLogger("mdc_llm_deploy")
configured_handlers = package_logger.handlers[:]
importlib.reload(mdc_llm_deploy)
logging.getLogger("mdc_llm_deploy.test").info("package info marker")
logging.getLogger("torch.onnx").info("dependency info marker")
print(json.dumps({
    "root_level": root_logger.level,
    "root_handlers": len(root_logger.handlers),
    "package_level": package_logger.level,
    "package_handlers": [
        isinstance(handler, RichHandler) for handler in package_logger.handlers
    ],
    "package_propagates": package_logger.propagate,
    "idempotent": package_logger.handlers == configured_handlers,
}))
"""
    )

    assert result == {
        "root_level": logging.WARNING,
        "root_handlers": 0,
        "package_level": logging.INFO,
        "package_handlers": [True],
        "package_propagates": False,
        "idempotent": True,
    }
    assert "package info marker" in log_output
    assert "dependency info marker" not in log_output


def test_package_import_preserves_existing_application_configuration() -> None:
    result, log_output = run_python(
        """
import json
import logging

root_logger = logging.getLogger()
application_handler = logging.NullHandler()
root_logger.addHandler(application_handler)
root_logger.setLevel(logging.WARNING)

import mdc_llm_deploy

package_logger = logging.getLogger("mdc_llm_deploy")
print(json.dumps({
    "root_level": root_logger.level,
    "root_handler_count": len(root_logger.handlers),
    "root_preserved": root_logger.handlers[0] is application_handler,
    "package_level": package_logger.level,
    "package_handler_count": len(package_logger.handlers),
    "package_propagates": package_logger.propagate,
}))
"""
    )

    assert result == {
        "root_level": logging.WARNING,
        "root_handler_count": 1,
        "root_preserved": True,
        "package_level": logging.NOTSET,
        "package_handler_count": 0,
        "package_propagates": True,
    }
    assert log_output == ""
