"""Package dependency direction contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2] / "mdc_llm_deploy"
)


def _import_roots(
    tree: ast.AST,
    *,
    parent_level: int = 2,
) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                components = alias.name.split(".")
                if (
                    len(components) >= 2
                    and components[0] == "mdc_llm_deploy"
                ):
                    roots.add(components[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level >= parent_level:
                if node.module:
                    roots.add(
                        node.module.split(".", maxsplit=1)[0]
                    )
                else:
                    roots.update(
                        alias.name for alias in node.names
                    )
            elif node.module == "mdc_llm_deploy":
                roots.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith(
                "mdc_llm_deploy."
            ):
                roots.add(node.module.split(".")[1])
    return roots


def _package_dependency_roots(package: str) -> set[str]:
    roots: set[str] = set()
    for source_path in (_PACKAGE_ROOT / package).rglob("*.py"):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8")
        )
        roots.update(_import_roots(tree))
    return roots


def test_import_roots_covers_relative_and_absolute_forms() -> None:
    tree = ast.parse(
        "\n".join(
            (
                "import mdc_llm_deploy.quantization.math",
                "from mdc_llm_deploy.onnx_export import api",
                "from mdc_llm_deploy import export",
                "from ..mdc_ops import operators",
                "from . import local",
            )
        )
    )

    assert _import_roots(tree) == {
        "export",
        "mdc_ops",
        "onnx_export",
        "quantization",
    }


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [
        (
            "config",
            {
                "export",
                "graph",
                "graph_contract",
                "graph_types",
                "graph_validation",
                "mdc_ops",
                "models",
                "onnx_export",
                "quantization",
            },
        ),
        (
            "models",
            {
                "config",
                "export",
                "graph",
                "graph_contract",
                "graph_types",
                "graph_validation",
                "mdc_ops",
                "onnx_export",
                "quantization",
            },
        ),
        (
            "export",
            {"onnx_export", "quantization"},
        ),
        (
            "onnx_export",
            {"export", "mdc_ops", "models", "quantization"},
        ),
        (
            "quantization",
            {"export", "mdc_ops", "models", "onnx_export"},
        ),
        (
            "mdc_ops",
            {
                "export",
                "graph",
                "graph_contract",
                "graph_types",
                "models",
                "onnx_export",
                "quantization",
            },
        ),
    ],
)
def test_package_has_no_reverse_dependencies(
    package: str,
    forbidden: set[str],
) -> None:
    assert _package_dependency_roots(package).isdisjoint(forbidden)


@pytest.mark.parametrize(
    ("module", "allowed"),
    [
        ("capabilities", {"errors"}),
        ("immutable_json", {"errors"}),
        ("model_properties", set()),
        ("onnx_protocol", set()),
        ("quantization_properties", set()),
        (
            "operator_schema",
            {"attention_layout", "onnx_protocol"},
        ),
        ("graph_types", {"immutable_json"}),
        (
            "graph_validation",
            {"capabilities", "errors", "graph_types"},
        ),
    ],
)
def test_core_module_dependencies_are_layered(
    module: str,
    allowed: set[str],
) -> None:
    source_path = _PACKAGE_ROOT / f"{module}.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    assert _import_roots(tree, parent_level=1) <= allowed


def test_onnx_domain_literal_has_one_production_source() -> None:
    owners = [
        source_path.relative_to(_PACKAGE_ROOT).as_posix()
        for source_path in _PACKAGE_ROOT.rglob("*.py")
        if '"ai.onnx"' in source_path.read_text(encoding="utf-8")
        or "'ai.onnx'" in source_path.read_text(encoding="utf-8")
    ]

    assert owners == ["onnx_protocol.py"]
