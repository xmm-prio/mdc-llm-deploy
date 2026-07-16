"""Package dependency direction contracts."""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "mdc_llm_deploy"
_TOOLS_ROOT = _REPOSITORY_ROOT / "tools"
_INTERNAL_ROOTS = ("mdc_llm_deploy", "tools")


def _module_name(source_path: Path, source_root: Path, root_module: str) -> str:
    relative = source_path.relative_to(source_root)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((root_module, *parts))


def _module_imports(
    tree: ast.AST,
    *,
    module_name: str,
    is_package: bool = False,
) -> set[str]:
    """Resolve internal imports using the importing source's full module name."""
    imports: set[str] = set()
    package = module_name if is_package else module_name.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                base = resolve_name(relative_name, package)
            else:
                base = node.module or ""
            candidates = (
                base,
                *(f"{base}.{alias.name}" for alias in node.names if alias.name != "*"),
            )
        else:
            continue
        imports.update(
            candidate
            for candidate in candidates
            if candidate in _INTERNAL_ROOTS
            or candidate.startswith(tuple(f"{root}." for root in _INTERNAL_ROOTS))
        )
    return imports


def _source_imports(source_path: Path, source_root: Path, root_module: str) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    return _module_imports(
        tree,
        module_name=_module_name(source_path, source_root, root_module),
        is_package=source_path.name == "__init__.py",
    )


def test_module_imports_resolves_nested_relative_and_absolute_forms() -> None:
    tree = ast.parse(
        "\n".join(
            (
                "import mdc_llm_deploy.quantization.algorithms.math",
                "from mdc_llm_deploy.onnx.transform import api",
                "from mdc_llm_deploy import export",
                "from ...operators.contracts import schema",
                "from .. import metadata",
                "from . import local",
            )
        )
    )

    assert _module_imports(
        tree,
        module_name="mdc_llm_deploy.graph.fx.inspection",
    ) == {
        "mdc_llm_deploy",
        "mdc_llm_deploy.export",
        "mdc_llm_deploy.graph",
        "mdc_llm_deploy.graph.fx",
        "mdc_llm_deploy.graph.fx.local",
        "mdc_llm_deploy.graph.metadata",
        "mdc_llm_deploy.onnx.transform",
        "mdc_llm_deploy.onnx.transform.api",
        "mdc_llm_deploy.operators.contracts",
        "mdc_llm_deploy.operators.contracts.schema",
        "mdc_llm_deploy.quantization.algorithms.math",
    }


def test_module_imports_resolves_package_init_relative_imports() -> None:
    tree = ast.parse("from .runtime import register\nfrom ..contracts import schema")

    assert _module_imports(
        tree,
        module_name="mdc_llm_deploy.operators.torch",
        is_package=True,
    ) == {
        "mdc_llm_deploy.operators.contracts",
        "mdc_llm_deploy.operators.contracts.schema",
        "mdc_llm_deploy.operators.torch.runtime",
        "mdc_llm_deploy.operators.torch.runtime.register",
    }


def _active_domain_prefixes() -> tuple[tuple[str, str], ...]:
    prefixes = [
        ("mdc_llm_deploy.capabilities", "capabilities"),
        ("mdc_llm_deploy.errors", "errors"),
        ("mdc_llm_deploy.export", "export"),
        ("mdc_llm_deploy.models", "models"),
        ("mdc_llm_deploy.quantization.config", "quantization.config"),
        ("mdc_llm_deploy.quantization", "quantization"),
    ]
    optional_packages = {
        "graph": (
            ("mdc_llm_deploy.graph.metadata", "graph.metadata"),
            ("mdc_llm_deploy.graph.fx", "graph.fx"),
            ("mdc_llm_deploy.graph", "graph.core"),
        ),
        "placement": (("mdc_llm_deploy.placement", "placement"),),
        "operators": (
            ("mdc_llm_deploy.operators.contracts", "operators.contracts"),
            ("mdc_llm_deploy.operators.runtime", "operators.runtime"),
            ("mdc_llm_deploy.operators.onnx", "operators.onnx"),
            ("mdc_llm_deploy.operators.torch", "operators.torch"),
            ("mdc_llm_deploy.operators", "operators.torch"),
        ),
        "onnx": (("mdc_llm_deploy.onnx", "onnx"),),
    }
    for package, package_prefixes in optional_packages.items():
        if (_PACKAGE_ROOT / package).is_dir():
            prefixes.extend(package_prefixes)
    if (_TOOLS_ROOT / "release").is_dir():
        prefixes.append(("tools.release", "release"))
    return tuple(sorted(prefixes, key=lambda item: len(item[0]), reverse=True))


def _domain_for_module(module_name: str) -> str | None:
    for prefix, domain in _active_domain_prefixes():
        if module_name == prefix or module_name.startswith(f"{prefix}."):
            return domain
    return None


_ALLOWED_DOMAIN_DEPENDENCIES = {
    "errors": set(),
    "capabilities": {"errors"},
    "graph.metadata": {"capabilities", "errors"},
    "graph.fx": {"graph.metadata"},
    "graph.core": {"capabilities", "errors", "graph.fx", "graph.metadata"},
    "placement": {"errors"},
    "operators.contracts": {"errors"},
    "operators.runtime": {"errors", "operators.contracts"},
    "operators.torch": {
        "errors",
        "operators.contracts",
        "operators.onnx",
        "operators.runtime",
    },
    "operators.onnx": {"errors", "operators.contracts"},
    "models": {"errors", "operators.runtime", "operators.torch"},
    "export": {
        "errors",
        "graph.core",
        "graph.fx",
        "graph.metadata",
        "operators.contracts",
        "placement",
    },
    "quantization.config": {"capabilities", "errors"},
    "quantization": {
        "capabilities",
        "errors",
        "graph.core",
        "graph.fx",
        "graph.metadata",
        "operators.runtime",
        "operators.torch",
        "placement",
        "quantization.config",
    },
    "onnx": {
        "capabilities",
        "errors",
        "graph.core",
        "graph.fx",
        "graph.metadata",
        "operators.contracts",
        "operators.onnx",
        "placement",
    },
    "release": {
        "capabilities",
        "errors",
        "export",
        "graph.metadata",
        "models",
        "onnx",
        "quantization",
    },
}


def _architecture_sources() -> list[tuple[Path, Path, str]]:
    sources = [(path, _PACKAGE_ROOT, "mdc_llm_deploy") for path in _PACKAGE_ROOT.rglob("*.py")]
    release_root = _TOOLS_ROOT / "release"
    if release_root.is_dir():
        sources.extend((path, _TOOLS_ROOT, "tools") for path in release_root.rglob("*.py"))
    return sources


def test_migrated_domains_follow_allowed_dependency_edges() -> None:
    violations: list[str] = []
    for source_path, source_root, root_module in _architecture_sources():
        source_module = _module_name(source_path, source_root, root_module)
        source_domain = _domain_for_module(source_module)
        if source_domain is None:
            continue
        allowed = _ALLOWED_DOMAIN_DEPENDENCIES[source_domain] | {source_domain}
        imported_domains = {
            domain
            for imported in _source_imports(source_path, source_root, root_module)
            if (domain := _domain_for_module(imported)) is not None
        }
        forbidden = imported_domains - allowed
        if forbidden:
            relative = source_path.relative_to(_REPOSITORY_ROOT).as_posix()
            violations.append(f"{relative}: {sorted(forbidden)}")

    assert not violations, "Forbidden domain dependencies:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    ("module", "allowed"),
    [
        ("capabilities", {"errors"}),
        ("graph/metadata/json", {"errors"}),
        ("graph/metadata/model", set()),
        ("operators/contracts/onnx", set()),
        ("graph/metadata/quantization", set()),
        (
            "operators/contracts/schema",
            {"operators"},
        ),
        ("graph/metadata/types", {"graph"}),
        (
            "graph/validation",
            {"capabilities", "errors", "graph"},
        ),
    ],
)
def test_core_module_dependencies_are_layered(
    module: str,
    allowed: set[str],
) -> None:
    source_path = _PACKAGE_ROOT / f"{module}.py"
    imported_roots = {
        imported.split(".")[1]
        for imported in _source_imports(source_path, _PACKAGE_ROOT, "mdc_llm_deploy")
        if imported.startswith("mdc_llm_deploy.")
    }

    assert imported_roots <= allowed


def _assignment_literal(tree: ast.Module, name: str) -> object:
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"Missing literal assignment: {name}")


def test_root_lazy_exports_target_public_domain_entries() -> None:
    tree = ast.parse((_PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8"))
    lazy_exports = _assignment_literal(tree, "_LAZY_EXPORTS")
    assert isinstance(lazy_exports, dict)
    expected_exports = {
        "AutoExportModel": ("mdc_llm_deploy.models", "AutoExportModel"),
        "ExportModelConfig": ("mdc_llm_deploy.models", "ExportModelConfig"),
        "Qwen3Config": ("mdc_llm_deploy.models", "Qwen3Config"),
        "Qwen3ForCausalLM": ("mdc_llm_deploy.models", "Qwen3ForCausalLM"),
        "Qwen3MoeConfig": ("mdc_llm_deploy.models", "Qwen3MoeConfig"),
        "Qwen3MoeForCausalLM": ("mdc_llm_deploy.models", "Qwen3MoeForCausalLM"),
        "convert_to_decode": ("mdc_llm_deploy.export", "convert_to_decode"),
        "export": ("mdc_llm_deploy.export", "export"),
        "oneshot": ("mdc_llm_deploy.quantization", "oneshot"),
        "onnx_export": ("mdc_llm_deploy.onnx.api", "onnx_export"),
    }

    assert lazy_exports == expected_exports


def test_release_tools_use_domain_package() -> None:
    release_package = _TOOLS_ROOT / "release"
    assert release_package.is_dir()
    assert not tuple(_TOOLS_ROOT.glob("release_*.py"))
    assert {path.name for path in release_package.glob("*.py")} == {
        "__init__.py",
        "matrix.py",
        "validation.py",
    }


def test_generic_module_directories_are_prohibited() -> None:
    generic_directories = {
        path.relative_to(_PACKAGE_ROOT).as_posix()
        for path in _PACKAGE_ROOT.rglob("*")
        if path.is_dir() and path.name in {"common", "misc"}
    }
    assert not generic_directories

    utils_root = _PACKAGE_ROOT / "utils"
    if (_PACKAGE_ROOT / "placement").is_dir():
        assert not utils_root.exists()
    elif utils_root.is_dir():
        assert {
            path.relative_to(_PACKAGE_ROOT).as_posix() for path in utils_root.rglob("*.py")
        } == {"utils/__init__.py", "utils/device.py"}


def test_onnx_domain_literal_has_one_production_source() -> None:
    owners = [
        source_path.relative_to(_PACKAGE_ROOT).as_posix()
        for source_path in _PACKAGE_ROOT.rglob("*.py")
        if '"ai.onnx"' in source_path.read_text(encoding="utf-8")
        or "'ai.onnx'" in source_path.read_text(encoding="utf-8")
    ]

    assert len(owners) == 1
    if (_PACKAGE_ROOT / "operators").is_dir():
        assert owners[0].startswith("operators/contracts/")
    else:
        assert owners == ["onnx_protocol.py"]
