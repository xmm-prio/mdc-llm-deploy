from __future__ import annotations

import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+\.md)\)")
EXPECTED_DOCUMENTS = {
    "docs/onnx.md",
    "docs/operators/ApplyRotaryPosEmb.md",
    "docs/operators/AscendDequant.md",
    "docs/operators/AscendQuantV2.md",
    "docs/operators/FusedInferAttentionScore.md",
    "docs/operators/RmsNorm.md",
    "docs/quantization.md",
}


def test_project_readme_is_nonempty_and_links_published_documentation() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)["project"]

    assert project["readme"] == "README.md"
    readme_path = PROJECT_ROOT / project["readme"]
    readme = readme_path.read_text(encoding="utf-8")
    linked_documents = set(DOCUMENT_LINK_PATTERN.findall(readme))

    assert readme.strip()
    assert linked_documents == EXPECTED_DOCUMENTS
    assert all((PROJECT_ROOT / link).is_file() for link in linked_documents)
