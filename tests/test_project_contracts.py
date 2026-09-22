from __future__ import annotations

import re
import tomllib
from pathlib import Path

import zyquant
from zyquant.core.versioning import FRAMEWORK_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_package_and_framework_versions_are_aligned():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == zyquant.__version__ == FRAMEWORK_VERSION


def test_current_documentation_has_no_broken_local_links():
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    missing = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1).split("#", 1)[0].strip().split(" ", 1)[0]
            if not target or "://" in target or target.startswith(("/", "mailto:")):
                continue
            if not (document.parent / target).resolve().exists():
                line = text.count("\n", 0, match.start()) + 1
                missing.append(f"{document.relative_to(ROOT)}:{line}: {target}")
    assert not missing, "broken documentation links:\n" + "\n".join(missing)
