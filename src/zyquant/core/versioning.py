from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .hashing import hash_file, hash_payload

FRAMEWORK_VERSION = "2.0.0"
SNAPSHOT_SCHEMA_VERSION = "1.0"
LEDGER_SCHEMA_VERSION = "1.0"
RUN_SCHEMA_VERSION = "1.0"
PLUGIN_PROTOCOL_VERSION = "1.0"


def derive_seed(root_seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        f"{root_seed}:{hash_payload(parts)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def source_tree_fingerprint(
    root: str | Path, patterns: Iterable[str] = ("*.py",)
) -> str:
    base = Path(root).expanduser().resolve()
    files: list[tuple[str, str]] = []
    for pattern in patterns:
        for path in sorted(base.rglob(pattern)):
            if any(part in {".git", "build", "__pycache__", ".mypy_cache"} for part in path.parts):
                continue
            files.append((path.relative_to(base).as_posix(), hash_file(path)))
    return hash_payload(files)


def git_metadata(root: str | Path) -> dict[str, object]:
    base = Path(root).expanduser().resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=base, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=base, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"available": True, "commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        source_root = base / "src" if (base / "src").exists() else base
        return {
            "available": False,
            "source_tree_fingerprint": source_tree_fingerprint(source_root),
        }


def environment_metadata() -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "pyarrow", "pydantic", "PyYAML", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "pid": os.getpid(),
    }


@dataclass(frozen=True)
class SchemaRef:
    name: str
    version: str
