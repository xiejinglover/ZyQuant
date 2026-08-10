from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .exceptions import PluginError


@contextmanager
def _project_import_path(project_root: str | Path | None) -> Iterator[None]:
    if project_root is None:
        yield
        return
    root = str(Path(project_root).expanduser().resolve())
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if inserted and root in sys.path:
            sys.path.remove(root)


def load_object(
    reference: str,
    project_root: str | Path | None = None,
) -> Any:
    """Load ``module:attribute`` from an explicit project root."""
    module_name, separator, attribute_path = reference.partition(":")
    if not separator or not module_name or not attribute_path:
        raise PluginError(
            f"object reference must use 'module:attribute': {reference}"
        )
    try:
        with _project_import_path(project_root):
            value: Any = importlib.import_module(module_name)
            for attribute in attribute_path.split("."):
                value = getattr(value, attribute)
            return value
    except (ImportError, AttributeError) as exc:
        raise PluginError(f"cannot load object reference: {reference}") from exc
