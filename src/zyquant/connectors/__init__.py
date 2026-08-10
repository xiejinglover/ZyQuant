"""Optional data-source connectors shipped with the ZyQuant wheel."""

from __future__ import annotations

from importlib import import_module
from typing import Any


BUILTIN_DATA_SOURCES = {
    "canonical-directory": "zyquant.connectors.directory:create_adapter",
    "hermes": "zyquant.connectors.hermes:create_adapter",
    "jqdata": "zyquant.connectors.jqdata:create_adapter",
    "sql": "zyquant.connectors.sql:create_adapter",
}


def builtin_factory(name: str) -> Any:
    reference = BUILTIN_DATA_SOURCES[name]
    module_name, attribute = reference.split(":", 1)
    return getattr(import_module(module_name), attribute)


__all__ = ["BUILTIN_DATA_SOURCES", "builtin_factory"]
