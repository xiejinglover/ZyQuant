from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from zyquant.core.exceptions import DataContractError
from zyquant.core.plugins import PluginMetadata
from zyquant.data.adapters import DirectoryDataAdapter


def create_adapter(
    request: Mapping[str, Any] | None = None,
) -> DirectoryDataAdapter:
    payload = dict(request or {})
    raw = payload.get("path") or payload.get("input")
    if not raw:
        raise DataContractError(
            "canonical-directory request requires 'path'"
        )
    return DirectoryDataAdapter(Path(str(raw)))


setattr(create_adapter, "plugin_metadata", PluginMetadata(
    name="canonical-directory",
    version="2.0.0",
    kind="data",
    input_schema="data.source/canonical-directory@1",
    output_schema="data.canonical_batch@1",
))
