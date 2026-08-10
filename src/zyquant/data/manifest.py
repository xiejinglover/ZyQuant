from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from zyquant.core.versioning import SNAPSHOT_SCHEMA_VERSION


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileManifest(ManifestModel):
    path: str
    table: str
    rows: int
    size: int
    sha256: str


class TableManifest(ManifestModel):
    name: str
    rows: int
    schema_hash: str
    content_hash: str
    files: tuple[str, ...]


class AdjustmentManifest(ManifestModel):
    materialized: bool
    method: str
    algorithm_version: str
    anchor: str
    factor_source: str
    event_adjustments: int


class SnapshotManifest(ManifestModel):
    manifest_schema_version: str = SNAPSHOT_SCHEMA_VERSION
    dataset_id: str
    schema_version: str
    as_of_date: date
    created_at: datetime
    fingerprint: str
    adjustment: AdjustmentManifest
    files: tuple[FileManifest, ...]
    tables: tuple[TableManifest, ...]
    capabilities: Mapping[str, Any] = Field(default_factory=dict)
    lineage: Mapping[str, Any]
    quality: Mapping[str, Any]
