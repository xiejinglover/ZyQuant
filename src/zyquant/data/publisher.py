from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from zyquant.core.exceptions import DataContractError
from zyquant.core.hashing import canonical_json, hash_file, hash_payload
from zyquant.core.versioning import SNAPSHOT_SCHEMA_VERSION

from .adjustment import AdjustmentProcessor, VENDOR_FACTOR_RTOL
from .contracts import (
    BASE_TABLES, FIELD_SPECS, FINANCIAL_TABLES,
)
from .normalization import normalize_table
from .snapshot import DataSnapshot
from .validation import SnapshotValidator


class SnapshotPublisher:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.datasets_root = self.root / "datasets"
        self.datasets_root.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        dataset_id: str,
        tables: Mapping[str, pd.DataFrame],
        vendor_factors: pd.DataFrame | None = None,
        schema_version: str = "1.0",
        as_of_date: date | None = None,
        lineage: Mapping[str, object] | None = None,
        vendor_factor_mode: str = "use",
        vendor_factor_rtol: float = VENDOR_FACTOR_RTOL,
    ) -> DataSnapshot:
        final = self.datasets_root / dataset_id
        if final.exists():
            raise DataContractError(f"immutable dataset already exists: {dataset_id}")
        unknown_tables = set(tables) - set(FIELD_SPECS)
        if unknown_tables:
            raise DataContractError(
                f"publishing unsupported canonical tables: {sorted(unknown_tables)}"
            )
        source_tables = set(BASE_TABLES) - {"daily_post_adjusted"}
        missing_source = source_tables - set(tables)
        if missing_source:
            raise DataContractError(
                f"publishing missing source tables: {sorted(missing_source)}"
            )
        working = {
            name: normalize_table(name, frame)
            for name, frame in tables.items()
            if name != "daily_post_adjusted"
        }
        financial_present = set(FINANCIAL_TABLES) & set(working)
        if financial_present and financial_present != set(FINANCIAL_TABLES):
            raise DataContractError(
                "financial capability requires all optional financial tables; "
                f"present={sorted(financial_present)}"
            )
        if financial_present and schema_version == "1.0":
            schema_version = "1.1"
        adjusted = AdjustmentProcessor().build(
            working["daily_raw"],
            working["corporate_actions"],
            vendor_factors,
            vendor_factor_mode,
            vendor_factor_rtol,
        )
        working["daily_post_adjusted"] = adjusted.daily_post_adjusted
        normalized = SnapshotValidator().validate(working)
        effective_lineage = dict(lineage or {})
        if "daily_money_flow" in normalized:
            flow = normalized["daily_money_flow"]
            declared_capabilities = effective_lineage.get("capabilities", {})
            flow_capabilities = (
                dict(declared_capabilities)
                if isinstance(declared_capabilities, Mapping)
                else {}
            )
            flow_capabilities.setdefault("daily_money_flow", {
                "schema_version": "1",
                "fields": sorted(flow.columns),
                "start_date": (
                    str(flow["trade_date"].min()) if not flow.empty else None
                ),
                "end_date": (
                    str(flow["trade_date"].max()) if not flow.empty else None
                ),
            })
            effective_lineage["capabilities"] = flow_capabilities
            effective_lineage.setdefault("daily_money_flow", {
                "schema_version": "1",
                "unit": "CNY",
                "visibility_field": "available_at",
            })
        staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.", dir=self.datasets_root))
        try:
            for name, frame in normalized.items():
                self._write_table(staging / name, frame)
            files = []
            for path in sorted(staging.rglob("*.parquet")):
                table_name = path.relative_to(staging).parts[0]
                files.append({
                    "path": path.relative_to(staging).as_posix(),
                    "table": table_name,
                    "rows": int(pd.read_parquet(path).shape[0]),
                    "size": path.stat().st_size,
                    "sha256": hash_file(path),
                })
            table_manifests = []
            for name, frame in sorted(normalized.items()):
                relative_files = tuple(
                    item["path"] for item in files if item["table"] == name
                )
                table_manifests.append({
                    "name": name,
                    "rows": len(frame),
                    "schema_hash": hash_payload([
                        {
                            "name": str(column),
                            "type": str(FIELD_SPECS[name][column].arrow_type),
                            "nullable": FIELD_SPECS[name][column].nullable,
                            "unit": FIELD_SPECS[name][column].unit,
                            "enum": sorted(
                                FIELD_SPECS[name][column].enum or ()
                            ) or None,
                        }
                        for column in frame.columns
                    ]),
                    "content_hash": hash_payload(
                        pd.util.hash_pandas_object(
                            frame, index=False, categorize=True
                        ).astype(str).tolist()
                    ),
                    "files": relative_files,
                })
            fingerprint_payload = {
                "schema_version": schema_version,
                "tables": table_manifests,
                "adjustment_version": adjusted.algorithm_version,
                "lineage": effective_lineage,
            }
            fingerprint = hashlib.sha256(
                canonical_json(fingerprint_payload).encode("utf-8")
            ).hexdigest()
            calendar = normalized["trade_calendar"]
            maximum = max(calendar["trade_date"]) if len(calendar) else None
            if maximum is None:
                raise DataContractError("trade calendar is empty")
            effective_as_of = as_of_date or maximum
            if effective_as_of < maximum:
                raise DataContractError("as_of_date cannot precede the latest calendar date")
            capabilities: dict[str, object] = {}
            if isinstance(effective_lineage, Mapping):
                declared_capabilities = effective_lineage.get("capabilities", {})
                if isinstance(declared_capabilities, Mapping):
                    capabilities = dict(declared_capabilities)
            manifest = {
                "manifest_schema_version": SNAPSHOT_SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "schema_version": schema_version,
                "as_of_date": str(effective_as_of),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "fingerprint": fingerprint,
                "adjustment": {
                    "materialized": True,
                    "method": "post_adjusted_total_return",
                    "algorithm_version": adjusted.algorithm_version,
                    "anchor": "instrument_first_valid_bar",
                    "factor_source": adjusted.diagnostics.factor_source,
                    "event_adjustments": adjusted.diagnostics.event_adjustments,
                },
                "files": files,
                "tables": table_manifests,
                "capabilities": capabilities,
                "lineage": effective_lineage,
                "quality": {
                    "status": "passed",
                    "tables": {
                        name: {"rows": len(frame), "columns": len(frame.columns)}
                        for name, frame in sorted(normalized.items())
                    },
                    "raw_adjusted_key_match": True,
                    "adjustment_rows": adjusted.diagnostics.rows,
                    "vendor_factors": dict(
                        adjusted.diagnostics.vendor_factors
                    ),
                },
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(staging, final)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return DataSnapshot(final)

    def publish_adapter(
        self,
        dataset_id: str,
        adapter,
        request=None,
        schema_version: str = "1.0",
        as_of_date: date | None = None,
    ) -> DataSnapshot:
        batch = adapter.ingest(request)
        return self.publish(
            dataset_id, batch.tables, batch.vendor_factors,
            schema_version=schema_version, as_of_date=as_of_date,
            lineage=batch.source_metadata,
            vendor_factor_mode=batch.vendor_factor_mode,
            vendor_factor_rtol=batch.vendor_factor_rtol,
        )

    @staticmethod
    def _write_table(path: Path, frame: pd.DataFrame) -> None:
        path.mkdir(parents=True, exist_ok=True)
        date_column = "trade_date" if "trade_date" in frame else None
        if date_column and not frame.empty:
            years = pd.to_datetime(frame[date_column]).dt.year
            for year, part in frame.groupby(years, sort=True):
                destination = path / f"year={int(year)}"
                destination.mkdir(parents=True, exist_ok=True)
                part.to_parquet(destination / "part-000.parquet", index=False)
        else:
            frame.to_parquet(path / "part-000.parquet", index=False)


class ParquetDataProvider:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def open_snapshot(self, dataset_id: str, verify_hashes: bool = True) -> DataSnapshot:
        return DataSnapshot(self.root / "datasets" / dataset_id, verify_hashes=verify_hashes)

    def list_snapshots(self) -> list[str]:
        base = self.root / "datasets"
        if not base.exists():
            return []
        return sorted(path.name for path in base.iterdir() if (path / "manifest.json").exists())
