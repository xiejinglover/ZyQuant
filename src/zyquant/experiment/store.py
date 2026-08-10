from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from zyquant.core.exceptions import ExperimentError, SchemaVersionError
from zyquant.core.hashing import canonical_json, hash_file
from zyquant.core.versioning import RUN_SCHEMA_VERSION


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_type: str
    status: str
    created_at: str
    parent_run_id: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentStore:
    """SQLite metadata index. Callers must keep all writes in the controller."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _migrate(self) -> None:
        existing = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_info'"
        ).fetchone()
        if existing:
            version = self.connection.execute(
                "SELECT version FROM schema_info WHERE name='experiment'"
            ).fetchone()
            if version is None or version["version"] != RUN_SCHEMA_VERSION:
                raise SchemaVersionError(
                    "experiment database is not v1; create a new database"
                )
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS schema_info (
            name TEXT PRIMARY KEY,
            version TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            run_type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            parent_run_id TEXT REFERENCES runs(run_id),
            config_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS parameters (
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            value_json TEXT NOT NULL,
            PRIMARY KEY (run_id, name)
        );
        CREATE TABLE IF NOT EXISTS metrics (
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            value REAL,
            value_json TEXT,
            PRIMARY KEY (run_id, name)
        );
        CREATE TABLE IF NOT EXISTS artifacts (
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size INTEGER NOT NULL,
            schema_version TEXT NOT NULL,
            PRIMARY KEY (run_id, name)
        );
        CREATE TABLE IF NOT EXISTS trials (
            trial_key TEXT PRIMARY KEY,
            search_run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            trial_run_id TEXT NOT NULL,
            status TEXT NOT NULL,
            parameters_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            objective REAL,
            error TEXT,
            attempts INTEGER NOT NULL,
            heartbeat_at TEXT NOT NULL
        );
        """)
        self.connection.execute(
            "INSERT OR IGNORE INTO schema_info VALUES ('experiment', ?)",
            (RUN_SCHEMA_VERSION,),
        )
        self.connection.commit()

    def start_run(
        self,
        run_id: str,
        run_type: str,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        parent_run_id: str | None = None,
    ) -> RunRecord:
        created = _now()
        try:
            self.connection.execute(
                "INSERT INTO runs VALUES (?, ?, 'running', ?, ?, NULL, ?, ?, ?, NULL)",
                (
                    run_id, run_type, created, created, parent_run_id,
                    canonical_json(config), canonical_json(metadata or {}),
                ),
            )
            self.connection.commit()
        except sqlite3.IntegrityError as exc:
            raise ExperimentError(f"run already exists or has invalid parent: {run_id}") from exc
        return RunRecord(run_id, run_type, "running", created, parent_run_id)

    def finish_run(self, run_id: str, metrics: Mapping[str, Any]) -> None:
        for name, value in metrics.items():
            numeric = (
                float(value)
                if isinstance(value, (int, float)) and value is not None else None
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO metrics VALUES (?, ?, ?, ?)",
                (run_id, name, numeric, canonical_json(value)),
            )
        now = _now()
        self.connection.execute(
            "UPDATE runs SET status='succeeded', updated_at=?, completed_at=? WHERE run_id=?",
            (now, now, run_id),
        )
        self.connection.commit()

    def fail_run(self, run_id: str, error: str, status: str = "failed") -> None:
        if status not in {"failed", "interrupted"}:
            raise ValueError("terminal run status must be failed or interrupted")
        now = _now()
        self.connection.execute(
            "UPDATE runs SET status=?, updated_at=?, completed_at=?, error=? WHERE run_id=?",
            (status, now, now, error, run_id),
        )
        self.connection.commit()

    def log_parameters(self, run_id: str, parameters: Mapping[str, Any]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO parameters VALUES (?, ?, ?)",
            [(run_id, key, canonical_json(value)) for key, value in parameters.items()],
        )
        self.connection.commit()

    def log_artifact(
        self, run_id: str, name: str, path: str | Path,
        schema_version: str = "1.0",
    ) -> None:
        source = Path(path).expanduser().resolve()
        self.connection.execute(
            "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id, name, str(source), hash_file(source),
                source.stat().st_size, schema_version,
            ),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()

    def list_runs(self, limit: int = 100) -> list[sqlite3.Row]:
        return list(self.connection.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ))

    def compare_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        result = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run is None:
                continue
            metrics = {
                row["name"]: json.loads(row["value_json"])
                for row in self.connection.execute(
                    "SELECT name, value_json FROM metrics WHERE run_id=?", (run_id,)
                )
            }
            result.append({
                "run_id": run_id, "run_type": run["run_type"],
                "status": run["status"], **metrics,
            })
        return result

    def get_trial(self, trial_key: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM trials WHERE trial_key=?", (trial_key,)
        ).fetchone()

    def upsert_trial(
        self,
        trial_key: str,
        search_run_id: str,
        trial_run_id: str,
        status: str,
        parameters: Mapping[str, Any],
        objective: float | None = None,
        error: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        attempts: int = 1,
    ) -> None:
        if status not in {
            "queued", "running", "succeeded", "failed", "interrupted",
            "constrained", "completed_ineligible_return_floor",
            "technical_failed",
        }:
            raise ValueError(f"invalid trial status: {status}")
        self.connection.execute(
            "INSERT OR REPLACE INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trial_key, search_run_id, trial_run_id, status,
                canonical_json(parameters), canonical_json(metrics or {}),
                objective, error, attempts, _now(),
            ),
        )
        self.connection.commit()

    def heartbeat(self, trial_keys: list[str]) -> None:
        now = _now()
        self.connection.executemany(
            "UPDATE trials SET heartbeat_at=? WHERE trial_key=? AND status='running'",
            [(now, key) for key in trial_keys],
        )
        self.connection.commit()

    def interrupt_running_trials(self, search_run_id: str) -> None:
        self.connection.execute(
            "UPDATE trials SET status='interrupted', heartbeat_at=? "
            "WHERE search_run_id=? AND status IN ('queued','running')",
            (_now(), search_run_id),
        )
        self.connection.commit()
