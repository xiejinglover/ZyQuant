from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from zyquant.core.exceptions import DataContractError, ResourceError
from zyquant.core.hashing import canonical_json, hash_file, hash_payload


HERMES_SOURCE_TABLES = (
    "md_security",
    "md_trade_cal",
    "mkt_equd",
    "mkt_equd_adj_af",
    "mkt_limit",
    "md_sec_halt",
    "mkt_adjf_af",
    "equ_div_pit",
    "equ_splits",
    "equ_allot",
    "md_inst_type",
    "md_type",
    "equ_inst_sstate",
    "equ_shares_change",
    "equ_free_shares",
    "mkt_equd_eval",
    "mkt_equd_eval_new",
    "mkt_div_yield",
    "vw_fdmt_bs_new",
    "vw_fdmt_is_new",
    "vw_fdmt_cf_new",
)
MONTHLY_TABLES = {
    "mkt_equd": "TRADE_DATE",
    "mkt_equd_adj_af": "TRADE_DATE",
    "mkt_limit": "TRADE_DATE",
    "mkt_equd_eval": "TRADE_DATE",
    "mkt_equd_eval_new": "TRADE_DATE",
    "mkt_div_yield": "TRADE_DATE",
}
FINANCIAL_TABLES = {
    "vw_fdmt_bs_new",
    "vw_fdmt_is_new",
    "vw_fdmt_cf_new",
}
DIRECT_EXCHANGE_TABLES = {
    "md_security",
    "mkt_equd",
    "mkt_equd_adj_af",
    "mkt_limit",
    "md_sec_halt",
    "mkt_adjf_af",
    "equ_div_pit",
    "equ_splits",
    "equ_allot",
    *FINANCIAL_TABLES,
}
SECURITY_ID_TABLES = {
    "mkt_equd",
    "mkt_equd_adj_af",
    "mkt_limit",
    "md_sec_halt",
    "mkt_adjf_af",
    "equ_div_pit",
    "equ_splits",
    "equ_allot",
    "mkt_equd_eval",
    "mkt_equd_eval_new",
    "mkt_div_yield",
    "equ_inst_sstate",
}
DATE_FILTERS = {
    "md_security": "COALESCE(DELIST_DATE, %s) >= %s AND LIST_DATE <= %s",
    "md_trade_cal": "CALENDAR_DATE BETWEEN %s AND %s",
    "md_sec_halt": (
        "DATE(HALT_BEGIN_TIME) <= %s AND "
        "DATE(COALESCE(RESUMP_BEGIN_TIME, %s)) >= %s"
    ),
    "mkt_adjf_af": "EX_DIV_DATE BETWEEN %s AND %s",
    "equ_div_pit": "EX_DIV_DATE BETWEEN %s AND %s",
    "equ_splits": "COALESCE(RE_TRADE_DATE, SPLITS_BASE_DATE) BETWEEN %s AND %s",
    "equ_allot": "EX_RIGHTS_DATE BETWEEN %s AND %s",
    "equ_shares_change": "CHANGE_DATE <= %s",
    "equ_free_shares": "CHANGE_DATE <= %s",
    "md_inst_type": "INTO_DATE <= %s",
    "md_type": "BEGIN_DATE <= %s",
    # A name set long before the window is still the name in force inside it,
    # so take every change up to the end date rather than a bounded range.
    "equ_inst_sstate": "EFF_DATE <= %s",
}


@dataclass(frozen=True)
class HermesCredentials:
    host: str
    user: str
    password: str
    database: str = "hermes"
    port: int = 3306

    @classmethod
    def from_env(cls) -> "HermesCredentials":
        names = {
            "host": "HERMES_MYSQL_HOST",
            "user": "HERMES_MYSQL_USER",
            "password": "HERMES_MYSQL_PASSWORD",
        }
        missing = [environment for environment in names.values() if not os.getenv(environment)]
        if missing:
            raise DataContractError(
                "missing Hermes credentials in environment: " + ", ".join(missing)
            )
        return cls(
            host=os.environ[names["host"]],
            user=os.environ[names["user"]],
            password=os.environ[names["password"]],
            database=os.getenv("HERMES_MYSQL_DATABASE", "hermes"),
            port=int(os.getenv("HERMES_MYSQL_PORT", "3306")),
        )

    def safe_metadata(self) -> dict[str, object]:
        return {
            "host": self.host,
            "database": self.database,
            "port": self.port,
            "user_hash": hashlib.sha256(self.user.encode()).hexdigest()[:16],
        }


@dataclass(frozen=True)
class HermesResourceLimits:
    max_connections: int = 8
    max_compute_processes: int = 32
    compute_threads_per_process: int = 2
    max_logical_cpus: int = 64
    target_memory_gib: float = 78.0
    hard_memory_gib: float = 84.0
    reserve_memory_gib: float = 10.0
    fetch_rows: int = 50_000
    parquet_row_group_rows: int = 262_144

    def __post_init__(self) -> None:
        if not 1 <= self.max_connections <= 8:
            raise ValueError("Hermes RDS connections must be between 1 and 8")
        if self.hard_memory_gib > 84:
            raise ValueError("Hermes hard memory limit cannot exceed 84 GiB")
        if self.target_memory_gib >= self.hard_memory_gib:
            raise ValueError("target memory must be below hard memory")


@dataclass(frozen=True)
class HermesAcquisitionRequest:
    job_id: str = "hermes-cn-a-2010-20260724"
    start_date: date = date(2010, 1, 1)
    end_date: date = date(2026, 7, 24)
    financial_warmup_start: date = date(2009, 1, 1)
    exchanges: tuple[str, ...] = ("XSHG", "XSHE", "XBEI")
    root: Path = Path("data")
    limits: HermesResourceLimits = HermesResourceLimits()

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date must not follow end_date")
        if not self.job_id or "/" in self.job_id or self.job_id.startswith("."):
            raise ValueError("job_id must be a safe directory name")
        unknown = set(self.exchanges) - {"XSHG", "XSHE", "XBEI"}
        if unknown:
            raise ValueError(f"unsupported exchanges: {sorted(unknown)}")

    @property
    def job_root(self) -> Path:
        return self.root.expanduser().resolve() / "acquisitions" / self.job_id


@dataclass(frozen=True)
class ExtractionChunk:
    chunk_id: str
    table_name: str
    partition: str
    sql: str
    parameters: tuple[Any, ...]
    relative_path: str


class AcquisitionState:
    """Controller-owned durable state for a resumable acquisition."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            self.path, check_same_thread=False, timeout=60
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS job (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                source_watermark TEXT NOT NULL,
                schema_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                partition_key TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                rows INTEGER,
                size INTEGER,
                sha256 TEXT,
                started_at TEXT,
                completed_at TEXT,
                error_type TEXT,
                error_message TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_chunks_status
                ON chunks(status, table_name, partition_key);
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def initialize(
        self,
        request: HermesAcquisitionRequest,
        watermark: str,
        schema_hash: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = asdict(request)
        payload["root"] = str(request.root)
        payload["start_date"] = request.start_date.isoformat()
        payload["end_date"] = request.end_date.isoformat()
        payload["financial_warmup_start"] = (
            request.financial_warmup_start.isoformat()
        )
        with self._lock:
            existing = self._connection.execute(
                "SELECT request_json, source_watermark, schema_hash FROM job "
                "WHERE job_id=?",
                (request.job_id,),
            ).fetchone()
            encoded = canonical_json(payload)
            if existing:
                if (
                    existing["request_json"] != encoded
                    or existing["source_watermark"] != watermark
                    or existing["schema_hash"] != schema_hash
                ):
                    raise DataContractError(
                        "acquisition request, watermark or source schema changed"
                    )
                return
            self._connection.execute(
                "INSERT INTO job VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    request.job_id,
                    "queued",
                    encoded,
                    watermark,
                    schema_hash,
                    now,
                    now,
                ),
            )
            self._connection.commit()

    def add_chunks(self, chunks: Sequence[ExtractionChunk]) -> None:
        with self._lock:
            for chunk in chunks:
                query_hash = hash_payload([chunk.sql, chunk.parameters])
                row = self._connection.execute(
                    "SELECT query_hash, relative_path FROM chunks WHERE chunk_id=?",
                    (chunk.chunk_id,),
                ).fetchone()
                if row and (
                    row["query_hash"] != query_hash
                    or row["relative_path"] != chunk.relative_path
                ):
                    raise DataContractError(
                        f"chunk definition changed: {chunk.chunk_id}"
                    )
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO chunks (
                        chunk_id, table_name, partition_key, query_hash,
                        relative_path, status
                    ) VALUES (?, ?, ?, ?, ?, 'queued')
                    """,
                    (
                        chunk.chunk_id,
                        chunk.table_name,
                        chunk.partition,
                        query_hash,
                        chunk.relative_path,
                    ),
                )
            self._connection.commit()

    def prepare_resume(self, root: Path) -> None:
        with self._lock:
            rows = self._connection.execute(
                "SELECT chunk_id, relative_path, sha256 FROM chunks "
                "WHERE status='succeeded'"
            ).fetchall()
            for row in rows:
                path = root / row["relative_path"]
                if not path.exists() or hash_file(path) != row["sha256"]:
                    self._connection.execute(
                        "UPDATE chunks SET status='queued', rows=NULL, size=NULL, "
                        "sha256=NULL, error_type='IntegrityError', "
                        "error_message='completed file missing or hash mismatch' "
                        "WHERE chunk_id=?",
                        (row["chunk_id"],),
                    )
            self._connection.execute(
                "UPDATE chunks SET status='queued', error_type='Interrupted', "
                "error_message='previous process stopped before commit' "
                "WHERE status='running'"
            )
            self._connection.commit()

    def start(self, chunk_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE chunks SET status='running', attempts=attempts+1,
                    started_at=?, error_type=NULL, error_message=NULL
                WHERE chunk_id=? AND status IN ('queued', 'failed')
                """,
                (datetime.now(timezone.utc).isoformat(), chunk_id),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def succeed(self, chunk_id: str, rows: int, path: Path) -> None:
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            self._connection.execute(
                """
                UPDATE chunks SET status='succeeded', rows=?, size=?, sha256=?,
                    completed_at=? WHERE chunk_id=?
                """,
                (
                    rows,
                    path.stat().st_size,
                    hash_file(path),
                    now,
                    chunk_id,
                ),
            )
            self._connection.execute(
                "UPDATE job SET updated_at=?", (now,)
            )
            self._connection.commit()

    def fail(self, chunk_id: str, error: BaseException) -> None:
        message = str(error)
        for secret_name in ("HERMES_MYSQL_PASSWORD",):
            secret = os.getenv(secret_name)
            if secret:
                message = message.replace(secret, "<redacted>")
        with self._lock:
            self._connection.execute(
                """
                UPDATE chunks SET status='failed', error_type=?,
                    error_message=?, completed_at=? WHERE chunk_id=?
                """,
                (
                    type(error).__name__,
                    message[:2000],
                    datetime.now(timezone.utc).isoformat(),
                    chunk_id,
                ),
            )
            self._connection.commit()

    def set_job_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE job SET status=?, updated_at=?, error=?",
                (status, datetime.now(timezone.utc).isoformat(), error),
            )
            self._connection.commit()

    def status(self) -> dict[str, Any]:
        job = self._connection.execute("SELECT * FROM job").fetchone()
        counts = self._connection.execute(
            "SELECT status, COUNT(*) AS count, COALESCE(SUM(rows), 0) AS rows, "
            "COALESCE(SUM(size), 0) AS size FROM chunks GROUP BY status"
        ).fetchall()
        tables = self._connection.execute(
            "SELECT table_name, status, COUNT(*) AS count, "
            "COALESCE(SUM(rows), 0) AS rows FROM chunks "
            "GROUP BY table_name, status ORDER BY table_name, status"
        ).fetchall()
        return {
            "job": dict(job) if job else None,
            "chunks": [dict(row) for row in counts],
            "tables": [dict(row) for row in tables],
        }


class HermesMySQLClient:
    def __init__(
        self,
        credentials: HermesCredentials,
        connect_timeout: int = 20,
        read_timeout: int = 1800,
    ):
        self.credentials = credentials
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

    def connect(self, streaming: bool = True):
        try:
            import pymysql
        except ImportError as exc:
            raise DataContractError(
                "Hermes connector requires: pip install 'zyquant[hermes]'"
            ) from exc
        cursor = (
            pymysql.cursors.SSDictCursor
            if streaming
            else pymysql.cursors.DictCursor
        )
        connection = pymysql.connect(
            host=self.credentials.host,
            port=self.credentials.port,
            user=self.credentials.user,
            password=self.credentials.password,
            database=self.credentials.database,
            charset="utf8mb4",
            autocommit=False,
            cursorclass=cursor,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            write_timeout=30,
        )
        with connection.cursor() as current:
            current.execute("SET SESSION TRANSACTION READ ONLY")
            current.execute("SET SESSION time_zone = '+00:00'")
            current.execute("START TRANSACTION READ ONLY")
        return connection

    def scalar(self, sql: str, parameters: Sequence[Any] = ()) -> Any:
        connection = self.connect(streaming=False)
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, parameters)
                row = cursor.fetchone()
                return next(iter(row.values())) if row else None
        finally:
            connection.rollback()
            connection.close()

    def rows(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
        fetch_rows: int = 50_000,
    ) -> Iterator[list[Mapping[str, Any]]]:
        connection = self.connect(streaming=True)
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, parameters)
                while True:
                    rows = cursor.fetchmany(fetch_rows)
                    if not rows:
                        break
                    yield rows
        finally:
            connection.rollback()
            connection.close()

    def schema_inventory(self) -> list[Mapping[str, Any]]:
        connection = self.connect(streaming=False)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT TABLE_NAME, COLUMN_NAME, ORDINAL_POSITION, DATA_TYPE,
                           IS_NULLABLE, COLUMN_KEY, COLUMN_COMMENT
                    FROM information_schema.columns
                    WHERE table_schema=%s
                    ORDER BY TABLE_NAME, ORDINAL_POSITION
                    """,
                    (self.credentials.database,),
                )
                return list(cursor.fetchall())
        finally:
            connection.rollback()
            connection.close()


class ResourceController:
    def __init__(self, limits: HermesResourceLimits):
        self.limits = limits

    def wait_for_capacity(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        while True:
            memory = psutil.virtual_memory()
            used_gib = (memory.total - memory.available) / (1024**3)
            available_gib = memory.available / (1024**3)
            if (
                used_gib < self.limits.hard_memory_gib
                and available_gib >= self.limits.reserve_memory_gib
            ):
                return
            time.sleep(2.0)


class HermesExtractionPlanner:
    def __init__(self, request: HermesAcquisitionRequest, watermark: str):
        self.request = request
        self.watermark = watermark

    def plan(self, party_groups: Sequence[Sequence[int]]) -> list[ExtractionChunk]:
        chunks: list[ExtractionChunk] = []
        for table in HERMES_SOURCE_TABLES:
            if table in MONTHLY_TABLES:
                chunks.extend(self._monthly(table, MONTHLY_TABLES[table]))
            elif table in FINANCIAL_TABLES:
                chunks.extend(self._financial(table, party_groups))
            else:
                chunks.append(self._single(table))
        return chunks

    def _base_predicates(self, table: str) -> tuple[list[str], list[Any]]:
        predicates: list[str] = []
        parameters: list[Any] = []
        if table in DIRECT_EXCHANGE_TABLES:
            placeholders = ", ".join(["%s"] * len(self.request.exchanges))
            predicates.append(f"EXCHANGE_CD IN ({placeholders})")
            parameters.extend(self.request.exchanges)
        if table == "md_security":
            predicates.append("ASSET_CLASS='E'")
        if table == "md_trade_cal":
            placeholders = ", ".join(["%s"] * len(self.request.exchanges))
            predicates.extend([
                f"EXCHANGE_CD IN ({placeholders})",
                "IS_OPEN=1",
            ])
            parameters.extend(self.request.exchanges)
        if table == "equ_div_pit":
            predicates.append("EVENT_PROCESS_CD=6")
        if table in SECURITY_ID_TABLES:
            predicates.append(
                "EXISTS (SELECT 1 FROM md_security zyq_s "
                f"WHERE zyq_s.SECURITY_ID={table}.SECURITY_ID "
                "AND zyq_s.ASSET_CLASS='E' "
                "AND zyq_s.TRANS_CURR_CD='CNY' "
                "AND zyq_s.EXCHANGE_CD IN ('XSHG','XSHE','XBEI') "
                "AND zyq_s.LIST_DATE <= %s "
                "AND COALESCE(zyq_s.DELIST_DATE, %s) >= %s)"
            )
            parameters.extend([
                self.request.end_date,
                self.request.end_date,
                self.request.start_date,
            ])
        if table in FINANCIAL_TABLES:
            predicates.append("END_DATE_REP >= %s")
            parameters.append(self.request.financial_warmup_start)
            predicates.append("ACT_PUBTIME < %s")
            parameters.append(
                datetime.combine(
                    self.request.end_date, datetime.max.time()
                )
            )
        if table in {"equ_shares_change", "equ_free_shares", "md_inst_type"}:
            predicates.append(
                "EXISTS (SELECT 1 FROM md_security s "
                f"WHERE s.PARTY_ID={table}.PARTY_ID AND s.ASSET_CLASS='E' "
                "AND s.EXCHANGE_CD IN ('XSHG','XSHE','XBEI'))"
            )
        predicates.append("(UPDATE_TIME IS NULL OR UPDATE_TIME <= %s)")
        parameters.append(self.watermark)
        return predicates, parameters

    def _single(self, table: str) -> ExtractionChunk:
        predicates, parameters = self._base_predicates(table)
        date_filter = DATE_FILTERS.get(table)
        if date_filter:
            predicates.append(date_filter)
            if table == "md_security":
                parameters.extend([
                    self.request.end_date,
                    self.request.start_date,
                    self.request.end_date,
                ])
            elif table == "md_sec_halt":
                parameters.extend([
                    self.request.end_date,
                    self.request.end_date,
                    self.request.start_date,
                ])
            elif table in {
                "equ_shares_change", "equ_free_shares",
                "md_inst_type", "md_type", "equ_inst_sstate",
            }:
                parameters.append(self.request.end_date)
            else:
                parameters.extend([self.request.start_date, self.request.end_date])
        where = " AND ".join(predicates) or "1=1"
        sql = f"SELECT * FROM `{table}` WHERE {where} ORDER BY ID"
        if table == "md_security":
            sql = f"SELECT * FROM `{table}` WHERE {where} ORDER BY SECURITY_ID"
        chunk_id = f"{table}:all"
        return ExtractionChunk(
            chunk_id, table, "all", sql, tuple(parameters),
            f"raw/{table}/part-all.parquet",
        )

    def _monthly(self, table: str, date_column: str) -> list[ExtractionChunk]:
        output = []
        cursor = self.request.start_date.replace(day=1)
        while cursor <= self.request.end_date:
            if cursor.month == 12:
                next_month = cursor.replace(
                    year=cursor.year + 1, month=1
                )
            else:
                next_month = cursor.replace(month=cursor.month + 1)
            upper = min(self.request.end_date, next_month - timedelta(days=1))
            predicates, parameters = self._base_predicates(table)
            predicates.append(f"{date_column} BETWEEN %s AND %s")
            parameters.extend([cursor, upper])
            partition = cursor.strftime("%Y-%m")
            sql = (
                f"SELECT * FROM `{table}` WHERE "
                + " AND ".join(predicates)
                + f" ORDER BY {date_column}, SECURITY_ID, ID"
            )
            output.append(ExtractionChunk(
                f"{table}:{partition}",
                table,
                partition,
                sql,
                tuple(parameters),
                f"raw/{table}/year={cursor.year}/month={cursor.month:02d}/"
                f"part-{partition}.parquet",
            ))
            cursor = next_month
        return output

    def _financial(
        self, table: str, party_groups: Sequence[Sequence[int]]
    ) -> list[ExtractionChunk]:
        output = []
        for number, party_ids in enumerate(party_groups):
            predicates, parameters = self._base_predicates(table)
            placeholders = ", ".join(["%s"] * len(party_ids))
            predicates.append(f"PARTY_ID IN ({placeholders})")
            parameters.extend(party_ids)
            partition = f"group-{number:05d}"
            sql = (
                f"SELECT * FROM `{table}` WHERE "
                + " AND ".join(predicates)
                + " ORDER BY PARTY_ID, END_DATE_REP, ACT_PUBTIME, ID"
            )
            output.append(ExtractionChunk(
                f"{table}:{partition}",
                table,
                partition,
                sql,
                tuple(parameters),
                f"raw/{table}/{partition}.parquet",
            ))
        return output


class HermesDataAdapter:
    """Hermes-specific read-only, resumable acquisition adapter."""

    def __init__(
        self,
        credentials: HermesCredentials | None = None,
        client: HermesMySQLClient | None = None,
    ):
        self.credentials = credentials or HermesCredentials.from_env()
        self.client = client or HermesMySQLClient(self.credentials)

    def ingest(self, request: Mapping[str, Any] | None = None):
        raise DataContractError(
            "Hermes full-market ingestion is streaming; use "
            "'zyq data acquire --source hermes --action run' and then publish"
        )

    def run(
        self,
        request: HermesAcquisitionRequest,
        resume: bool = False,
    ) -> dict[str, Any]:
        root = request.job_root
        for name in ("raw", "canonical", "quarantine", "logs", ".partial"):
            (root / name).mkdir(parents=True, exist_ok=True)
        state = AcquisitionState(root / "state.sqlite")
        try:
            inventory = self.client.schema_inventory()
            self._validate_inventory(inventory)
            arrow_schemas = self._arrow_schemas(inventory)
            schema_hash = hash_payload(inventory)
            inventory_path = root / "source_schema.json"
            if inventory_path.exists():
                existing = json.loads(inventory_path.read_text(encoding="utf-8"))
                if hash_payload(existing) != schema_hash:
                    raise DataContractError(
                        "Hermes source schema changed after acquisition started"
                    )
            else:
                temporary = root / ".partial" / "source_schema.json.tmp"
                temporary.write_text(
                    json.dumps(inventory, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                os.replace(temporary, inventory_path)
            state_row = state.status()["job"]
            if state_row:
                watermark = str(state_row["source_watermark"])
            else:
                watermark = str(
                    self.client.scalar("SELECT UTC_TIMESTAMP(6)")
                )
            state.initialize(request, watermark, schema_hash)
            parties = self._party_ids(request)
            groups = [
                parties[index:index + 100]
                for index in range(0, len(parties), 100)
            ]
            chunks = HermesExtractionPlanner(request, watermark).plan(groups)
            state.add_chunks(chunks)
            if resume:
                state.prepare_resume(root)
            state.set_job_status("running")
            pending_status = {
                row["chunk_id"]: row["status"]
                for row in state._connection.execute(
                    "SELECT chunk_id, status FROM chunks"
                )
            }
            pending = [
                item for item in chunks
                if pending_status.get(item.chunk_id) != "succeeded"
            ]
            controller = ResourceController(request.limits)
            failures: list[tuple[str, BaseException]] = []
            with ThreadPoolExecutor(
                max_workers=request.limits.max_connections,
                thread_name_prefix="hermes-rds",
            ) as executor:
                futures = {
                    executor.submit(
                        self._extract_chunk,
                        item,
                        request,
                        state,
                        controller,
                        arrow_schemas[item.table_name],
                    ): item
                    for item in pending
                }
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        future.result()
                    except BaseException as exc:
                        failures.append((item.chunk_id, exc))
            if failures:
                state.set_job_status(
                    "failed",
                    f"{len(failures)} chunks failed; first={failures[0][0]}:"
                    f"{type(failures[0][1]).__name__}",
                )
                raise DataContractError(
                    f"Hermes acquisition has {len(failures)} failed chunks; "
                    "run status, then resume"
                )
            state.set_job_status("acquired")
            return state.status()
        finally:
            state.close()

    def _party_ids(self, request: HermesAcquisitionRequest) -> list[int]:
        sql = """
            SELECT DISTINCT PARTY_ID
            FROM md_security
            WHERE ASSET_CLASS='E'
              AND EXCHANGE_CD IN ('XSHG','XSHE','XBEI')
              AND TRANS_CURR_CD='CNY'
              AND LIST_DATE <= %s
              AND COALESCE(DELIST_DATE, %s) >= %s
              AND PARTY_ID IS NOT NULL
            ORDER BY PARTY_ID
        """
        values: list[int] = []
        for batch in self.client.rows(
            sql,
            (request.end_date, request.end_date, request.start_date),
            fetch_rows=20_000,
        ):
            values.extend(int(row["PARTY_ID"]) for row in batch)
        return values

    @staticmethod
    def _validate_inventory(inventory: Sequence[Mapping[str, Any]]) -> None:
        present = {str(row["TABLE_NAME"]) for row in inventory}
        missing = set(HERMES_SOURCE_TABLES) - present
        if missing:
            raise DataContractError(
                f"Hermes is missing required source tables: {sorted(missing)}"
            )

    def _extract_chunk(
        self,
        chunk: ExtractionChunk,
        request: HermesAcquisitionRequest,
        state: AcquisitionState,
        controller: ResourceController,
        source_schema: pa.Schema,
    ) -> None:
        if not state.start(chunk.chunk_id):
            return
        destination = request.job_root / chunk.relative_path
        temporary = (
            request.job_root / ".partial"
            / f"{hashlib.sha256(chunk.chunk_id.encode()).hexdigest()}.parquet.tmp"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        for retry in range(3):
            rows_written = 0
            writer: pq.ParquetWriter | None = None
            try:
                controller.wait_for_capacity()
                for rows in self.client.rows(
                    chunk.sql,
                    chunk.parameters,
                    request.limits.fetch_rows,
                ):
                    controller.wait_for_capacity()
                    table = pa.Table.from_pylist(
                        [
                            {
                                key: (
                                    float(value)
                                    if isinstance(value, Decimal)
                                    else value
                                )
                                for key, value in row.items()
                            }
                            for row in rows
                        ],
                        schema=source_schema,
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary,
                            table.schema,
                            compression="zstd",
                            compression_level=6,
                            use_dictionary=True,
                        )
                    writer.write_table(
                        table,
                        row_group_size=request.limits.parquet_row_group_rows,
                    )
                    rows_written += table.num_rows
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        source_schema,
                        compression="zstd",
                    )
                writer.close()
                writer = None
                os.replace(temporary, destination)
                state.succeed(chunk.chunk_id, rows_written, destination)
                return
            except BaseException as exc:
                if writer is not None:
                    writer.close()
                temporary.unlink(missing_ok=True)
                if self._is_transient(exc) and retry < 2:
                    time.sleep(2**retry)
                    continue
                state.fail(chunk.chunk_id, exc)
                if self._is_transient(exc):
                    raise ResourceError(str(exc)) from exc
                raise

    @staticmethod
    def _is_transient(error: BaseException) -> bool:
        if isinstance(error, (TimeoutError, ConnectionError, OSError)):
            return True
        arguments = getattr(error, "args", ())
        return bool(arguments and arguments[0] in {
            1040, 1205, 1213, 2006, 2013,
        })

    @staticmethod
    def status(root: str | Path, job_id: str) -> dict[str, Any]:
        job_root = Path(root).expanduser().resolve() / "acquisitions" / job_id
        state_path = job_root / "state.sqlite"
        if not state_path.exists():
            raise DataContractError(f"acquisition does not exist: {job_id}")
        state = AcquisitionState(state_path)
        try:
            return state.status()
        finally:
            state.close()

    @staticmethod
    def _arrow_schemas(
        inventory: Sequence[Mapping[str, Any]],
    ) -> dict[str, pa.Schema]:
        type_map = {
            "bigint": pa.int64(),
            "int": pa.int64(),
            "integer": pa.int64(),
            "mediumint": pa.int64(),
            "smallint": pa.int64(),
            "tinyint": pa.int64(),
            "decimal": pa.float64(),
            "numeric": pa.float64(),
            "double": pa.float64(),
            "float": pa.float64(),
            "date": pa.date32(),
            "datetime": pa.timestamp("us"),
            "timestamp": pa.timestamp("us"),
            "time": pa.string(),
            "char": pa.string(),
            "varchar": pa.string(),
            "text": pa.string(),
            "tinytext": pa.string(),
            "mediumtext": pa.string(),
            "longtext": pa.string(),
            "json": pa.string(),
            "blob": pa.binary(),
            "binary": pa.binary(),
            "varbinary": pa.binary(),
        }
        fields: dict[str, list[pa.Field]] = {
            table: [] for table in HERMES_SOURCE_TABLES
        }
        for row in inventory:
            table = str(row["TABLE_NAME"])
            if table not in fields:
                continue
            mysql_type = str(row["DATA_TYPE"]).lower()
            arrow_type = type_map.get(mysql_type)
            if arrow_type is None:
                raise DataContractError(
                    f"unsupported Hermes MySQL type {mysql_type!r} "
                    f"for {table}.{row['COLUMN_NAME']}"
                )
            fields[table].append(pa.field(
                str(row["COLUMN_NAME"]),
                arrow_type,
                nullable=str(row.get("IS_NULLABLE", "YES")).upper() == "YES",
            ))
        return {
            table: pa.schema(columns)
            for table, columns in fields.items()
        }
