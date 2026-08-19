from __future__ import annotations

import json
import os
import shutil
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from zyquant.core.exceptions import DataContractError
from zyquant.core.hashing import hash_file, hash_payload
from zyquant.core.versioning import SNAPSHOT_SCHEMA_VERSION

from zyquant.data.contracts import (
    FIELD_SPECS, FINANCIAL_TABLES, REQUIRED_COLUMNS,
)
from zyquant.data.financial import (
    FUNDAMENTAL_CALCULATION_VERSION,
    ITEM_CATALOG_VERSION,
    FinancialProcessor,
)
from .acquisition import AcquisitionState, HermesAcquisitionRequest
from zyquant.data.normalization import normalize_table
from zyquant.data.snapshot import DataSnapshot


FINANCIAL_SOURCE_MAP = {
    "vw_fdmt_bs_new": "balance",
    "vw_fdmt_is_new": "income",
    "vw_fdmt_cf_new": "cash_flow",
}
HERMES_DIRECT_METRICS = {
    "fdmt_main_data_q_pit": {
        "ROE": ("hermes_roe_q_pct", "single_quarter", "percent", 1.0),
        "ROE_CUT": (
            "hermes_inc_return_q_pct", "single_quarter", "percent", 1.0,
        ),
        "T_REVENUE_YOY": (
            "hermes_total_revenue_yoy_q_pct", "single_quarter", "percent", 1.0,
        ),
        "NI_YOY": (
            "hermes_net_profit_yoy_q_pct", "single_quarter", "percent", 1.0,
        ),
    },
    "fdmt_md_n_ttmp": {
        "EPS": ("hermes_eps_ttm", "ttm", "CNY/share", 1.0),
        "N_CF_OPA_NIA": (
            # Hermes stores this percentage-style (90 means 0.90), while the
            # formula fallback is a dimensionless ratio. Normalize before the
            # two sources are combined cross-sectionally.
            "hermes_nocf_coverage_ttm", "ttm", "ratio", 0.01,
        ),
    },
}
FINANCIAL_ITEM_MAP = {
    "T_ASSETS": "total_assets",
    "T_LIAB": "total_liability",
    "T_SH_EQUITY": "total_owner_equities",
    "T_EQUITY_ATTR_P": "equities_parent_company_owners",
    "T_CA": "total_current_assets",
    "T_CL": "total_current_liability",
    "T_REVENUE": "total_operating_revenue",
    "REVENUE": "operating_revenue",
    "T_COGS": "operating_cost",
    "OPERATE_PROFIT": "operating_profit",
    "N_INCOME": "net_profit",
    "N_INCOME_ATTR_P": "np_parent_company_owners",
    "BASIC_EPS": "basic_eps",
    "DILUTED_EPS": "diluted_eps",
    "INT_INCOME": "interest_income",
    "PREM_EARNED": "premiums_earned",
    "COMMIS_INCOME": "commission_income",
    "N_CF_OPERATE_A": "net_operate_cash_flow",
    "PUR_FIX_ASSETS_OTH": "fix_intan_other_asset_acqui_cash",
}
FINANCIAL_METADATA = {
    "ID", "PARTY_ID", "TICKER_SYMBOL", "EXCHANGE_CD", "ACT_PUBTIME",
    "PUBLISH_DATE", "END_DATE_REP", "END_DATE", "REPORT_TYPE",
    "FISCAL_PERIOD", "MERGED_FLAG", "ACCOUTING_STANDARDS",
    "CURRENCY_CD", "INDUSTRY_CATEGORY", "UPDATE_TIME",
}
MONEY_FLOW_SOURCE_FIELDS = {
    "INFLOW": "inflow",
    "OUTFLOW": "outflow",
    "NET_FLOW": "net_inflow",
    "INFLOW_S": "inflow_s",
    "INFLOW_M": "inflow_m",
    "INFLOW_L": "inflow_l",
    "INFLOW_XL": "inflow_xl",
    "OUTFLOW_S": "outflow_s",
    "OUTFLOW_M": "outflow_m",
    "OUTFLOW_L": "outflow_l",
    "OUTFLOW_XL": "outflow_xl",
    "NET_FLOW_S": "net_inflow_s",
    "NET_FLOW_M": "net_inflow_m",
    "NET_FLOW_L": "net_inflow_l",
    "NET_FLOW_XL": "net_inflow_xl",
    "MAIN_FLOW": "main_net_inflow",
    "SMAIN_FLOW": "retail_net_inflow",
    "NET_IN_OPN": "net_in_open",
    "NET_IN_CLS": "net_in_close",
    "TURNOVER_VALUE": "turnover_value",
}
MONEY_FLOW_MAPPER_VERSION = "1"


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "__empty__" in frame:
        return pd.DataFrame()
    return frame


def _write(
    root: Path,
    table: str,
    partition: str,
    frame: pd.DataFrame,
    normalize: bool = True,
) -> Path:
    output = normalize_table(table, frame) if normalize else frame
    destination = root / table / partition
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = root / ".partial" / f"{table}-{hash_payload(partition)[:20]}.tmp"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    arrow = pa.Table.from_pandas(output, preserve_index=False)
    pq.write_table(
        arrow,
        temporary,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        row_group_size=262_144,
    )
    os.replace(temporary, destination)
    return destination


def _instrument_id(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["TICKER_SYMBOL"].astype(str)
        + "."
        + frame["EXCHANGE_CD"].astype(str)
    )


def _source_fields(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    identifiers = frame.get("ID")
    if identifiers is None:
        identifiers = frame.get("SECURITY_ID")
    if identifiers is None:
        identifiers = pd.Series(range(len(frame)), index=frame.index)
    return {
        "source_record_id": prefix + ":" + identifiers.astype(str),
        "source_updated_at": pd.to_datetime(
            frame.get("UPDATE_TIME"), utc=True, errors="coerce"
        ),
    }


def _finite_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def _cumulative_flow_rows(
    source: pd.DataFrame, statement_name: str,
) -> pd.DataFrame:
    """Keep YTD rows when Hermes publishes YTD and standalone-quarter rows.

    `FISCAL_PERIOD` is the number of represented months.  Q3 income commonly
    contains both period=3 (standalone Q3) and period=9 (YTD) with the same
    report date.  FinancialProcessor expects YTD facts and derives
    `single_quarter`, so accepting both let an arbitrary report-id hash choose
    the semantic.
    """
    if statement_name not in {"income", "cash_flow"}:
        return source
    period = pd.to_numeric(source["FISCAL_PERIOD"], errors="coerce")
    expected = pd.to_datetime(source["END_DATE"]).dt.month.astype(float)
    keep = period.isna() | period.eq(expected)
    return source.loc[keep].copy()


def _direct_metric_rows(
    raw: Path,
    filename: str,
    batch_id: str,
    instrument_by_party: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_name, field_map in HERMES_DIRECT_METRICS.items():
        source = _read(raw / source_name / filename)
        if source.empty:
            continue
        if "MERGED_FLAG" in source:
            source = source[source["MERGED_FLAG"].astype(str).eq("1")].copy()
        source["PUBLISH_DATE"] = pd.to_datetime(
            source["PUBLISH_DATE"], errors="coerce"
        ).dt.date
        source["END_DATE"] = pd.to_datetime(
            source["END_DATE"], errors="coerce"
        ).dt.date
        source["UPDATE_TIME"] = pd.to_datetime(
            source.get("UPDATE_TIME"), utc=True, errors="coerce"
        )
        source = source.dropna(subset=["PARTY_ID", "PUBLISH_DATE", "END_DATE"])
        source = source.sort_values(
            ["PARTY_ID", "END_DATE", "PUBLISH_DATE", "UPDATE_TIME", "ID"],
            kind="mergesort",
        ).drop_duplicates(
            ["PARTY_ID", "END_DATE", "PUBLISH_DATE"], keep="last"
        )
        for row in source.itertuples(index=False):
            instrument_id = instrument_by_party.get(int(row.PARTY_ID))
            if instrument_id is None:
                continue
            for source_field, (metric_code, basis, unit, scale) in field_map.items():
                value = _finite_or_none(getattr(row, source_field, None))
                if value is None:
                    continue
                value *= scale
                payload = {
                    "source": source_name,
                    "id": str(row.ID),
                    "field": source_field,
                    "metric": metric_code,
                    "period": row.END_DATE.isoformat(),
                    "available": row.PUBLISH_DATE.isoformat(),
                    "version": "hermes-direct-pit-v1",
                }
                metric_id = hash_payload(payload)
                rows.append({
                    "metric_id": metric_id,
                    "instrument_id": instrument_id,
                    "metric_code": metric_code,
                    "fiscal_period_end": row.END_DATE,
                    "basis": basis,
                    "value": value,
                    "unit": unit,
                    "available_at": row.PUBLISH_DATE,
                    "calculation_version": "hermes-direct-pit-v1",
                    "source_report_ids": json.dumps(
                        [f"{source_name}:{row.ID}"], separators=(",", ":")
                    ),
                    "quality_status": "complete",
                    "source_record_id": f"{source_name}:{row.ID}:{source_field}",
                    "source_batch_id": batch_id,
                    "source_updated_at": row.UPDATE_TIME,
                })
    return pd.DataFrame(rows)


def _financial_group_worker(
    raw_root: str,
    canonical_root: str,
    filename: str,
    trade_days: list[date],
    batch_id: str,
) -> dict[str, int]:
    raw = Path(raw_root)
    canonical = Path(canonical_root)
    statements: dict[str, pd.DataFrame] = {}
    instrument_by_party: dict[int, str] = {}
    excluded = 0
    for source_name, statement_name in FINANCIAL_SOURCE_MAP.items():
        source = _read(raw / source_name / filename)
        if source.empty:
            statements[statement_name] = pd.DataFrame()
            continue
        consolidated = source["MERGED_FLAG"].astype(str).eq("1")
        excluded += int((~consolidated).sum())
        source = source[consolidated].copy()
        instrument_by_party.update({
            int(party): str(code)
            for party, code in zip(
                pd.to_numeric(source["PARTY_ID"], errors="coerce"),
                _instrument_id(source),
            )
            if pd.notna(party)
        })
        source = _cumulative_flow_rows(source, statement_name)
        converted = pd.DataFrame({
            "id": source["ID"],
            "code": _instrument_id(source),
            "pub_date": pd.to_datetime(source["ACT_PUBTIME"]).dt.date,
            "start_date": pd.to_datetime(
                source["END_DATE"]
            ).dt.to_period("Y").dt.start_time.dt.date,
            "end_date": source["END_DATE"],
            "report_date": source["END_DATE_REP"],
            "report_type": np.where(
                pd.to_datetime(source["END_DATE"]).eq(
                    pd.to_datetime(source["END_DATE_REP"])
                ),
                0,
                1,
            ),
        })
        numeric_columns: dict[str, pd.Series] = {}
        for column in source.columns:
            if column in FINANCIAL_METADATA:
                continue
            values = pd.to_numeric(source[column], errors="coerce")
            if values.notna().any():
                code = FINANCIAL_ITEM_MAP.get(column, column.lower())
                numeric_columns[code] = values
        if numeric_columns:
            converted = pd.concat(
                [converted, pd.DataFrame(numeric_columns)],
                axis=1,
            )
        statements[statement_name] = converted
    result = FinancialProcessor().build(
        statements, trade_days, batch_id, source_name="Hermes"
    )
    direct = _direct_metric_rows(
        raw, filename, batch_id, instrument_by_party
    )
    metrics = result.metrics
    if not direct.empty:
        metrics = pd.concat([metrics, direct], ignore_index=True)
    _write(
        canonical, "financial_reports", filename,
        result.reports,
    )
    _write(
        canonical, "financial_facts", filename,
        result.facts,
    )
    _write(
        canonical, "fundamental_metrics", filename,
        metrics,
    )
    return {
        "reports": len(result.reports),
        "facts": len(result.facts),
        "metrics": len(metrics),
        "excluded_non_consolidated": excluded,
    }


class HermesCanonicalizer:
    """Transform immutable Hermes source chunks into canonical partitions."""

    def __init__(self, request: HermesAcquisitionRequest):
        self.request = request
        self.root = request.job_root
        self.raw = self.root / "raw"
        self.canonical = self.root / "canonical"
        self.quarantine = self.root / "quarantine"
        self.canonical.mkdir(parents=True, exist_ok=True)
        self.quarantine.mkdir(parents=True, exist_ok=True)
        (self.canonical / ".partial").mkdir(parents=True, exist_ok=True)
        self.security: pd.DataFrame | None = None
        self.instrument_by_security: dict[int, str] = {}
        self.instrument_by_party: dict[int, str] = {}
        self.exchange_by_instrument: dict[str, str] = {}
        self.superseded_aliases: list[dict[str, Any]] = []
        self.special_treatment_windows = 0

    def run(self, resume: bool = False) -> dict[str, Any]:
        state = AcquisitionState(self.root / "state.sqlite")
        try:
            status = state.status()
            if not status["job"] or status["job"]["status"] not in {
                "acquired", "normalizing", "normalized", "failed",
            }:
                raise DataContractError(
                    "source acquisition must complete before normalization"
                )
            state.set_job_status("normalizing")
            self._build_instruments()
            self._build_special_treatment()
            self._build_calendar()
            self._build_universe()
            self._build_market_rules()
            self._build_actions()
            self._build_industry()
            self._build_share_capital()
            market_quality = self._build_market()
            self._build_valuation()
            money_flow_quality = (
                self._build_money_flow()
                if self.request.include_money_flow
                else None
            )
            financial_quality = self._build_financials()
            coverage = self._coverage()
            capabilities: dict[str, Any] = {
                "backtest_ready": False,
                "backtest_block_reason": (
                    "Hermes does not contain a confirmed complete historical "
                    "commission/tax/transfer-fee rule series"
                ),
                "financials": {
                    "schema_version": "1.1",
                    "item_catalog_version": ITEM_CATALOG_VERSION,
                    "calculation_version": FUNDAMENTAL_CALCULATION_VERSION,
                    "pit_validated": True,
                },
                "exchanges": ["XSHG", "XSHE", "XBEI"],
                "rights_issue": True,
            }
            if money_flow_quality is not None:
                capabilities["daily_money_flow"] = money_flow_quality
            manifest = {
                "schema_version": "1.2",
                "source": "Hermes",
                "normalized_at": datetime.now(timezone.utc).isoformat(),
                "market": market_quality,
                "financials": financial_quality,
                "coverage": coverage,
                "excluded_instruments": self.superseded_aliases,
                "special_treatment_windows": self.special_treatment_windows,
                "daily_money_flow": money_flow_quality,
                "capabilities": capabilities,
            }
            path = self.canonical / "_acquisition_manifest.json"
            temporary = self.canonical / ".partial" / "manifest.tmp"
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(temporary, path)
            (self.canonical / "_SUCCESS").touch()
            state.set_job_status("normalized")
            return manifest
        except Exception as exc:
            state.set_job_status(
                "failed", f"{type(exc).__name__}: {str(exc)[:1000]}"
            )
            raise
        finally:
            state.close()

    def _drop_superseded_aliases(self, source: pd.DataFrame) -> pd.DataFrame:
        """Remove retired-ticker records that carry no market data at all.

        A recoded listing (000043 中航善达 -> 001914 招商积余) keeps one
        stable ``SECURITY_ID`` whose whole history stays under that id, and
        leaves behind a second delisted record for the retired ticker. That
        record can never receive a bar, so keeping it makes the calendar
        expect a bar for every trading day of its listed life. Absence of
        market data is the only safe criterion here: a genuine relisting
        (600087 退市长油 -> 601975 招商南油) also shares ``PARTY_ID`` but does
        own its bars, and must be kept.
        """
        traded: set[int] = set()
        for raw_path in sorted((self.raw / "mkt_equd").rglob("*.parquet")):
            frame = pd.read_parquet(raw_path, columns=["SECURITY_ID"])
            traded.update(
                frame["SECURITY_ID"].dropna().astype("int64").unique().tolist()
            )
        if not traded:
            raise DataContractError("Hermes mkt_equd returned no A-share rows")
        untraded = ~source["SECURITY_ID"].astype("int64").isin(traded)
        if not untraded.any():
            self.superseded_aliases = []
            return source
        dropped = source.loc[untraded]
        self.superseded_aliases = [
            {
                "instrument_id": str(row.instrument_id),
                "security_id": int(row.SECURITY_ID),
                "symbol": str(row.TICKER_SYMBOL),
                "name": (
                    None if pd.isna(row.SEC_SHORT_NAME)
                    else str(row.SEC_SHORT_NAME)
                ),
                "list_status": str(row.LIST_STATUS_CD),
                "delist_date": (
                    None if pd.isna(row.DELIST_DATE)
                    else str(row.DELIST_DATE)
                ),
                "party_id": (
                    None if pd.isna(row.PARTY_ID) else int(row.PARTY_ID)
                ),
                "reason": "no_market_data_in_window",
            }
            for row in dropped.itertuples(index=False)
        ]
        return source.loc[~untraded].copy()

    def _build_instruments(self) -> None:
        source = _read(self.raw / "md_security" / "part-all.parquet")
        if source.empty:
            raise DataContractError("Hermes md_security returned no A-share rows")
        source = source[
            source["EXCHANGE_CD"].isin(self.request.exchanges)
            & source["ASSET_CLASS"].eq("E")
            & source["TRANS_CURR_CD"].fillna("CNY").eq("CNY")
        ].copy()
        source["instrument_id"] = _instrument_id(source)
        source.sort_values(
            ["instrument_id", "LIST_DATE", "UPDATE_TIME", "SECURITY_ID"],
            inplace=True,
        )
        source = source.drop_duplicates("instrument_id", keep="last")
        source = self._drop_superseded_aliases(source)
        frame = pd.DataFrame({
            "instrument_id": source["instrument_id"],
            "symbol": source["TICKER_SYMBOL"].astype(str),
            "exchange": source["EXCHANGE_CD"].astype(str),
            "asset_type": "stock",
            "list_date": source["LIST_DATE"],
            "delist_date": source["DELIST_DATE"],
            "lot_size": 100,
            "sell_delay_days": 1,
            "name": source["SEC_SHORT_NAME"],
            "currency": "CNY",
            **_source_fields(source, "md_security"),
        })
        _write(self.canonical, "instruments", "part-000.parquet", frame)
        self.security = source
        self.instrument_by_security = dict(zip(
            source["SECURITY_ID"].astype(int), source["instrument_id"]
        ))
        self.instrument_by_party = dict(zip(
            source["PARTY_ID"].dropna().astype(int),
            source.loc[source["PARTY_ID"].notna(), "instrument_id"],
        ))
        self.exchange_by_instrument = dict(zip(
            source["instrument_id"], source["EXCHANGE_CD"]
        ))

    def _build_special_treatment(self) -> None:
        """Transcribe the vendor's special-treatment state log into windows.

        ``md_security`` carries only today's short name, so a name-based screen
        such as excluding ``ST`` issues would otherwise apply the current label
        to every historical day. ``equ_inst_sstate`` is not a general rename
        history — it records entries into and exits from special treatment,
        each carrying the short name that took effect — which is exactly what a
        point-in-time special-treatment screen needs.

        Every transition becomes one window, including the ones that return an
        issuer to a normal name, so the caller can see what was in force rather
        than inferring it. Deciding which names count as special treatment is a
        research choice and deliberately left to the strategy.

        ``known_at`` is the announcement date: a transition is invisible until
        disclosed even though its window opens on the effective date. Periods
        before an issuer's first recorded transition get no window at all,
        which reads as "not under special treatment" — safe, because the first
        record is overwhelmingly an entry rather than an exit.
        """
        path = self.raw / "equ_inst_sstate" / "part-all.parquet"
        if not path.exists():
            self.special_treatment_windows = 0
            return
        source = _read(path)
        rows: list[dict[str, Any]] = []
        if not source.empty:
            source = source.copy()
            source["instrument_id"] = source["SECURITY_ID"].map(
                lambda value: self.instrument_by_security.get(int(value))
                if pd.notna(value) else None
            )
            source = source[
                source["instrument_id"].notna()
                & source["SEC_SHORT_NAME"].notna()
                & source["EFF_DATE"].notna()
            ]
            source["EFF_DATE"] = pd.to_datetime(source["EFF_DATE"]).dt.date
            source["announced"] = pd.to_datetime(
                source["PUBLISH_DATE"], errors="coerce"
            ).dt.date
            source = source.sort_values(
                ["instrument_id", "EFF_DATE", "ID"], kind="mergesort"
            )
            for instrument, group in source.groupby("instrument_id", sort=True):
                records = group.drop_duplicates("EFF_DATE", keep="last")
                dates = list(records["EFF_DATE"])
                for position, item in enumerate(
                    records.itertuples(index=False)
                ):
                    rows.append({
                        "instrument_id": str(instrument),
                        "name": str(item.SEC_SHORT_NAME),
                        "state_code": (
                            None if pd.isna(item.PARTY_STATE)
                            else int(item.PARTY_STATE)
                        ),
                        "reason_code": (
                            None if pd.isna(item.REASON)
                            else int(item.REASON)
                        ),
                        "effective_from": item.EFF_DATE,
                        # Windows abut: a transition ends the previous state on
                        # the day the next one takes effect.
                        "effective_to": (
                            dates[position + 1]
                            if position + 1 < len(dates) else None
                        ),
                        "known_at": (
                            item.announced
                            if item.announced is not None
                            and pd.notna(item.announced)
                            else item.EFF_DATE
                        ),
                        "source_record_id": f"equ_inst_sstate:{item.ID}",
                        "source_batch_id": self.request.job_id,
                        "source_updated_at": pd.to_datetime(
                            item.UPDATE_TIME, utc=True, errors="coerce"
                        ),
                    })
        frame = pd.DataFrame(
            rows, columns=list(FIELD_SPECS["special_treatment"])
        )
        frame = frame.sort_values(
            ["instrument_id", "effective_from"], kind="mergesort"
        ).reset_index(drop=True)
        _write(
            self.canonical, "special_treatment", "part-000.parquet", frame
        )
        self.special_treatment_windows = len(frame)

    def _build_calendar(self) -> None:
        source = _read(self.raw / "md_trade_cal" / "part-all.parquet")
        source = source[
            source["IS_OPEN"].eq(1)
            & source["EXCHANGE_CD"].isin(self.request.exchanges)
        ].copy()
        frame = pd.DataFrame({
            "trade_date": source["CALENDAR_DATE"],
            "exchange": source["EXCHANGE_CD"],
            **_source_fields(source, "md_trade_cal"),
        })
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        frame = frame[
            (frame["trade_date"] >= self.request.start_date)
            & (frame["trade_date"] <= self.request.end_date)
        ]
        for year, part in frame.groupby(
            pd.to_datetime(frame["trade_date"]).dt.year, sort=True
        ):
            _write(
                self.canonical,
                "trade_calendar",
                f"year={year}/part-000.parquet",
                part,
            )

    def _build_universe(self) -> None:
        assert self.security is not None
        source = self.security
        calendar = pd.read_parquet(self.canonical / "trade_calendar")
        calendar["trade_date"] = pd.to_datetime(
            calendar["trade_date"]
        ).dt.date
        days = {
            exchange: sorted(set(group["trade_date"]))
            for exchange, group in calendar.groupby("exchange")
        }
        effective_to: list[date | None] = []
        for row in source.itertuples(index=False):
            if pd.isna(row.DELIST_DATE):
                effective_to.append(None)
                continue
            delist = pd.Timestamp(row.DELIST_DATE).date()
            earlier = [
                day for day in days.get(str(row.EXCHANGE_CD), ())
                if day < delist
            ]
            effective_to.append(
                earlier[-1] if earlier else delist - timedelta(days=1)
            )
        frame = pd.DataFrame({
            "universe_id": "CN_ALL_A",
            "instrument_id": source["instrument_id"],
            "effective_from": source["LIST_DATE"],
            "effective_to": effective_to,
            "known_at": source["LIST_DATE"],
            **_source_fields(source, "md_security"),
        })
        _write(
            self.canonical,
            "universe_membership",
            "part-000.parquet",
            frame,
        )

    def _build_market_rules(self) -> None:
        rows = []
        for exchange in self.request.exchanges:
            rows.append({
                "rule_id": f"source-missing-{exchange}-stock",
                "exchange": exchange,
                "asset_type": "stock",
                "effective_from": self.request.start_date,
                "effective_to": self.request.end_date,
                "commission_bps": None,
                "minimum_commission": None,
                "sell_tax_bps": None,
                "buy_tax_bps": None,
                "transfer_fee_bps": None,
                "currency": "CNY",
                "source": "source_missing",
                "rule_version": "hermes-search-v1",
                "scenario": False,
            })
        _write(
            self.canonical,
            "market_rules",
            "part-000.parquet",
            pd.DataFrame(rows),
        )

    def _action_row(
        self,
        source: pd.Series,
        table: str,
        kind: str,
        ex_date: Any,
        announced: Any,
        cash: float = 0.0,
        ratio: float = 0.0,
        subscription_price: float | None = None,
        record_date: Any = None,
        pay_date: Any = None,
    ) -> dict[str, Any] | None:
        instrument = self.instrument_by_security.get(int(source["SECURITY_ID"]))
        if not instrument or pd.isna(ex_date) or pd.isna(announced):
            return None
        source_id = str(source["ID"])
        return {
            "event_id": hash_payload([table, source_id, kind]),
            "instrument_id": instrument,
            "event_type": kind,
            "record_date": record_date,
            "ex_date": ex_date,
            "pay_date": pay_date,
            "cash_per_share": cash,
            "share_ratio": ratio,
            "subscription_price": subscription_price,
            "status": "active",
            "announced_at": announced,
            "source_record_id": f"{table}:{source_id}",
            "source_updated_at": pd.to_datetime(
                source.get("UPDATE_TIME"), utc=True, errors="coerce"
            ),
        }

    def _build_actions(self) -> None:
        rows: list[dict[str, Any]] = []
        dividend = _read(self.raw / "equ_div_pit" / "part-all.parquet")
        for _, item in dividend.iterrows():
            # equ_div_pit.PUBLISH_DATE is the board-proposal date and the
            # source stopped populating it: it is empty for 81% of 2019
            # ex-dates and for every ex-date from 2020 on, which silently
            # dropped 99.5% of post-2018 dividends. IM_PUBLISH_DATE is the
            # distribution-implementation announcement, populated for every
            # row and never later than the ex-date, so it gates visibility
            # for the whole history under one consistent definition.
            announced = item.get("IM_PUBLISH_DATE")
            common = (
                item.get("EX_DIV_DATE"),
                announced,
                item.get("RECORD_DATE"),
                item.get("PAY_CASH_DATE"),
            )
            cash = _finite_or_none(item.get("PER_CASH_DIV")) or 0.0
            if cash > 0:
                row = self._action_row(
                    item, "equ_div_pit", "cash_dividend",
                    common[0], common[1], cash=cash,
                    record_date=common[2], pay_date=common[3],
                )
                if row:
                    rows.append(row)
            ratio = (
                (_finite_or_none(item.get("PER_SHARE_DIV_RATIO")) or 0.0)
                + (_finite_or_none(item.get("PER_SHARE_TRANS_RATIO")) or 0.0)
            )
            if ratio > 0:
                row = self._action_row(
                    item, "equ_div_pit", "bonus",
                    common[0], common[1], ratio=ratio,
                    record_date=common[2],
                    pay_date=item.get("BONUS_SHARE_LIST_DATE"),
                )
                if row:
                    rows.append(row)
        splits = _read(self.raw / "equ_splits" / "part-all.parquet")
        for _, item in splits.iterrows():
            ratio = _finite_or_none(item.get("SPLITS_RATIO")) or 0.0
            if ratio > 0:
                row = self._action_row(
                    item, "equ_splits",
                    "split" if ratio >= 1 else "merge",
                    (
                        item.get("RE_TRADE_DATE")
                        if pd.notna(item.get("RE_TRADE_DATE"))
                        else item.get("SPLITS_BASE_DATE")
                    ),
                    item.get("PUBLISH_DATE"), ratio=ratio,
                    record_date=item.get("SPLITS_BASE_DATE"),
                )
                if row:
                    rows.append(row)
        allot = _read(self.raw / "equ_allot" / "part-all.parquet")
        if not allot.empty and "IS_ALLOTMENT" in allot:
            allot = allot[allot["IS_ALLOTMENT"].fillna(0).astype(int).eq(1)]
        for _, item in allot.iterrows():
            ratio = _finite_or_none(item.get("ALLOTMENT_RATIO")) or 0.0
            price = _finite_or_none(item.get("ALLOTMENT_PRICE"))
            if ratio > 0 and price is not None:
                row = self._action_row(
                    item, "equ_allot", "rights_issue",
                    item.get("EX_RIGHTS_DATE"), item.get("PUBLISH_DATE"),
                    ratio=ratio, subscription_price=price,
                    record_date=item.get("RECORD_DATE"),
                    pay_date=item.get("PAY_BEGIN_DATE"),
                )
                if row:
                    rows.append(row)
        columns = list(FIELD_SPECS["corporate_actions"])
        _write(
            self.canonical,
            "corporate_actions",
            "part-000.parquet",
            pd.DataFrame(rows, columns=columns),
        )

    def _build_industry(self) -> None:
        assignments = _read(self.raw / "md_inst_type" / "part-all.parquet")
        types = _read(self.raw / "md_type" / "part-all.parquet")
        if assignments.empty or types.empty:
            raise DataContractError("Hermes industry source tables are empty")
        shenwan = types[
            types["INDUSTRY"].astype(str).str.contains(
                "申万|SW", case=False, regex=True, na=False
            )
        ].copy()
        if shenwan.empty:
            raise DataContractError(
                "unable to identify Shenwan level-1 codes in md_type"
            )
        by_id = {
            str(row.TYPE_ID): row
            for row in shenwan.itertuples(index=False)
        }
        mappings: list[dict[str, Any]] = []
        for source_type, row in by_id.items():
            current = row
            visited = {source_type}
            while (
                pd.notna(current.INDUSTRY_LEVEL)
                and int(current.INDUSTRY_LEVEL) > 1
            ):
                parent = str(current.PARENT_ID)
                if parent in visited or parent not in by_id:
                    current = None
                    break
                visited.add(parent)
                current = by_id[parent]
            if (
                current is not None
                and pd.notna(current.INDUSTRY_LEVEL)
                and int(current.INDUSTRY_LEVEL) == 1
            ):
                mappings.append({
                    "TYPE_ID": source_type,
                    "L1_SYMBOL": current.TYPE_SYMBOL,
                    "L1_TYPE_ID": current.TYPE_ID,
                    "L1_NAME": current.TYPE_NAME,
                    "INDUSTRY_VERSION": current.INDUSTRY_VERSION,
                })
        descendants = pd.DataFrame(mappings)
        if descendants.empty:
            raise DataContractError(
                "unable to resolve Shenwan descendants to level-1 codes"
            )
        merged = assignments.merge(
            descendants,
            on="TYPE_ID",
            how="inner",
        )
        merged["instrument_id"] = merged["PARTY_ID"].map(
            self.instrument_by_party
        )
        merged = merged[merged["instrument_id"].notna()].copy()
        merged = merged[
            merged["INTO_DATE"].notna()
            & (
                merged["OUT_DATE"].isna()
                | (
                    pd.to_datetime(merged["OUT_DATE"]).dt.date
                    >= self.request.start_date
                )
            )
        ]
        frame = pd.DataFrame({
            "classification": (
                "SW_L1:" + merged["INDUSTRY_VERSION"].fillna("unknown").astype(str)
            ),
            "industry_id": merged["L1_SYMBOL"].fillna(
                merged["L1_TYPE_ID"]
            ).astype(str),
            "instrument_id": merged["instrument_id"],
            "effective_from": merged["INTO_DATE"],
            "effective_to": merged["OUT_DATE"],
            "known_at": merged["INTO_DATE"],
            **_source_fields(merged, "md_inst_type"),
        })
        frame.sort_values(
            ["classification", "instrument_id", "effective_from"],
            inplace=True,
        )
        frame = frame.drop_duplicates(
            ["classification", "instrument_id", "effective_from"],
            keep="last",
        )
        _write(
            self.canonical,
            "industry_membership",
            "part-000.parquet",
            frame,
        )

    def _build_share_capital(self) -> None:
        source = _read(self.raw / "equ_shares_change" / "part-all.parquet")
        free = _read(self.raw / "equ_free_shares" / "part-all.parquet")
        source["CHANGE_DATE"] = pd.to_datetime(source["CHANGE_DATE"])
        free["CHANGE_DATE"] = pd.to_datetime(free["CHANGE_DATE"])
        free = free.sort_values(["PARTY_ID", "CHANGE_DATE", "ID"])
        if not source.empty and not free.empty:
            source = pd.merge_asof(
                source.sort_values(["CHANGE_DATE", "PARTY_ID"]),
                free[["PARTY_ID", "CHANGE_DATE", "FREE_SHARES"]].sort_values(
                    ["CHANGE_DATE", "PARTY_ID"]
                ),
                on="CHANGE_DATE",
                by="PARTY_ID",
                direction="backward",
            )
        else:
            source["FREE_SHARES"] = np.nan
        source["instrument_id"] = source["PARTY_ID"].map(
            self.instrument_by_party
        )
        source = source[source["instrument_id"].notna()].copy()
        published = source["PUBLISH_DATE"].where(
            source["PUBLISH_DATE"].notna(), source["CHANGE_DATE"]
        )
        frame = pd.DataFrame({
            "capital_event_id": "equ_shares_change:" + source["ID"].astype(str),
            "instrument_id": source["instrument_id"],
            "effective_from": source["CHANGE_DATE"],
            "announced_at": published,
            "available_at": published,
            "change_reason_code": source["CHANGE_TYPE"].astype("string"),
            "change_reason": None,
            "total_shares": source["TOTAL_SHARES"],
            "nontradable_shares": source["NONF_SHARES"],
            "restricted_shares": source["REST_SHARES"],
            "tradable_shares": source["FLOAT_SHARES"],
            "a_shares": source["FLOAT_A"],
            "b_shares": source["FLOAT_B"],
            "h_shares": source["FLOAT_H"],
            **_source_fields(source, "equ_shares_change"),
        })
        _write(
            self.canonical,
            "share_capital",
            "part-000.parquet",
            frame,
        )

    def _halt_keys(self) -> set[tuple[date, str]]:
        halt = _read(self.raw / "md_sec_halt" / "part-all.parquet")
        calendar = pd.read_parquet(self.canonical / "trade_calendar")
        calendar["trade_date"] = pd.to_datetime(
            calendar["trade_date"]
        ).dt.date
        days = {
            exchange: sorted(set(group["trade_date"]))
            for exchange, group in calendar.groupby("exchange")
        }
        output: set[tuple[date, str]] = set()
        for row in halt.itertuples(index=False):
            instrument = self.instrument_by_security.get(int(row.SECURITY_ID))
            if not instrument or pd.isna(row.HALT_BEGIN_TIME):
                continue
            begin = pd.Timestamp(row.HALT_BEGIN_TIME).date()
            end = (
                pd.Timestamp(row.RESUMP_BEGIN_TIME).date()
                if pd.notna(row.RESUMP_BEGIN_TIME)
                else self.request.end_date + timedelta(days=1)
            )
            exchange = self.exchange_by_instrument[instrument]
            for day in days.get(exchange, ()):
                if begin <= day < end:
                    output.add((day, instrument))
        return output

    def _build_market(self) -> dict[str, Any]:
        halt_keys = self._halt_keys()
        instruments = pd.read_parquet(self.canonical / "instruments")
        instruments["list_date"] = pd.to_datetime(
            instruments["list_date"]
        ).dt.date
        instruments["delist_date"] = pd.to_datetime(
            instruments["delist_date"], errors="coerce"
        ).dt.date
        calendar_frame = pd.read_parquet(
            self.canonical / "trade_calendar"
        )
        calendar_frame["trade_date"] = pd.to_datetime(
            calendar_frame["trade_date"]
        ).dt.date
        calendar = {
            exchange: sorted(set(group["trade_date"]))
            for exchange, group in calendar_frame.groupby("exchange")
        }
        previous_close: dict[str, float] = {}
        previous_factor: dict[str, float] = {}
        quarantined: list[pd.DataFrame] = []
        rows = 0
        paused_rows = 0
        for raw_path in sorted((self.raw / "mkt_equd").rglob("*.parquet")):
            relative = raw_path.relative_to(self.raw / "mkt_equd")
            source = _read(raw_path)
            limit = _read(self.raw / "mkt_limit" / relative)
            adjusted = _read(self.raw / "mkt_equd_adj_af" / relative)
            source["instrument_id"] = source["SECURITY_ID"].map(
                self.instrument_by_security
            )
            source["trade_date"] = pd.to_datetime(
                source["TRADE_DATE"]
            ).dt.date
            source = source[source["instrument_id"].notna()].copy()
            if not limit.empty:
                limit["trade_date"] = pd.to_datetime(
                    limit["TRADE_DATE"]
                ).dt.date
                limit["instrument_id"] = limit["SECURITY_ID"].map(
                    self.instrument_by_security
                )
                limit = limit[[
                    "trade_date", "instrument_id",
                    "LIMIT_UP_PRICE", "LIMIT_DOWN_PRICE",
                ]].drop_duplicates(["trade_date", "instrument_id"], keep="last")
                source = source.merge(
                    limit,
                    on=["trade_date", "instrument_id"],
                    how="left",
                )
                limit_lookup = {
                    (row.trade_date, str(row.instrument_id)): (
                        _finite_or_none(row.LIMIT_UP_PRICE),
                        _finite_or_none(row.LIMIT_DOWN_PRICE),
                    )
                    for row in limit.itertuples(index=False)
                }
            else:
                source["LIMIT_UP_PRICE"] = np.nan
                source["LIMIT_DOWN_PRICE"] = np.nan
                limit_lookup = {}
            if source.empty:
                continue
            month_start = min(source["trade_date"])
            month_end = max(source["trade_date"])
            existing_keys = set(zip(
                source["trade_date"], source["instrument_id"].astype(str)
            ))
            source_closes = {
                (row.trade_date, str(row.instrument_id)): _finite_or_none(
                    row.CLOSE_PRICE
                )
                for row in source.itertuples(index=False)
            }
            missing_rows: list[dict[str, Any]] = []
            unexplained: list[dict[str, Any]] = []
            for instrument in instruments.itertuples(index=False):
                code = str(instrument.instrument_id)
                dates = calendar.get(str(instrument.exchange), ())
                last_price = previous_close.get(code)
                for day in dates:
                    if day < month_start or day > month_end:
                        continue
                    if day < instrument.list_date:
                        continue
                    if (
                        pd.notna(instrument.delist_date)
                        and day >= instrument.delist_date
                    ):
                        continue
                    key = (day, code)
                    if key in existing_keys:
                        current_close = source_closes.get(key)
                        if current_close is not None:
                            last_price = current_close
                        continue
                    if key not in halt_keys:
                        unexplained.append({
                            "trade_date": day,
                            "instrument_id": code,
                            "quarantine_reason": "unexplained_missing_bar",
                        })
                        continue
                    price = last_price
                    limits = limit_lookup.get(key, (None, None))
                    if price is None:
                        unexplained.append({
                            "trade_date": day,
                            "instrument_id": code,
                            "quarantine_reason": (
                                "halt_fill_missing_previous_close"
                            ),
                        })
                        continue
                    missing_rows.append({
                        "ID": -len(missing_rows) - 1,
                        "SECURITY_ID": None,
                        "UPDATE_TIME": None,
                        "trade_date": day,
                        "instrument_id": code,
                        "OPEN_PRICE": price,
                        "HIGHEST_PRICE": price,
                        "LOWEST_PRICE": price,
                        "CLOSE_PRICE": price,
                        "PRE_CLOSE_PRICE": price,
                        "TURNOVER_VOL": 0,
                        "TURNOVER_VALUE": 0.0,
                        "LIMIT_UP_PRICE": limits[0],
                        "LIMIT_DOWN_PRICE": limits[1],
                        "paused": True,
                    })
                    last_price = price
            if missing_rows:
                source = pd.concat(
                    [source, pd.DataFrame(missing_rows)],
                    ignore_index=True,
                    sort=False,
                )
            if unexplained:
                quarantined.append(pd.DataFrame(unexplained))
            source["paused"] = [
                (day, code) in halt_keys
                for day, code in zip(
                    source["trade_date"], source["instrument_id"]
                )
            ]
            price_columns = {
                "OPEN_PRICE": "open",
                "HIGHEST_PRICE": "high",
                "LOWEST_PRICE": "low",
                "CLOSE_PRICE": "close",
                "PRE_CLOSE_PRICE": "pre_close",
            }
            for source_name, canonical_name in price_columns.items():
                source[canonical_name] = pd.to_numeric(
                    source[source_name], errors="coerce"
                )
            bad_price = source[list(price_columns.values())].isna().any(axis=1)
            fillable = bad_price & source["paused"]
            for index in source.index[fillable]:
                code = str(source.at[index, "instrument_id"])
                fallback = _finite_or_none(source.at[index, "PRE_CLOSE_PRICE"])
                if fallback is None:
                    fallback = previous_close.get(code)
                if fallback is not None:
                    for column in price_columns.values():
                        source.at[index, column] = fallback
                    source.at[index, "TURNOVER_VOL"] = 0
                    source.at[index, "TURNOVER_VALUE"] = 0.0
            bad_price = source[list(price_columns.values())].isna().any(axis=1)
            # A halted bar has no price band: the source publishes no limit
            # for it, and none is meaningful because the day is untradable.
            # The execution layer rejects halted bars before it ever consults
            # limit_up/limit_down, so a null band there changes no outcome.
            # On a tradable day the band is still mandatory.
            missing_band = (
                source["LIMIT_UP_PRICE"].isna()
                | source["LIMIT_DOWN_PRICE"].isna()
            )
            core_missing = bad_price | (
                missing_band & ~source["paused"].astype(bool)
            )
            if core_missing.any():
                rejected = source.loc[core_missing].copy()
                rejected["quarantine_reason"] = "missing_core_market_field"
                quarantined.append(rejected)
                source = source.loc[~core_missing].copy()
            source.sort_values(
                ["trade_date", "instrument_id"], inplace=True
            )
            for item in source.itertuples(index=False):
                previous_close[str(item.instrument_id)] = float(item.close)
            frame = pd.DataFrame({
                "trade_date": source["trade_date"],
                "instrument_id": source["instrument_id"],
                "open": source["open"],
                "high": source["high"],
                "low": source["low"],
                "close": source["close"],
                "pre_close": source["pre_close"],
                "volume": pd.to_numeric(
                    source["TURNOVER_VOL"], errors="coerce"
                ).round().astype("int64"),
                "amount": source["TURNOVER_VALUE"],
                "paused": source["paused"],
                "limit_up": source["LIMIT_UP_PRICE"],
                "limit_down": source["LIMIT_DOWN_PRICE"],
                **_source_fields(source, "mkt_equd"),
            })
            output_partition = relative.as_posix()
            _write(
                self.canonical, "daily_raw", output_partition, frame
            )
            post = self._post_adjusted(
                adjusted, frame, previous_factor
            )
            _write(
                self.canonical,
                "daily_post_adjusted",
                output_partition,
                post,
            )
            rows += len(frame)
            paused_rows += int(frame["paused"].sum())
        quarantine_rows = sum(len(frame) for frame in quarantined)
        if quarantined:
            _write(
                self.quarantine,
                "market",
                "missing-core.parquet",
                pd.concat(quarantined, ignore_index=True),
                normalize=False,
            )
        tolerance = max(100, int((rows + quarantine_rows) * 0.00001))
        if quarantine_rows > tolerance:
            raise DataContractError(
                f"market core-field quarantine {quarantine_rows} exceeds "
                f"tolerance {tolerance}"
            )
        return {
            "rows": rows,
            "paused_rows": paused_rows,
            "quarantined_rows": quarantine_rows,
            "quarantine_tolerance": tolerance,
        }

    def _post_adjusted(
        self,
        adjusted: pd.DataFrame,
        raw: pd.DataFrame,
        previous_factor: dict[str, float],
    ) -> pd.DataFrame:
        if not adjusted.empty:
            adjusted["trade_date"] = pd.to_datetime(
                adjusted["TRADE_DATE"]
            ).dt.date
            adjusted["instrument_id"] = adjusted["SECURITY_ID"].map(
                self.instrument_by_security
            )
            adjusted = adjusted[
                adjusted["instrument_id"].notna()
            ].drop_duplicates(
                ["trade_date", "instrument_id"], keep="last"
            )
            adjusted = adjusted.set_index(["trade_date", "instrument_id"])
        first_factor: dict[str, float] = {}
        if not adjusted.empty:
            reset = adjusted.reset_index().sort_values(
                ["instrument_id", "trade_date"]
            )
            for vendor in reset.itertuples(index=False):
                factor = _finite_or_none(vendor.ACCUM_ADJ_FACTOR_2)
                if factor and factor > 0:
                    first_factor.setdefault(
                        str(vendor.instrument_id),
                        factor,
                    )
        rows = []
        for item in raw.sort_values(
            ["instrument_id", "trade_date"]
        ).itertuples(index=False):
            key = (item.trade_date, item.instrument_id)
            vendor = adjusted.loc[key] if not adjusted.empty and key in adjusted.index else None
            if isinstance(vendor, pd.DataFrame):
                vendor = vendor.iloc[-1]
            factor = None
            values = None
            if vendor is not None:
                factor = _finite_or_none(
                    vendor.get("ACCUM_ADJ_FACTOR_2")
                )
            if factor is None:
                factor = previous_factor.get(str(item.instrument_id))
            if factor is None:
                factor = first_factor.get(str(item.instrument_id))
            if factor is None or factor <= 0:
                raise DataContractError(
                    f"missing adjustment factor for {key}"
                )
            previous_factor[str(item.instrument_id)] = factor
            values = {
                "open_post": float(item.open) * factor,
                "high_post": float(item.high) * factor,
                "low_post": float(item.low) * factor,
                "close_post": float(item.close) * factor,
                "pre_close_post": float(item.pre_close) * factor,
            }
            rows.append({
                "trade_date": item.trade_date,
                "instrument_id": item.instrument_id,
                **values,
                "adjustment_factor": factor,
                "factor_source": "vendor",
                "adjustment_version": "hermes-af-1.0",
            })
        return pd.DataFrame(
            rows, columns=list(FIELD_SPECS["daily_post_adjusted"])
        )

    def _build_valuation(self) -> None:
        capital = pd.read_parquet(self.canonical / "share_capital")
        capital["instrument_id"] = capital["instrument_id"].astype(str)
        capital["effective_from"] = pd.to_datetime(
            capital["effective_from"]
        )
        capital = capital[[
            "instrument_id", "effective_from", "total_shares",
            "tradable_shares", "a_shares",
        ]].sort_values(["effective_from", "instrument_id"])
        free = _read(self.raw / "equ_free_shares" / "part-all.parquet")
        free["instrument_id"] = free["PARTY_ID"].map(
            self.instrument_by_party
        )
        free = free[free["instrument_id"].notna()].copy()
        free["instrument_id"] = free["instrument_id"].astype(str)
        free["free_effective_from"] = pd.to_datetime(free["CHANGE_DATE"])
        free = free[[
            "instrument_id", "free_effective_from", "FREE_SHARES",
        ]].sort_values(["free_effective_from", "instrument_id"])
        for eval_path in sorted(
            (self.raw / "mkt_equd_eval").rglob("*.parquet")
        ):
            relative = eval_path.relative_to(self.raw / "mkt_equd_eval")
            base = _read(eval_path)
            newer = _read(self.raw / "mkt_equd_eval_new" / relative)
            dividend = _read(self.raw / "mkt_div_yield" / relative)
            market = _read(self.raw / "mkt_equd" / relative)
            for frame in (base, newer, dividend, market):
                if frame.empty:
                    continue
                frame["trade_date"] = pd.to_datetime(
                    frame["TRADE_DATE"]
                ).dt.date
                frame["instrument_id"] = frame["SECURITY_ID"].map(
                    self.instrument_by_security
                )
            keys = ["trade_date", "instrument_id"]
            columns = [
                "trade_date", "instrument_id", "PE_T", "PB", "PS_T",
                "PCF_T", "PCF_OT", "MARKET_VALUE", "NEG_MARKET_VALUE",
                "ID", "UPDATE_TIME",
            ]
            value = base[[item for item in columns if item in base]].copy()
            if not newer.empty:
                value = value.merge(
                    newer[keys + ["PE_LYR", "FREE_MARKET_VALUE"]],
                    on=keys, how="left",
                )
            if not dividend.empty:
                # DIV_RATE_L12M is the trailing-twelve-month yield, populated
                # for every payer all year. DIV_RATE_TTM rolls off the last
                # completed report instead and collapses between January and
                # April: only about 4 issuers per thousand row-days clear 3%
                # there, against roughly 70 for L12M. A trailing-yield screen
                # driven by TTM would therefore find nothing for a third of
                # every year.
                value = value.merge(
                    dividend[keys + ["DIV_RATE_L12M"]],
                    on=keys, how="left",
                )
            if not market.empty:
                value = value.merge(
                    market[keys + ["TURNOVER_RATE", "CLOSE_PRICE"]],
                    on=keys, how="left",
                )
            value = value[value["instrument_id"].notna()].copy()
            value["instrument_id"] = value["instrument_id"].astype(str)
            value["trade_date"] = pd.to_datetime(value["trade_date"])
            value = pd.merge_asof(
                value.sort_values(["trade_date", "instrument_id"]),
                capital.sort_values(["effective_from", "instrument_id"]),
                left_on="trade_date",
                right_on="effective_from",
                by="instrument_id",
                direction="backward",
            )
            value = pd.merge_asof(
                value.sort_values(["trade_date", "instrument_id"]),
                free.sort_values(
                    ["free_effective_from", "instrument_id"]
                ),
                left_on="trade_date",
                right_on="free_effective_from",
                by="instrument_id",
                direction="backward",
            )
            frame = pd.DataFrame({
                "trade_date": value["trade_date"].dt.date,
                "instrument_id": value["instrument_id"],
                "pe_ttm": value.get("PE_T"),
                "pe_lyr": value.get("PE_LYR"),
                "pb": value.get("PB"),
                "ps_ttm": value.get("PS_T"),
                "pcf_ttm": value.get("PCF_T"),
                "pcf_operating_ttm": value.get("PCF_OT"),
                # Source reports the yield in percent; the contract stores a
                # ratio.
                "dividend_yield": pd.to_numeric(
                    value.get("DIV_RATE_L12M"), errors="coerce"
                ) / 100.0,
                "turnover_rate": pd.to_numeric(
                    value.get("TURNOVER_RATE"), errors="coerce"
                ),
                "total_shares": value.get("total_shares"),
                "market_cap": value.get("MARKET_VALUE"),
                "circulating_shares": value.get("tradable_shares"),
                "circulating_market_cap": value.get("NEG_MARKET_VALUE"),
                "free_float_shares": value.get("FREE_SHARES"),
                "free_float_market_cap": value.get("FREE_MARKET_VALUE"),
                "a_shares": value.get("a_shares"),
                "a_market_cap": (
                    pd.to_numeric(value.get("a_shares"), errors="coerce")
                    * pd.to_numeric(value.get("CLOSE_PRICE"), errors="coerce")
                ),
                "available_at": value["trade_date"].dt.date,
                "source_record_id": "mkt_equd_eval:" + value["ID"].astype(str),
                "source_updated_at": pd.to_datetime(
                    value["UPDATE_TIME"], utc=True, errors="coerce"
                ),
            })
            _write(
                self.canonical,
                "daily_valuation",
                relative.as_posix(),
                frame,
            )

    def _build_money_flow(self) -> dict[str, Any]:
        source_root = self.raw / "mkt_equ_mf_new"
        required_source = {
            "TRADE_DATE", "SECURITY_ID", "UPDATE_TIME",
            *MONEY_FLOW_SOURCE_FIELDS,
        }
        total_rows = 0
        minimum_date: date | None = None
        maximum_date: date | None = None
        source_paths = sorted(source_root.rglob("*.parquet"))
        if not source_paths:
            raise DataContractError(
                "money-flow acquisition is enabled but mkt_equ_mf_new "
                "has no raw partitions"
            )
        for source_path in source_paths:
            relative = source_path.relative_to(source_root)
            source = _read(source_path)
            if source.empty:
                frame = pd.DataFrame(columns=list(FIELD_SPECS["daily_money_flow"]))
                _write(
                    self.canonical,
                    "daily_money_flow",
                    relative.as_posix(),
                    frame,
                )
                continue
            missing = required_source - set(source)
            if missing:
                raise DataContractError(
                    "mkt_equ_mf_new is missing required source columns: "
                    f"{sorted(missing)}"
                )

            trade_date = pd.to_datetime(
                source["TRADE_DATE"], errors="coerce"
            ).dt.date
            if trade_date.isna().any():
                raise DataContractError(
                    "mkt_equ_mf_new.TRADE_DATE contains invalid values"
                )
            instrument_id = source["SECURITY_ID"].map(
                self.instrument_by_security
            )
            if instrument_id.isna().any():
                unknown = source.loc[
                    instrument_id.isna(), "SECURITY_ID"
                ].astype(str).unique()[:10]
                raise DataContractError(
                    "mkt_equ_mf_new references unknown SECURITY_ID values: "
                    f"{sorted(unknown)}"
                )
            updated_at = pd.to_datetime(
                source["UPDATE_TIME"], utc=True, errors="coerce"
            )
            invalid_update = source["UPDATE_TIME"].notna() & updated_at.isna()
            if invalid_update.any():
                raise DataContractError(
                    "mkt_equ_mf_new.UPDATE_TIME contains invalid values"
                )
            local_update_date = updated_at.dt.tz_convert(
                "Asia/Shanghai"
            ).dt.date
            available_at = pd.Series(
                [
                    max(day, revised) if pd.notna(revised) else day
                    for day, revised in zip(trade_date, local_update_date)
                ],
                index=source.index,
                dtype=object,
            )

            values: dict[str, pd.Series] = {}
            for source_name, canonical_name in MONEY_FLOW_SOURCE_FIELDS.items():
                numeric = pd.to_numeric(source[source_name], errors="coerce")
                invalid = source[source_name].notna() & numeric.isna()
                if invalid.any():
                    raise DataContractError(
                        f"mkt_equ_mf_new.{source_name} contains non-numeric values"
                    )
                values[canonical_name] = numeric.astype(float)

            if "ID" in source:
                record_id = (
                    "mkt_equ_mf_new:" + source["ID"].astype(str)
                )
            else:
                record_id = (
                    "mkt_equ_mf_new:"
                    + source["SECURITY_ID"].astype(str)
                    + ":"
                    + trade_date.astype(str)
                )
            frame = pd.DataFrame({
                "trade_date": trade_date,
                "instrument_id": instrument_id.astype(str),
                **values,
                "available_at": available_at,
                "source_record_id": record_id,
                "source_batch_id": self.request.job_id,
                "source_updated_at": updated_at,
            })
            _write(
                self.canonical,
                "daily_money_flow",
                relative.as_posix(),
                frame,
            )
            total_rows += len(frame)
            part_min = min(frame["trade_date"])
            part_max = max(frame["trade_date"])
            minimum_date = (
                part_min if minimum_date is None else min(minimum_date, part_min)
            )
            maximum_date = (
                part_max if maximum_date is None else max(maximum_date, part_max)
            )
        return {
            "schema_version": "1",
            "source_table": "mkt_equ_mf_new",
            "mapper_version": MONEY_FLOW_MAPPER_VERSION,
            "unit": "CNY",
            "pit_rule": (
                "available_at=max(trade_date,"
                "source_updated_at_Asia/Shanghai_date)"
            ),
            "rows": total_rows,
            "start_date": minimum_date.isoformat() if minimum_date else None,
            "end_date": maximum_date.isoformat() if maximum_date else None,
            "fields": sorted(FIELD_SPECS["daily_money_flow"]),
        }

    def _build_financials(self) -> dict[str, Any]:
        trade_days = sorted(set(
            pd.to_datetime(
                pd.read_parquet(
                    self.canonical / "trade_calendar"
                )["trade_date"]
            ).dt.date
        ))
        groups = sorted(
            path.name
            for path in (self.raw / "vw_fdmt_bs_new").glob("group-*.parquet")
        )
        totals = {
            "reports": 0,
            "facts": 0,
            "metrics": 0,
            "excluded_non_consolidated": 0,
        }
        for variable in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ[variable] = str(
                self.request.limits.compute_threads_per_process
            )
        workers = min(
            self.request.limits.max_compute_processes,
            max(1, self.request.limits.max_logical_cpus // 2),
            max(1, len(groups)),
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _financial_group_worker,
                    str(self.raw),
                    str(self.canonical),
                    filename,
                    trade_days,
                    self.request.job_id,
                )
                for filename in groups
            ]
            for future in as_completed(futures):
                result = future.result()
                for key in totals:
                    totals[key] += result[key]
        return {
            **totals,
            "compute_processes": workers,
            "threads_per_process": (
                self.request.limits.compute_threads_per_process
            ),
            "merged_flag_mapping": {"1": "consolidated", "2": "parent"},
            "warmup_start": self.request.financial_warmup_start.isoformat(),
        }

    def _coverage(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for table_path in sorted(
            path for path in self.canonical.iterdir()
            if path.is_dir() and path.name != ".partial"
        ):
            dataset = pads.dataset(table_path, format="parquet", partitioning="hive")
            rows = dataset.count_rows()
            nulls: dict[str, int] = {}
            for name, spec in FIELD_SPECS[table_path.name].items():
                if spec.nullable and name in dataset.schema.names:
                    count = 0
                    scanner = dataset.scanner(columns=[name], batch_size=262_144)
                    for batch in scanner.to_batches():
                        count += batch.column(0).null_count
                    nulls[name] = count
            output[table_path.name] = {
                "rows": rows,
                "nullable_field_coverage": {
                    name: (
                        1.0 - missing / rows if rows else None
                    )
                    for name, missing in nulls.items()
                },
            }
        return output


class HermesAcquisitionPublisher:
    """Atomically publish validated canonical acquisition files without loading them."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root).expanduser().resolve()
        self.datasets_root = self.data_root / "datasets"
        self.datasets_root.mkdir(parents=True, exist_ok=True)

    def publish(self, job_id: str, dataset_id: str) -> DataSnapshot:
        acquisition = self.data_root / "acquisitions" / job_id
        canonical = acquisition / "canonical"
        if not (canonical / "_SUCCESS").exists():
            raise DataContractError(
                f"acquisition is not normalized: {job_id}"
            )
        final = self.datasets_root / dataset_id
        if final.exists():
            raise DataContractError(
                f"immutable dataset already exists: {dataset_id}"
            )
        metadata = json.loads(
            (canonical / "_acquisition_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        staging = self.datasets_root / f".{dataset_id}.publishing"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            files = []
            tables = []
            for table_path in sorted(
                path for path in canonical.iterdir()
                if path.is_dir() and path.name != ".partial"
            ):
                name = table_path.name
                if name not in FIELD_SPECS:
                    raise DataContractError(
                        f"unregistered canonical table directory: {name}"
                    )
                table_files = []
                rows = 0
                schemas = set()
                for source in sorted(table_path.rglob("*.parquet")):
                    relative = source.relative_to(canonical)
                    destination = staging / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, destination)
                    parquet = pq.ParquetFile(source)
                    rows += parquet.metadata.num_rows
                    schemas.add(str(parquet.schema_arrow))
                    record = {
                        "path": relative.as_posix(),
                        "table": name,
                        "rows": parquet.metadata.num_rows,
                        "size": source.stat().st_size,
                        "sha256": hash_file(source),
                    }
                    files.append(record)
                    table_files.append(record["path"])
                dataset = pads.dataset(
                    table_path, format="parquet", partitioning="hive"
                )
                actual = set(dataset.schema.names)
                unknown_columns = actual - set(FIELD_SPECS[name])
                missing_columns = set(REQUIRED_COLUMNS[name]) - actual
                if unknown_columns or missing_columns:
                    raise DataContractError(
                        f"{name} has incompatible partition schemas; "
                        f"missing={sorted(missing_columns)}, "
                        f"unknown={sorted(unknown_columns)}"
                    )
                tables.append({
                    "name": name,
                    "rows": rows,
                    "schema_hash": hash_payload(sorted(schemas)),
                    "content_hash": hash_payload([
                        item["sha256"] for item in files
                        if item["table"] == name
                    ]),
                    "files": tuple(table_files),
                })
            required = {
                "instruments", "trade_calendar", "daily_raw",
                "daily_post_adjusted", "corporate_actions",
                "universe_membership", "industry_membership",
                "market_rules", *FINANCIAL_TABLES,
            }
            if "daily_money_flow" in metadata.get("capabilities", {}):
                required.add("daily_money_flow")
            missing = required - {table["name"] for table in tables}
            if missing:
                raise DataContractError(
                    f"normalized acquisition is incomplete: {sorted(missing)}"
                )
            table_lookup = {table["name"]: table for table in tables}
            if (
                table_lookup["daily_raw"]["rows"]
                != table_lookup["daily_post_adjusted"]["rows"]
            ):
                raise DataContractError(
                    "raw and post-adjusted row counts differ"
                )
            fingerprint = hash_payload({
                "schema_version": "1.2",
                "tables": tables,
                "source_schema": hash_file(
                    acquisition / "source_schema.json"
                ),
                "capabilities": metadata["capabilities"],
            })
            manifest = {
                "manifest_schema_version": SNAPSHOT_SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "schema_version": "1.2",
                "as_of_date": self._request_end(acquisition),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "fingerprint": fingerprint,
                "adjustment": {
                    "materialized": True,
                    "method": "vendor_post_adjusted",
                    "algorithm_version": "hermes-af-1.0",
                    "anchor": "Hermes ACCUM_ADJ_FACTOR_2",
                    "factor_source": "vendor",
                    "event_adjustments": metadata["coverage"][
                        "corporate_actions"
                    ]["rows"],
                },
                "files": files,
                "tables": tables,
                "capabilities": metadata["capabilities"],
                "lineage": {
                    "adapter": "HermesDataAdapter",
                    "acquisition_job_id": job_id,
                    "source_schema_hash": hash_file(
                        acquisition / "source_schema.json"
                    ),
                    "request_hash": self._request_hash(acquisition),
                    "source_watermark": self._source_watermark(acquisition),
                    "credentials_persisted": False,
                    "daily_money_flow": metadata.get("daily_money_flow"),
                    "normalization": metadata,
                },
                "quality": {
                    "status": "passed",
                    "streaming_validation": True,
                    "raw_adjusted_key_match": True,
                    "coverage": metadata["coverage"],
                    "quarantine": metadata["market"],
                },
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(staging, final)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return DataSnapshot(final)

    @staticmethod
    def _state_job(acquisition: Path) -> sqlite3.Row:
        connection = sqlite3.connect(acquisition / "state.sqlite")
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute("SELECT * FROM job").fetchone()
            if row is None:
                raise DataContractError("acquisition state has no job record")
            return row
        finally:
            connection.close()

    def _request_end(self, acquisition: Path) -> str:
        request = json.loads(self._state_job(acquisition)["request_json"])
        return str(request["end_date"])

    def _request_hash(self, acquisition: Path) -> str:
        return hash_payload(
            json.loads(self._state_job(acquisition)["request_json"])
        )

    def _source_watermark(self, acquisition: Path) -> str:
        return str(self._state_job(acquisition)["source_watermark"])
