from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from zyquant.core.exceptions import DataContractError
from zyquant.core.hashing import hash_payload

ITEM_CATALOG_VERSION = "1.1"
FUNDAMENTAL_CALCULATION_VERSION = "1.1"

STATEMENT_METADATA = {
    "id", "company_id", "company_name", "code", "a_code", "b_code", "h_code",
    "pub_date", "start_date", "end_date", "report_date", "report_type",
    "source_id", "source",
}
PER_SHARE_ITEMS = {"eps", "basic_eps", "diluted_eps"}
RATIO_ITEMS = {"debt_to_capital"}


@dataclass(frozen=True)
class FinancialBuildResult:
    reports: pd.DataFrame
    facts: pd.DataFrame
    metrics: pd.DataFrame


class FinancialProcessor:
    """Normalize versioned statements and build deterministic PIT metrics."""

    def build(
        self,
        statements: Mapping[str, pd.DataFrame],
        trade_days: Sequence[date],
        batch_id: str,
        share_capital: pd.DataFrame | None = None,
        source_name: str = "JQData",
    ) -> FinancialBuildResult:
        days = sorted(set(trade_days))
        report_rows: list[dict] = []
        fact_rows: list[dict] = []
        for statement_type, source in sorted(statements.items()):
            if statement_type not in {"balance", "income", "cash_flow"}:
                raise DataContractError(
                    f"unsupported financial statement type: {statement_type}"
                )
            if not isinstance(source, pd.DataFrame):
                raise DataContractError(
                    f"{statement_type} statement response must be a DataFrame"
                )
            if source.empty:
                continue
            required = {
                "id", "code", "pub_date", "start_date", "end_date",
                "report_date", "report_type",
            }
            missing = required - set(source)
            if missing:
                raise DataContractError(
                    f"{statement_type} statements missing fields: {sorted(missing)}"
                )
            for record in source.to_dict("records"):
                report_type = int(record["report_type"])
                if report_type not in {0, 1}:
                    raise DataContractError(
                        f"unsupported financial report_type: {report_type}"
                    )
                published = self._as_date(record["pub_date"])
                period_start = self._as_date(record["start_date"])
                period_end = self._as_date(record["end_date"])
                filing_end = self._as_date(record["report_date"])
                if None in {published, period_start, period_end, filing_end}:
                    raise DataContractError(
                        f"{statement_type} statement contains invalid dates"
                    )
                assert published is not None
                assert period_start is not None
                assert period_end is not None
                assert filing_end is not None
                available = self._next_trade_day(published, days)
                source_id = str(record["id"])
                report_id = hash_payload({
                    "source": source_name,
                    "table": statement_type,
                    "id": source_id,
                })
                common = {
                    "report_id": report_id,
                    "instrument_id": str(record["code"]),
                    "statement_type": statement_type,
                    "fiscal_period_start": period_start,
                    "fiscal_period_end": period_end,
                    "filing_period_end": filing_end,
                    "available_at": available,
                    "source_record_id": source_id,
                    "source_batch_id": batch_id,
                }
                report_rows.append({
                    **common,
                    "record_kind": (
                        "current" if report_type == 0 else "comparative"
                    ),
                    "published_at": published,
                    "revision_sequence": 0,
                    "currency": "CNY",
                    "source_report_type": str(report_type),
                })
                for field, raw_value in sorted(record.items()):
                    if field in STATEMENT_METADATA or pd.isna(raw_value):
                        continue
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(value):
                        raise DataContractError(
                            f"non-finite financial fact {statement_type}.{field}"
                        )
                    unit, basis = self._item_metadata(statement_type, field)
                    fact_rows.append({
                        **common,
                        "item_code": field,
                        "value": value,
                        "unit": unit,
                        "value_basis": basis,
                        "source_field": field,
                    })

        report_columns = [
            "report_id", "instrument_id", "statement_type",
            "fiscal_period_start", "fiscal_period_end", "filing_period_end",
            "record_kind", "published_at", "available_at",
            "revision_sequence", "currency", "source_report_type",
            "source_record_id", "source_batch_id",
        ]
        fact_columns = [
            "report_id", "item_code", "instrument_id", "statement_type",
            "fiscal_period_start", "fiscal_period_end", "filing_period_end",
            "available_at", "value", "unit", "value_basis", "source_field",
            "source_record_id", "source_batch_id",
        ]
        reports = pd.DataFrame(report_rows, columns=report_columns)
        facts = pd.DataFrame(fact_rows, columns=fact_columns)
        if not reports.empty:
            groups = [
                "instrument_id", "statement_type",
                "fiscal_period_start", "fiscal_period_end",
            ]
            reports.sort_values(
                groups + ["available_at", "record_kind", "report_id"],
                inplace=True,
                ignore_index=True,
            )
            reports["revision_sequence"] = (
                reports.groupby(groups, sort=False).cumcount() + 1
            )
            reports.sort_values("report_id", inplace=True, ignore_index=True)
        metrics = self._metrics(reports, facts, batch_id, share_capital)
        return FinancialBuildResult(reports, facts, metrics)

    def _metrics(
        self,
        reports: pd.DataFrame,
        facts: pd.DataFrame,
        batch_id: str,
        share_capital: pd.DataFrame | None,
    ) -> pd.DataFrame:
        columns = [
            "metric_id", "instrument_id", "metric_code",
            "fiscal_period_end", "basis", "value", "unit", "available_at",
            "calculation_version", "source_report_ids", "quality_status",
            "source_record_id", "source_batch_id",
        ]
        if reports.empty or facts.empty:
            return pd.DataFrame(columns=columns)
        merged = facts.merge(
            reports[["report_id", "record_kind"]],
            on="report_id",
            how="left",
            validate="many_to_one",
        )
        events = reports[
            reports["record_kind"] == "current"
        ][
            ["instrument_id", "fiscal_period_end", "available_at"]
        ].drop_duplicates()
        rows: list[dict] = []
        for event in events.sort_values(
            ["instrument_id", "available_at", "fiscal_period_end"]
        ).itertuples(index=False):
            known = merged[
                (merged["instrument_id"] == event.instrument_id)
                & (merged["available_at"] <= event.available_at)
            ]
            if known.empty:
                continue
            latest = known.sort_values(
                ["available_at", "report_id"]
            ).drop_duplicates(
                ["statement_type", "item_code", "fiscal_period_end"],
                keep="last",
            )
            source_ids: set[str] = set()

            def fact(
                statement: str,
                item: str,
                period_end: date,
            ) -> float | None:
                match = latest[
                    (latest["statement_type"] == statement)
                    & (latest["item_code"] == item)
                    & (latest["fiscal_period_end"] == period_end)
                ]
                if match.empty:
                    return None
                row = match.iloc[-1]
                source_ids.add(str(row["report_id"]))
                return float(row["value"])

            def first_fact(
                statement: str,
                items: Sequence[str],
                period_end: date,
            ) -> float | None:
                for item in items:
                    value = fact(statement, item, period_end)
                    if value is not None:
                        return value
                return None

            period_end = event.fiscal_period_end
            is_financial = any(
                first_fact("income", [item], period_end) is not None
                for item in ("interest_income", "premiums_earned", "commission_income")
            )

            def emit(
                code: str,
                basis: str,
                value: float | None,
                unit: str,
            ) -> None:
                if value is None or not np.isfinite(value):
                    return
                payload = {
                    "instrument": event.instrument_id,
                    "metric": code,
                    "period": period_end.isoformat(),
                    "basis": basis,
                    "available": event.available_at.isoformat(),
                    "version": FUNDAMENTAL_CALCULATION_VERSION,
                }
                metric_id = hash_payload(payload)
                rows.append({
                    "metric_id": metric_id,
                    "instrument_id": event.instrument_id,
                    "metric_code": code,
                    "fiscal_period_end": period_end,
                    "basis": basis,
                    "value": float(value),
                    "unit": unit,
                    "available_at": event.available_at,
                    "calculation_version": FUNDAMENTAL_CALCULATION_VERSION,
                    "source_report_ids": json.dumps(
                        sorted(source_ids), separators=(",", ":")
                    ),
                    "quality_status": "complete",
                    "source_record_id": metric_id,
                    "source_batch_id": batch_id,
                })

            total_assets = fact("balance", "total_assets", period_end)
            total_liability = fact("balance", "total_liability", period_end)
            parent_equity = fact(
                "balance", "equities_parent_company_owners", period_end
            )
            total_equity = first_fact(
                "balance",
                ["total_owner_equities", "total_sheet_owner_equities"],
                period_end,
            )
            emit("total_assets", "instant", total_assets, "CNY")
            emit("total_liability", "instant", total_liability, "CNY")
            emit("equity_parent", "instant", parent_equity, "CNY")
            emit("total_equity", "instant", total_equity, "CNY")
            basic_eps = first_fact(
                "income", ["basic_eps", "eps"], period_end
            )
            diluted_eps = fact("income", "diluted_eps", period_end)
            emit("basic_eps", "ytd", basic_eps, "CNY/share")
            emit("diluted_eps", "ytd", diluted_eps, "CNY/share")
            if (
                parent_equity is not None
                and share_capital is not None
                and not share_capital.empty
            ):
                eligible_capital = share_capital[
                    (share_capital["instrument_id"] == event.instrument_id)
                    & (share_capital["effective_from"] <= period_end)
                    & (share_capital["available_at"] <= event.available_at)
                ].sort_values([
                    "effective_from", "available_at", "capital_event_id",
                ])
                if not eligible_capital.empty:
                    shares = float(eligible_capital.iloc[-1]["total_shares"])
                    if shares > 0:
                        emit(
                            "book_value_per_share", "instant",
                            parent_equity / shares, "CNY/share",
                        )
            if total_assets and total_assets > 0 and total_liability is not None:
                emit(
                    "debt_to_assets", "instant",
                    total_liability / total_assets, "ratio",
                )
            current_assets = fact("balance", "total_current_assets", period_end)
            current_liability = fact(
                "balance", "total_current_liability", period_end
            )
            if (
                not is_financial and current_assets is not None
                and current_liability is not None and current_liability > 0
            ):
                emit(
                    "current_ratio", "instant",
                    current_assets / current_liability, "ratio",
                )

            series = {
                "revenue": (
                    "income",
                    ("total_operating_revenue", "operating_revenue"),
                ),
                "net_profit_parent": (
                    "income", ("np_parent_company_owners",),
                ),
                "net_profit": ("income", ("net_profit",)),
                "operating_profit": ("income", ("operating_profit",)),
                "operating_cost": ("income", ("operating_cost",)),
                "operating_cash_flow": (
                    "cash_flow", ("net_operate_cash_flow",),
                ),
                "capex": (
                    "cash_flow", ("fix_intan_other_asset_acqui_cash",),
                ),
            }
            values: dict[tuple[str, str], float | None] = {}
            for code, (statement, items) in series.items():
                ytd = first_fact(statement, items, period_end)
                single = self._single_quarter(
                    lambda day, s=statement, i=items: first_fact(s, i, day),
                    period_end,
                )
                ttm = self._ttm(
                    lambda day, s=statement, i=items: self._single_quarter(
                        lambda target: first_fact(s, i, target), day
                    ),
                    period_end,
                )
                values[(code, "ytd")] = ytd
                values[(code, "single_quarter")] = single
                values[(code, "ttm")] = ttm
                emit(code, "ytd", ytd, "CNY")
                emit(code, "single_quarter", single, "CNY")
                emit(code, "ttm", ttm, "CNY")

            revenue_ttm = values.get(("revenue", "ttm"))
            net_parent_ttm = values.get(("net_profit_parent", "ttm"))
            net_profit_ttm = values.get(("net_profit", "ttm"))
            operating_profit_ttm = values.get(("operating_profit", "ttm"))
            operating_cost_ttm = values.get(("operating_cost", "ttm"))
            ocf_ttm = values.get(("operating_cash_flow", "ttm"))
            capex_ttm = values.get(("capex", "ttm"))
            if revenue_ttm is not None and revenue_ttm > 0:
                if net_parent_ttm is not None:
                    emit(
                        "net_margin", "ttm",
                        net_parent_ttm / revenue_ttm, "ratio",
                    )
                if operating_profit_ttm is not None:
                    emit(
                        "operating_margin", "ttm",
                        operating_profit_ttm / revenue_ttm, "ratio",
                    )
                if not is_financial and operating_cost_ttm is not None:
                    emit(
                        "gross_margin", "ttm",
                        (revenue_ttm - operating_cost_ttm) / revenue_ttm,
                        "ratio",
                    )
            if (
                net_profit_ttm is not None and net_profit_ttm != 0
                and ocf_ttm is not None and not is_financial
            ):
                emit(
                    "ocf_to_net_profit", "ttm",
                    ocf_ttm / net_profit_ttm, "ratio",
                )
            if (
                not is_financial and ocf_ttm is not None
                and capex_ttm is not None
            ):
                emit("free_cash_flow", "ttm", ocf_ttm - capex_ttm, "CNY")

            prior_year = date(period_end.year - 1, period_end.month, period_end.day)
            prior_assets = fact("balance", "total_assets", prior_year)
            prior_equity = fact(
                "balance", "equities_parent_company_owners", prior_year
            )
            if (
                net_profit_ttm is not None and total_assets is not None
                and prior_assets is not None and total_assets + prior_assets > 0
            ):
                emit(
                    "roa", "ttm",
                    net_profit_ttm / ((total_assets + prior_assets) / 2.0),
                    "ratio",
                )
            if (
                net_parent_ttm is not None and parent_equity is not None
                and prior_equity is not None and parent_equity + prior_equity > 0
            ):
                emit(
                    "roe", "ttm",
                    net_parent_ttm / ((parent_equity + prior_equity) / 2.0),
                    "ratio",
                )

            for code, (statement, items) in (
                ("revenue", series["revenue"]),
                ("net_profit_parent", series["net_profit_parent"]),
                ("operating_cash_flow", series["operating_cash_flow"]),
            ):
                current_ytd = values.get((code, "ytd"))
                prior_ytd = first_fact(statement, items, prior_year)
                if (
                    current_ytd is not None and prior_ytd is not None
                    and prior_ytd != 0
                ):
                    emit(
                        f"{code}_yoy", "ytd",
                        current_ytd / prior_ytd - 1.0, "ratio",
                    )
                current_single = values.get((code, "single_quarter"))
                prior_single = self._single_quarter(
                    lambda day, s=statement, i=items: first_fact(s, i, day),
                    prior_year,
                )
                if (
                    current_single is not None and prior_single is not None
                    and prior_single != 0
                ):
                    emit(
                        f"{code}_yoy", "single_quarter",
                        current_single / prior_single - 1.0, "ratio",
                    )
                current_ttm = values.get((code, "ttm"))
                prior_ttm = self._ttm(
                    lambda day, s=statement, i=items: self._single_quarter(
                        lambda target: first_fact(s, i, target), day
                    ),
                    prior_year,
                )
                if (
                    current_ttm is not None and prior_ttm is not None
                    and prior_ttm != 0
                ):
                    emit(
                        f"{code}_yoy", "ttm",
                        current_ttm / prior_ttm - 1.0, "ratio",
                    )

        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _single_quarter(get_ytd, period_end: date) -> float | None:
        current = get_ytd(period_end)
        if current is None:
            return None
        if period_end.month == 3:
            return current
        previous_month = {6: 3, 9: 6, 12: 9}.get(period_end.month)
        if previous_month is None:
            return None
        previous_day = 31 if previous_month in {3, 12} else 30
        previous = get_ytd(date(period_end.year, previous_month, previous_day))
        return None if previous is None else current - previous

    @staticmethod
    def _ttm(get_single, period_end: date) -> float | None:
        endpoints = []
        year, month = period_end.year, period_end.month
        if month not in {3, 6, 9, 12}:
            return None
        for _ in range(4):
            day = 31 if month in {3, 12} else 30
            endpoints.append(date(year, month, day))
            month -= 3
            if month <= 0:
                month += 12
                year -= 1
        values = [get_single(day) for day in endpoints]
        if any(value is None for value in values):
            return None
        return float(sum(value for value in values if value is not None))

    @staticmethod
    def _item_metadata(statement: str, item: str) -> tuple[str, str]:
        if item in PER_SHARE_ITEMS:
            return "CNY/share", "per_share"
        if item in RATIO_ITEMS:
            return "ratio", "ratio"
        return "CNY", "instant" if statement == "balance" else "ytd"

    @staticmethod
    def _next_trade_day(published: date, days: Sequence[date]) -> date:
        for day in days:
            if day > published:
                return day
        cursor = published
        while True:
            cursor = date.fromordinal(cursor.toordinal() + 1)
            if cursor.weekday() < 5:
                return cursor

    @staticmethod
    def _as_date(value) -> date | None:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).date()
