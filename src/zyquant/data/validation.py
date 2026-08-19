from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import pyarrow as pa

from zyquant.core.exceptions import DataContractError

from .contracts import BASE_TABLES, FIELD_SPECS, FINANCIAL_TABLES, REQUIRED_COLUMNS
from .normalization import normalize_table


MONEY_FLOW_RTOL = 1e-6
MONEY_FLOW_ATOL_CNY = 0.01


class SnapshotValidator:
    def validate(self, tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        unknown_tables = set(tables) - set(REQUIRED_COLUMNS)
        if unknown_tables:
            raise DataContractError(
                f"unsupported canonical tables: {sorted(unknown_tables)}"
            )
        missing_tables = set(BASE_TABLES) - set(tables)
        if missing_tables:
            raise DataContractError(f"missing canonical tables: {sorted(missing_tables)}")
        normalized = {
            name: normalize_table(name, tables[name])
            for name in tables
        }
        self._validate_arrow_compatibility(normalized)
        self._validate_prices(normalized["daily_raw"], normalized["daily_post_adjusted"])
        self._validate_relations(normalized)
        self._validate_market_rules(normalized["market_rules"])
        if set(FINANCIAL_TABLES) <= set(normalized):
            self._validate_financials(normalized)
        if "daily_money_flow" in normalized:
            self._validate_money_flow(normalized)
        return normalized

    @staticmethod
    def _validate_arrow_compatibility(
        tables: Mapping[str, pd.DataFrame],
    ) -> None:
        for table_name, frame in tables.items():
            for column in frame:
                spec = FIELD_SPECS[table_name][column]
                try:
                    pa.array(
                        frame[column],
                        type=spec.arrow_type,
                        from_pandas=True,
                        safe=True,
                    )
                except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
                    raise DataContractError(
                        f"{table_name}.{column} is incompatible with "
                        f"{spec.arrow_type}"
                    ) from exc

    def _validate_prices(self, raw: pd.DataFrame, post: pd.DataFrame) -> None:
        raw_keys = raw[["trade_date", "instrument_id"]]
        post_keys = post[["trade_date", "instrument_id"]]
        if not raw_keys.equals(post_keys):
            raise DataContractError("daily_raw and daily_post_adjusted keys must match exactly")
        raw_price = raw[["open", "high", "low", "close", "pre_close"]].apply(
            pd.to_numeric, errors="coerce"
        )
        post_price = post[
            ["open_post", "high_post", "low_post", "close_post", "pre_close_post"]
        ].apply(pd.to_numeric, errors="coerce")
        if (
            raw_price.isna().any().any() or post_price.isna().any().any()
            or (raw_price <= 0).any().any() or (post_price <= 0).any().any()
            or not np.isfinite(raw_price.to_numpy()).all()
            or not np.isfinite(post_price.to_numpy()).all()
        ):
            raise DataContractError("raw and adjusted prices must be finite and positive")
        factors = pd.to_numeric(post["adjustment_factor"], errors="coerce").to_numpy()
        if not np.isfinite(factors).all() or (factors <= 0).any():
            raise DataContractError("adjustment factors must be finite and positive")
        ratios = np.column_stack([
            post[f"{name}_post"].to_numpy(dtype=float) / raw[name].to_numpy(dtype=float)
            for name in ("open", "high", "low", "close")
        ])
        if not np.allclose(ratios, factors[:, None], rtol=1e-10, atol=1e-12):
            raise DataContractError("materialized adjusted OHLC does not match adjustment_factor")

    def _validate_relations(self, tables: Mapping[str, pd.DataFrame]) -> None:
        instruments = set(tables["instruments"]["instrument_id"].astype(str))
        related = [
            "daily_raw", "daily_post_adjusted", "corporate_actions",
            "universe_membership", "industry_membership",
        ]
        related.extend(name for name in FINANCIAL_TABLES if name in tables)
        related.extend(
            name
            for name in ("special_treatment", "daily_money_flow")
            if name in tables
        )
        for name in related:
            unknown = set(tables[name]["instrument_id"].astype(str)) - instruments
            if unknown:
                raise DataContractError(f"{name} references unknown instruments: {sorted(unknown)[:10]}")
        if (pd.to_numeric(tables["daily_raw"]["volume"], errors="coerce") < 0).any():
            raise DataContractError("daily_raw.volume must be non-negative")
        if (pd.to_numeric(tables["daily_raw"]["amount"], errors="coerce") < 0).any():
            raise DataContractError("daily_raw.amount must be non-negative")
        raw = tables["daily_raw"]
        if (
            (raw["high"] < raw[["open", "close", "low"]].max(axis=1)).any()
            or (raw["low"] > raw[["open", "close", "high"]].min(axis=1)).any()
        ):
            raise DataContractError("daily_raw violates OHLC envelope")
        calendar_days = set(tables["trade_calendar"]["trade_date"])
        unknown_days = set(raw["trade_date"]) - calendar_days
        if unknown_days:
            raise DataContractError(
                f"daily_raw contains dates outside trade_calendar: {sorted(unknown_days)[:10]}"
            )
        instrument_frame = tables["instruments"]
        lot_sizes = pd.to_numeric(instrument_frame["lot_size"], errors="coerce")
        delays = pd.to_numeric(instrument_frame["sell_delay_days"], errors="coerce")
        if lot_sizes.isna().any() or (lot_sizes < 1).any():
            raise DataContractError("instrument lot_size must be positive")
        if delays.isna().any() or (delays < 0).any():
            raise DataContractError("instrument sell_delay_days must be non-negative")
        if set(instrument_frame["asset_type"].astype(str)) - {"stock", "etf"}:
            raise DataContractError("v1 instruments must be stock or etf")
        actions = tables["corporate_actions"]
        invalid_announcement = actions[
            actions["announced_at"].notna() & actions["ex_date"].notna()
            & (actions["announced_at"] > actions["ex_date"])
        ]
        if not invalid_announcement.empty:
            raise DataContractError("corporate action announcement cannot follow ex_date")

        for name in ("universe_membership", "industry_membership"):
            frame = tables[name]
            invalid = frame[
                frame["effective_to"].notna()
                & (frame["effective_to"] < frame["effective_from"])
            ]
            if not invalid.empty:
                raise DataContractError(f"{name} has an invalid effective interval")
            identity = (
                ["universe_id", "instrument_id"]
                if name == "universe_membership"
                else ["classification", "instrument_id"]
            )
            for key, group in frame.groupby(identity, dropna=False):
                ordered = group.sort_values("effective_from")
                intervals = list(ordered[
                    ["effective_from", "effective_to"]
                ].itertuples(index=False))
                for previous, current in zip(intervals, intervals[1:]):
                    if (
                        previous.effective_to is None
                        or current.effective_from <= previous.effective_to
                    ):
                        raise DataContractError(
                            f"{name} has overlapping intervals for {key}"
                        )

    @staticmethod
    def _validate_market_rules(frame: pd.DataFrame) -> None:
        if frame.empty:
            raise DataContractError("market_rules must contain at least one rule")
        if set(frame["asset_type"].astype(str)) - {"stock", "etf"}:
            raise DataContractError("market_rules asset_type must be stock or etf")
        for column in (
            "commission_bps", "minimum_commission", "sell_tax_bps",
            "buy_tax_bps", "transfer_fee_bps",
        ):
            values = pd.to_numeric(frame[column], errors="coerce")
            source_missing = frame.get(
                "source", pd.Series("", index=frame.index)
            ).astype(str).eq("source_missing")
            invalid_missing = values.isna() & ~source_missing
            finite = values.dropna()
            if (
                invalid_missing.any()
                or not np.isfinite(finite).all()
                or (finite < 0).any()
            ):
                raise DataContractError(
                    f"market_rules.{column} must be finite and non-negative, "
                    "unless source is source_missing"
                )
        invalid = frame[
            frame["effective_to"].notna()
            & (frame["effective_to"] < frame["effective_from"])
        ]
        if not invalid.empty:
            raise DataContractError("market_rules has an invalid effective interval")
        for (exchange, asset_type), group in frame.groupby(["exchange", "asset_type"]):
            ordered = group.sort_values("effective_from")
            intervals = list(ordered[["effective_from", "effective_to"]].itertuples(index=False))
            for previous, current in zip(intervals, intervals[1:]):
                if previous.effective_to is None or current.effective_from <= previous.effective_to:
                    raise DataContractError(
                        f"overlapping market rules for {exchange}/{asset_type}"
                    )

    @staticmethod
    def _validate_money_flow(tables: Mapping[str, pd.DataFrame]) -> None:
        frame = tables["daily_money_flow"]
        calendar_days = set(tables["trade_calendar"]["trade_date"])
        unknown_days = set(frame["trade_date"]) - calendar_days
        if unknown_days:
            raise DataContractError(
                "daily_money_flow contains dates outside trade_calendar: "
                f"{sorted(unknown_days)[:10]}"
            )
        if (frame["available_at"] < frame["trade_date"]).any():
            raise DataContractError(
                "daily_money_flow.available_at cannot precede trade_date"
            )

        relationships = (
            ("inflow", "outflow", "net_inflow"),
            ("inflow_s", "outflow_s", "net_inflow_s"),
            ("inflow_m", "outflow_m", "net_inflow_m"),
            ("inflow_l", "outflow_l", "net_inflow_l"),
            ("inflow_xl", "outflow_xl", "net_inflow_xl"),
        )
        for inflow, outflow, net in relationships:
            if not {inflow, outflow, net} <= set(frame):
                continue
            complete = frame[[inflow, outflow, net]].notna().all(axis=1)
            if not complete.any():
                continue
            expected = frame.loc[complete, inflow] - frame.loc[complete, outflow]
            actual = frame.loc[complete, net]
            if not np.isclose(
                actual.to_numpy(dtype=float),
                expected.to_numpy(dtype=float),
                rtol=MONEY_FLOW_RTOL,
                atol=MONEY_FLOW_ATOL_CNY,
            ).all():
                raise DataContractError(
                    f"daily_money_flow.{net} does not reconcile with "
                    f"{inflow} - {outflow}"
                )

    @staticmethod
    def _validate_financials(tables: Mapping[str, pd.DataFrame]) -> None:
        reports = tables["financial_reports"]
        invalid_period = reports[
            (reports["fiscal_period_start"] > reports["fiscal_period_end"])
            | (reports["fiscal_period_end"] > reports["filing_period_end"])
            | (reports["published_at"] > reports["available_at"])
            | (reports["fiscal_period_end"] >= reports["available_at"])
        ]
        if not invalid_period.empty:
            raise DataContractError(
                "financial_reports contains invalid period or availability dates"
            )
        group_columns = [
            "instrument_id", "statement_type",
            "fiscal_period_start", "fiscal_period_end",
        ]
        for key, group in reports.groupby(group_columns, dropna=False):
            ordered = group.sort_values(
                ["available_at", "record_kind", "report_id"]
            )
            expected = list(range(1, len(ordered) + 1))
            actual = ordered["revision_sequence"].astype(int).tolist()
            if actual != expected:
                raise DataContractError(
                    f"financial report revisions are not contiguous for {key}"
                )

        facts = tables["financial_facts"]
        unknown_reports = set(facts["report_id"]) - set(reports["report_id"])
        if unknown_reports:
            raise DataContractError(
                "financial_facts references unknown reports: "
                f"{sorted(unknown_reports)[:10]}"
            )
        report_lookup = reports.set_index("report_id")
        for row in facts[
            [
                "report_id", "instrument_id", "statement_type",
                "fiscal_period_start", "fiscal_period_end",
                "filing_period_end", "available_at",
            ]
        ].itertuples(index=False):
            report = report_lookup.loc[row.report_id]
            for column in (
                "instrument_id", "statement_type", "fiscal_period_start",
                "fiscal_period_end", "filing_period_end", "available_at",
            ):
                if getattr(row, column) != report[column]:
                    raise DataContractError(
                        f"financial_facts does not match report {row.report_id}"
                    )

        metrics = tables["fundamental_metrics"]
        if (
            metrics["fiscal_period_end"] >= metrics["available_at"]
        ).any():
            raise DataContractError(
                "fundamental metrics cannot be available before period end"
            )

        valuation = tables["daily_valuation"]
        calendar_days = set(tables["trade_calendar"]["trade_date"])
        unknown_days = set(valuation["trade_date"]) - calendar_days
        if unknown_days:
            raise DataContractError(
                "daily_valuation contains dates outside trade_calendar: "
                f"{sorted(unknown_days)[:10]}"
            )
        if (valuation["available_at"] != valuation["trade_date"]).any():
            raise DataContractError(
                "daily valuation must become available on its trade date after close"
            )

        capital = tables["share_capital"]
        if (capital["announced_at"] > capital["available_at"]).any():
            raise DataContractError(
                "share capital availability cannot precede announcement"
            )
        components = [
            "nontradable_shares", "restricted_shares", "tradable_shares",
            "a_shares", "b_shares", "h_shares",
        ]
        for column in components:
            invalid = (
                capital[column].notna()
                & (capital[column] > capital["total_shares"] * 1.000001)
            )
            if invalid.any():
                raise DataContractError(
                    f"share_capital.{column} exceeds total shares"
                )

        balance = facts[
            (facts["statement_type"] == "balance")
            & facts["item_code"].isin(
                ["total_assets", "total_liability", "total_owner_equities"]
            )
        ]
        if not balance.empty:
            pivot = balance.pivot(
                index="report_id", columns="item_code", values="value"
            ).dropna()
            if not pivot.empty:
                difference = (
                    pivot["total_assets"]
                    - pivot["total_liability"]
                    - pivot["total_owner_equities"]
                ).abs()
                tolerance = np.maximum(
                    1.0, pivot["total_assets"].abs() * 1e-6
                )
                if (difference > tolerance).any():
                    raise DataContractError(
                        "financial balance-sheet identity does not reconcile"
                    )
