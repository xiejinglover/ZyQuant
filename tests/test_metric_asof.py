"""The vectorised as-of path must agree with the per-date reference exactly.

`metric_panel` used to call `latest_metrics` once per date, which is obviously
correct but costs a full table scan per date. The rewrite reads the table once
and reduces it to per-group step functions. These tests pin the rewrite against
the original semantics, including the ordering rule that makes a naive
`merge_asof` on `available_at` wrong.
"""
from __future__ import annotations

import tempfile
from datetime import date, timedelta

import pandas as pd
import pytest

from zyquant.data import SnapshotPublisher

from tests.support import canonical_tables

CODE = "600000.XSHG"
OTHER = "600001.XSHG"


def _metric(code, metric, basis, period, available, value, metric_id):
    return {
        "metric_id": metric_id,
        "instrument_id": code,
        "metric_code": metric,
        "fiscal_period_end": period,
        "basis": basis,
        "value": value,
        "unit": "CNY",
        "available_at": available,
        "calculation_version": "1.0",
        "source_report_ids": metric_id,
        "quality_status": "complete",
    }


def _publish(directory, metrics, days):
    tables, _ = canonical_tables()
    instruments = pd.DataFrame([
        {
            "instrument_id": code, "symbol": code.split(".")[0],
            "exchange": "XSHG", "asset_type": "stock",
            "list_date": days[0] - timedelta(days=4000), "delist_date": None,
            "lot_size": 100, "sell_delay_days": 1, "name": code,
            "currency": "CNY",
        }
        for code in (CODE, OTHER)
    ])
    calendar = pd.DataFrame(
        [{"trade_date": day, "exchange": "XSHG"} for day in days]
    )
    rows, valuation = [], []
    for code in (CODE, OTHER):
        for day in days:
            rows.append({
                "trade_date": day, "instrument_id": code,
                "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
                "pre_close": 10.0, "volume": 1_000_000, "amount": 1.0e7,
                "paused": False, "limit_up": 11.0, "limit_down": 9.0,
            })
            valuation.append({
                "trade_date": day, "instrument_id": code,
                "dividend_yield": 0.05, "circulating_market_cap": 1.0e10,
                "market_cap": 1.2e10, "total_shares": 1.0e9,
                "circulating_shares": 8.0e8, "free_float_shares": 7.0e8,
                "free_float_market_cap": 9.0e9, "a_shares": 1.0e9,
                "a_market_cap": 1.2e10, "pe_ttm": 12.0, "pe_lyr": 13.0,
                "pb": 1.2, "ps_ttm": 2.0, "pcf_ttm": 8.0,
                "pcf_operating_ttm": 7.0, "turnover_rate": 0.01,
                "available_at": day,
            })
    from zyquant.data.contracts import REQUIRED_COLUMNS

    def empty(name):
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS[name]))

    payload = {
        "instruments": instruments,
        "trade_calendar": calendar,
        "daily_raw": pd.DataFrame(rows),
        "corporate_actions": empty("corporate_actions"),
        "universe_membership": pd.DataFrame([
            {
                "universe_id": "TEST", "instrument_id": code,
                "effective_from": days[0], "effective_to": None,
                "known_at": days[0],
            }
            for code in (CODE, OTHER)
        ]),
        "industry_membership": pd.DataFrame([
            {
                "classification": "TEST", "industry_id": "IND",
                "instrument_id": code, "effective_from": days[0],
                "effective_to": None, "known_at": days[0],
            }
            for code in (CODE, OTHER)
        ]),
        "market_rules": tables["market_rules"],
        "financial_reports": empty("financial_reports"),
        "financial_facts": empty("financial_facts"),
        "fundamental_metrics": pd.DataFrame(metrics),
        "daily_valuation": pd.DataFrame(valuation),
        "share_capital": empty("share_capital"),
    }
    lineage = {
        "capabilities": {
            "financials": {"schema_version": "1.1", "pit_validated": True}
        }
    }
    return SnapshotPublisher(directory).publish(
        "asof-v1", payload, schema_version="1.1", lineage=lineage,
    )


def _trading_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _reference_panel(view, dates, metric_codes=None, bases=None):
    """The original implementation: one `latest_metrics` call per date."""
    outputs = []
    for day in sorted(set(dates)):
        current = view.latest_metrics(day, None, metric_codes, bases).copy()
        current.insert(0, "as_of_date", day)
        outputs.append(current)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True, sort=False)


def _compare(view, dates, metric_codes=None, bases=None):
    fast = view.metric_panel(dates, None, metric_codes, bases)
    slow = _reference_panel(view, dates, metric_codes, bases)
    order = ["as_of_date", "instrument_id", "metric_code", "basis"]
    if slow.empty:
        assert fast.empty
        return
    left = fast.sort_values(order).reset_index(drop=True)
    right = slow.sort_values(order).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left[right.columns], right, check_dtype=False
    )


def test_panel_matches_per_date_reference_on_a_plain_series():
    days = _trading_days(date(2024, 1, 2), 120)
    metrics = [
        _metric(CODE, "revenue", "ytd", date(2023, 12, 31),
                date(2024, 1, 10), 100.0, "m1"),
        _metric(CODE, "revenue", "ytd", date(2024, 3, 31),
                date(2024, 4, 20), 130.0, "m2"),
        _metric(OTHER, "revenue", "ytd", date(2023, 12, 31),
                date(2024, 1, 15), 200.0, "m3"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        snapshot = _publish(directory, metrics, days)
        _compare(snapshot.financial(days[-1]), days)


def test_a_late_restatement_of_an_older_period_does_not_win():
    """The ordering key is (fiscal_period_end, available_at, metric_id).

    A restatement of an older period published *after* a newer period is
    already on file has the later `available_at` but the earlier
    `fiscal_period_end`, so it must lose. A naive `merge_asof` on
    `available_at` alone would wrongly pick it.
    """
    days = _trading_days(date(2024, 1, 2), 140)
    metrics = [
        # Newer period, published first.
        _metric(CODE, "revenue", "ytd", date(2024, 3, 31),
                date(2024, 4, 15), 130.0, "m_new"),
        # Older period, restated and published later.
        _metric(CODE, "revenue", "ytd", date(2023, 12, 31),
                date(2024, 5, 20), 999.0, "m_old_restated"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        snapshot = _publish(directory, metrics, days)
        view = snapshot.financial(days[-1])
        _compare(view, days)

        wide = view.metric_matrix(days, "revenue", "ytd")
        after_restatement = [day for day in days if day >= date(2024, 5, 20)]
        # The newer period keeps winning even after the restatement lands.
        assert wide.loc[after_restatement[0], CODE] == pytest.approx(130.0)
        assert wide[CODE].dropna().unique().tolist() == [130.0]


def test_same_day_eligibility_resolves_by_the_full_ordering_key():
    days = _trading_days(date(2024, 1, 2), 120)
    stamp = date(2024, 5, 6)
    metrics = [
        _metric(CODE, "revenue", "ytd", date(2023, 12, 31), stamp, 10.0, "a"),
        _metric(CODE, "revenue", "ytd", date(2024, 3, 31), stamp, 20.0, "b"),
        # Same period and availability as the winner, larger metric_id.
        _metric(CODE, "revenue", "ytd", date(2024, 3, 31), stamp, 30.0, "c"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        snapshot = _publish(directory, metrics, days)
        view = snapshot.financial(days[-1])
        _compare(view, days)
        wide = view.metric_matrix(days, "revenue", "ytd")
        assert wide.loc[stamp, CODE] == pytest.approx(30.0)


def test_contract_forbids_availability_before_period_end():
    """Why the implementation takes max(available_at, fiscal_period_end).

    `latest_metrics` filters on both columns, so a metric available before its
    period ended would need the later of the two to become usable. The data
    contract already rules that state out, so the max() is defensive rather
    than load-bearing — but it costs nothing and keeps the reduction correct
    if the contract is ever loosened.
    """
    from zyquant.core.exceptions import DataContractError

    days = _trading_days(date(2024, 1, 2), 60)
    metrics = [
        # Published before the period it covers has even ended.
        _metric(CODE, "revenue", "ytd", date(2024, 3, 31),
                date(2024, 2, 1), 500.0, "impossible"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(DataContractError, match="available before period"):
            _publish(directory, metrics, days)


def test_multiple_metrics_and_bases_stay_separated():
    days = _trading_days(date(2024, 1, 2), 60)
    metrics = [
        _metric(CODE, "revenue", "ytd", date(2023, 12, 31),
                date(2024, 1, 10), 100.0, "r_ytd"),
        _metric(CODE, "revenue", "ttm", date(2023, 12, 31),
                date(2024, 1, 10), 400.0, "r_ttm"),
        _metric(CODE, "net_profit", "ytd", date(2023, 12, 31),
                date(2024, 1, 10), 20.0, "n_ytd"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        snapshot = _publish(directory, metrics, days)
        view = snapshot.financial(days[-1])
        _compare(view, days)
        _compare(view, days, ["revenue"], ["ytd"])

        assert view.metric_matrix(
            days, "revenue", "ttm"
        ).loc[days[-1], CODE] == pytest.approx(400.0)
        assert view.metric_matrix(
            days, "net_profit", "ytd"
        ).loc[days[-1], CODE] == pytest.approx(20.0)


def test_dates_before_any_report_are_missing_not_zero():
    days = _trading_days(date(2024, 1, 2), 60)
    metrics = [
        _metric(CODE, "revenue", "ytd", date(2023, 12, 31),
                date(2024, 2, 20), 100.0, "m1"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        snapshot = _publish(directory, metrics, days)
        view = snapshot.financial(days[-1])
        _compare(view, days)
        wide = view.metric_matrix(days, "revenue", "ytd")
        assert wide.loc[days[0], CODE] != wide.loc[days[0], CODE]  # NaN
        assert wide.loc[date(2024, 2, 20), CODE] == pytest.approx(100.0)


def test_matrix_and_panel_agree_cell_by_cell():
    days = _trading_days(date(2024, 1, 2), 70)
    metrics = [
        _metric(CODE, "revenue", "ytd", date(2023, 12, 31),
                date(2024, 1, 10), 100.0, "m1"),
        _metric(CODE, "revenue", "ytd", date(2024, 3, 31),
                date(2024, 4, 20), 130.0, "m2"),
        _metric(OTHER, "revenue", "ytd", date(2023, 12, 31),
                date(2024, 1, 15), 200.0, "m3"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        snapshot = _publish(directory, metrics, days)
        view = snapshot.financial(days[-1])
        long = view.metric_panel(days, None, ["revenue"], ["ytd"])
        wide = view.metric_matrix(days, "revenue", "ytd")

        for row in long.itertuples(index=False):
            assert wide.loc[row.as_of_date, row.instrument_id] == pytest.approx(
                row.value
            )
        # And every non-null wide cell appears in the long form.
        assert int(wide.notna().sum().sum()) == len(long)


def test_guard_rejects_a_date_past_the_cutoff():
    days = _trading_days(date(2024, 1, 2), 40)
    metrics = [
        _metric(CODE, "revenue", "ytd", date(2023, 12, 31),
                date(2024, 1, 10), 100.0, "m1"),
    ]
    from zyquant.core.exceptions import FutureDataError

    with tempfile.TemporaryDirectory() as directory:
        snapshot = _publish(directory, metrics, days)
        view = snapshot.financial(days[20])
        with pytest.raises(FutureDataError):
            view.metric_panel(days)
        with pytest.raises(FutureDataError):
            view.metric_matrix(days, "revenue", "ytd")
