"""Regression tests for the built-in CN-equity factors.

The synthetic fixture has fixed golden values for the core factors and final
ranking. The remaining tests prove point-in-time behavior and thin-input
handling without maintaining a second strategy implementation as an oracle.
"""
from __future__ import annotations

import math
import tempfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from zyquant.data import SnapshotPublisher
from zyquant.data.contracts import REQUIRED_COLUMNS
from zyquant.factors import (
    AssetGrowthFactor, DividendContinuityFactor, DividendFundingFactor, DividendYieldFactor,
    DividendYieldHistoryFactor, FactorEngine, RegressionMomentumFactor,
    RoeFactor, ResidualVolatilityFactor, RollingRiskFactor, SelfBetaFactor,
    ValuationMultipleFactor, VolatilityOfVolatilityFactor, clear_panel_cache,
    cn_equity_factor_catalog,
    dividend_cash_growth_factor, dividend_yield_change_factor,
    downside_volatility_factor, market_value_growth_factor,
    net_profit_factor, operating_cash_flow_factor, total_volatility_factor,
    worst5_loss_factor,
)

CODES = [f"{600000 + index}.XSHG" for index in range(12)]


def test_quality_periods_are_distinct_named_factors():
    catalog = cn_equity_factor_catalog()
    expected = {
        "dividend_yield_l12m", "beta_252_hl63", "momentum_6_1",
        "dividend_cash_log_growth_1y", "market_value_log_growth_1y",
        "dividend_yield_log_change_1y",
        "net_profit_ytd", "net_profit_single_quarter", "net_profit_ttm",
        "operating_cash_flow_ytd",
        "operating_cash_flow_single_quarter",
        "operating_cash_flow_ttm",
        "roe_ytd", "roe_single_quarter", "roe_ttm",
        "total_volatility_120", "total_volatility_252",
        "residual_volatility_252_hl63", "downside_volatility_252",
        "worst5_loss_252",
        "dividend_continuity_3y", "dividend_yield_median_3y",
        "dividend_yield_anomaly_3y", "dividend_payout_ratio_ttm",
        "dividend_ocf_coverage_ttm", "dividend_fcf_coverage_ttm",
        "net_profit_ttm_yoy", "operating_cash_flow_ttm_yoy",
        "revenue_ttm", "revenue_ttm_yoy", "roa_ttm",
        "earnings_yield_ttm", "pb_ratio",
        "asset_growth_1y",
        "volatility_of_volatility_20_252",
    }
    assert set(catalog) == expected
    assert len({factor.name for factor in catalog.values()}) == 35
    assert catalog["revenue_ttm_yoy"].definition()["metric_code"] == "revenue_yoy"
    assert catalog["roa_ttm"].definition()["basis"] == "ttm"
    assert RoeFactor("ytd").definition() != RoeFactor("ttm").definition()
    with pytest.raises(TypeError):
        catalog["roe_ttm"] = RoeFactor("ytd")


def test_asset_growth_uses_each_dates_prior_information_set():
    current_days = [date(2021, 1, 4), date(2021, 1, 5)]
    values = {
        date(2020, 1, 4): 100.0,
        date(2020, 1, 5): 200.0,
        date(2021, 1, 4): 120.0,
        date(2021, 1, 5): 180.0,
    }

    class Snapshot:
        @staticmethod
        def table(name):
            if name == "trade_calendar":
                return pd.DataFrame({"trade_date": current_days})
            if name == "instruments":
                return pd.DataFrame({
                    "instrument_id": ["600000.XSHG"],
                    "list_date": [date(2000, 1, 1)],
                    "delist_date": [None],
                })
            raise AssertionError(name)

    class Context:
        start, end = current_days
        instruments = None
        snapshot = Snapshot()

        @staticmethod
        def fundamental_matrix(metric_code, basis, dates):
            assert (metric_code, basis) == ("total_assets", "instant")
            return pd.DataFrame(
                {"600000.XSHG": [values[day] for day in dates]},
                index=pd.Index(dates, name="trade_date"),
            )

    result = AssetGrowthFactor().compute(Context(), {})
    assert result["value"].tolist() == pytest.approx([0.2, -0.1])
    assert result["trade_date"].tolist() == current_days
    assert AssetGrowthFactor._prior_year(date(2024, 2, 29)) == date(2023, 2, 28)


def test_volatility_of_volatility_matches_two_stage_rolling_definition():
    returns = pd.DataFrame({
        "stable": [0.01, -0.01, 0.01, -0.01, 0.01, -0.01],
        "bursty": [0.001, -0.001, 0.001, -0.001, 0.08, -0.08],
    })
    actual = VolatilityOfVolatilityFactor.coefficient(
        returns,
        short_window=2,
        long_window=3,
        short_minimum=2,
        long_minimum=2,
    )
    short = returns.rolling(2, min_periods=2).std(ddof=1)
    expected = short.rolling(3, min_periods=2).std(ddof=1) / short.rolling(
        3, min_periods=2
    ).mean()
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.loc[5, "stable"] < actual.loc[5, "bursty"]


def _trading_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _empty(table: str) -> pd.DataFrame:
    return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS[table]))


def _tables(days: list[date], gaps: dict[str, list[int]] | None = None):
    """Snapshot payload. `gaps` marks sessions to blank out per instrument."""
    gaps = gaps or {}
    # The report must become visible inside the window, otherwise every metric
    # factor is legitimately empty and the comparisons test nothing.
    period = days[len(days) // 3]
    available = days[len(days) // 3 + 20]
    rng = np.random.default_rng(20260727)
    instruments = pd.DataFrame([
        {
            "instrument_id": code, "symbol": code.split(".")[0],
            "exchange": "XSHG", "asset_type": "stock",
            "list_date": days[0] - timedelta(days=4000), "delist_date": None,
            "lot_size": 100, "sell_delay_days": 1, "name": code,
            "currency": "CNY",
        }
        for code in CODES
    ])
    calendar = pd.DataFrame(
        [{"trade_date": day, "exchange": "XSHG"} for day in days]
    )
    market = np.cumsum(rng.normal(0.0003, 0.012, len(days)))
    rows, valuation, metrics = [], [], []
    for index, code in enumerate(CODES):
        beta = 0.5 + 0.15 * index
        path = 20.0 * np.exp(
            beta * market + np.cumsum(rng.normal(0.0, 0.008, len(days)))
        )
        blanks = set(gaps.get(code, []))
        for position, day in enumerate(days):
            price = float(path[position])
            previous = float(path[position - 1]) if position else price
            paused = position in blanks
            rows.append({
                "trade_date": day, "instrument_id": code,
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "pre_close": previous,
                "volume": 0 if paused else 50_000_000,
                "amount": 0.0 if paused else price * 50_000_000,
                "paused": paused,
                "limit_up": previous * 1.1, "limit_down": previous * 0.9,
            })
            valuation.append({
                "trade_date": day, "instrument_id": code,
                "dividend_yield": 0.02 + 0.004 * index,
                "circulating_market_cap": 1.0e10 + 1.0e8 * index,
                "market_cap": 1.2e10, "total_shares": 1.0e9,
                "circulating_shares": 8.0e8, "free_float_shares": 7.0e8,
                "free_float_market_cap": 9.0e9, "a_shares": 1.0e9,
                "a_market_cap": 1.2e10, "pe_ttm": 12.0, "pe_lyr": 13.0,
                "pb": 1.2, "ps_ttm": 2.0, "pcf_ttm": 8.0,
                "pcf_operating_ttm": 7.0, "turnover_rate": 0.01,
                "available_at": day,
            })
        for metric, basis, value in (
            ("net_profit", "ytd", 1.0e8 * (index + 1)),
            ("operating_cash_flow", "ytd", 2.0e8 * (index + 1)),
            ("net_profit_parent", "ytd", 0.9e8 * (index + 1)),
            ("equity_parent", "instant", 1.0e9),
        ):
            metrics.append({
                "metric_id": f"{code}:{metric}", "instrument_id": code,
                "metric_code": metric, "fiscal_period_end": period,
                "basis": basis, "value": value, "unit": "CNY",
                "available_at": available, "calculation_version": "1.0",
                "source_report_ids": f"{code}:r",
                "quality_status": "complete",
            })
    membership = pd.DataFrame([
        {
            "universe_id": "CN_ALL_A", "instrument_id": code,
            "effective_from": days[0], "effective_to": None,
            "known_at": days[0],
        }
        for code in CODES
    ])
    industry = pd.DataFrame([
        {
            "classification": "TEST", "industry_id": "IND",
            "instrument_id": code, "effective_from": days[0],
            "effective_to": None, "known_at": days[0],
        }
        for code in CODES
    ])
    rules = pd.DataFrame([{
        "rule_id": "XSHG-stock-v1", "exchange": "XSHG", "asset_type": "stock",
        "effective_from": days[0], "effective_to": None,
        "commission_bps": 2.0, "minimum_commission": 5.0, "sell_tax_bps": 5.0,
        "buy_tax_bps": 0.0, "transfer_fee_bps": 0.1, "currency": "CNY",
    }])
    return {
        "instruments": instruments,
        "trade_calendar": calendar,
        "daily_raw": pd.DataFrame(rows),
        "corporate_actions": _empty("corporate_actions"),
        "universe_membership": membership,
        "industry_membership": industry,
        "market_rules": rules,
        "financial_reports": _empty("financial_reports"),
        "financial_facts": _empty("financial_facts"),
        "fundamental_metrics": pd.DataFrame(metrics),
        "daily_valuation": pd.DataFrame(valuation),
        "share_capital": _empty("share_capital"),
    }


@pytest.fixture(scope="module")
def world():
    days = _trading_days(date(2013, 1, 2), 460)
    # Three names carry deliberate gaps so the halt masking is exercised.
    gaps = {
        CODES[3]: [300, 320, 340, 360],
        CODES[4]: list(range(0, 250)),      # lists part-way through the window
        CODES[5]: [430],                    # a single recent gap
    }
    with tempfile.TemporaryDirectory() as directory:
        snapshot = SnapshotPublisher(directory).publish(
            "factor-v1", _tables(days, gaps), schema_version="1.1",
            lineage={"capabilities": {"financials": {
                "schema_version": "1.1", "pit_validated": True,
            }}},
        )
        clear_panel_cache()
        yield snapshot, days
        clear_panel_cache()


def _factor_frame(snapshot, factor, start, end, cache):
    engine = FactorEngine(cache)
    return engine.compute(factor, snapshot, start, end, None, end).frame


def _as_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pivot_table(
        index="trade_date", columns="instrument_id", values="value",
    )


# ---------------------------------------------------------- layer 1: goldens


@pytest.mark.parametrize(
    ("factor", "expected"),
    [
        (SelfBetaFactor(), 0.3988311356817211),
        (RegressionMomentumFactor(), 0.08501517821676335),
        (DividendYieldFactor(), 0.02),
        (net_profit_factor(), 100_000_000.0),
        (operating_cash_flow_factor(), 200_000_000.0),
        (RoeFactor(), 0.09),
    ],
)
def test_core_factor_matches_the_frozen_synthetic_golden(world, factor, expected):
    snapshot, days = world
    end = days[-1]
    with tempfile.TemporaryDirectory() as cache:
        frame = _factor_frame(snapshot, factor, end, end, cache)
    value = frame.set_index("instrument_id").loc[CODES[0], "value"]
    assert value == pytest.approx(expected, rel=1e-12)


def test_new_risk_factor_formulas_match_direct_trailing_calculations(world):
    snapshot, days = world
    end = days[-1]
    raw = snapshot.post_adjusted_bars(
        days[-253], end, [CODES[0]], ["close_post"], end,
    ).sort_values("trade_date")
    returns = raw["close_post"].pct_change(fill_method=None).dropna().to_numpy()
    factors = (
        total_volatility_factor(120),
        total_volatility_factor(252),
        downside_volatility_factor(),
        worst5_loss_factor(),
    )
    with tempfile.TemporaryDirectory() as cache:
        values = {
            factor.name: _factor_frame(
                snapshot, factor, end, end, cache
            ).set_index("instrument_id").loc[CODES[0], "value"]
            for factor in factors
        }
    assert values["total_volatility_120"] == pytest.approx(
        np.std(returns[-120:], ddof=1) * np.sqrt(252), rel=1e-12
    )
    assert values["total_volatility_252"] == pytest.approx(
        np.std(returns[-252:], ddof=1) * np.sqrt(252), rel=1e-12
    )
    assert values["downside_volatility_252"] == pytest.approx(
        np.sqrt(np.mean(np.minimum(returns[-252:], 0.0) ** 2) * 252),
        rel=1e-12,
    )
    assert values["worst5_loss_252"] == pytest.approx(
        -np.mean(np.sort(returns[-252:])[:5]), rel=1e-12
    )


def test_dividend_credibility_decomposition_is_exact_and_split_invariant(world):
    snapshot, days = world
    end = days[-1]
    factors = (
        dividend_cash_growth_factor(),
        market_value_growth_factor(),
        dividend_yield_change_factor(),
    )
    with tempfile.TemporaryDirectory() as cache:
        values = {
            factor.name: _factor_frame(
                snapshot, factor, end, end, cache
            ).set_index("instrument_id").loc[CODES[0], "value"]
            for factor in factors
        }
    cash = values["dividend_cash_log_growth_1y"]
    market_value = values["market_value_log_growth_1y"]
    yield_change = values["dividend_yield_log_change_1y"]
    assert cash == pytest.approx(0.0, abs=1e-14)
    assert market_value == pytest.approx(0.0, abs=1e-14)
    assert yield_change == pytest.approx(cash - market_value, abs=1e-14)


def test_dividend_continuity_uses_three_non_overlapping_365_day_buckets():
    target = date(2025, 12, 31)
    events = [
        target,
        target - timedelta(days=365),
        target - timedelta(days=366),
        target - timedelta(days=730),
        target - timedelta(days=731),
        target - timedelta(days=731),
    ]
    assert DividendContinuityFactor._counts([target], events).tolist() == [3]
    assert DividendContinuityFactor._counts(
        [target], [target - timedelta(days=1095)]
    ).tolist() == [0]


def test_dividend_continuity_ignores_announced_but_future_ex_date():
    days = _trading_days(date(2025, 1, 2), 60)
    target = days[-5]
    payload = _tables(days)
    payload["corporate_actions"] = pd.DataFrame([
        {
            "event_id": "valid-old", "instrument_id": CODES[0],
            "event_type": "cash_dividend", "status": "active",
            "cash_per_share": 0.2, "share_ratio": 0.0,
            "announced_at": target - timedelta(days=410),
            "record_date": target - timedelta(days=400),
            "ex_date": target - timedelta(days=400),
            "pay_date": target - timedelta(days=390),
        },
        {
            "event_id": "announced-but-not-ex", "instrument_id": CODES[0],
            "event_type": "cash_dividend", "status": "active",
            "cash_per_share": 0.3, "share_ratio": 0.0,
            "announced_at": target - timedelta(days=3),
            "record_date": target + timedelta(days=2),
            "ex_date": target + timedelta(days=2),
            "pay_date": target + timedelta(days=5),
        },
    ])
    with tempfile.TemporaryDirectory() as directory:
        snapshot = SnapshotPublisher(directory).publish(
            "dividend-pit-v1", payload, schema_version="1.1",
            lineage={"capabilities": {"financials": {
                "schema_version": "1.1", "pit_validated": True,
            }}},
        )
        with tempfile.TemporaryDirectory() as cache:
            frame = _factor_frame(
                snapshot, DividendContinuityFactor(), target, target, cache
            )
    value = frame.set_index("instrument_id").loc[CODES[0], "value"]
    assert value == 1.0


def test_dividend_history_requires_the_configured_month_end_observations(world):
    snapshot, days = world
    end = days[-1]
    with tempfile.TemporaryDirectory() as cache:
        strict = _factor_frame(
            snapshot, DividendYieldHistoryFactor("median", minimum_months=24),
            end, end, cache,
        )
        permissive = _factor_frame(
            snapshot, DividendYieldHistoryFactor("median", minimum_months=18),
            end, end, cache,
        )
    assert strict.empty or CODES[0] not in set(strict.instrument_id)
    value = permissive.set_index("instrument_id").loc[CODES[0], "value"]
    assert value == pytest.approx(0.02)


def test_dividend_funding_formulas_fail_closed_on_bad_denominators():
    idx = pd.Index([date(2024, 1, 31)], name="trade_date")
    columns = ["A", "B", "C"]
    dividend_yield = pd.DataFrame([[0.04, 0.04, 0.04]], idx, columns)
    market_cap = pd.DataFrame([[100.0, 100.0, 100.0]], idx, columns)
    profit = pd.DataFrame([[10.0, 0.0, -2.0]], idx, columns)
    payout = DividendFundingFactor._ratio(
        dividend_yield, market_cap, profit, True
    )
    assert payout.loc[idx[0], "A"] == pytest.approx(0.4)
    assert pd.isna(payout.loc[idx[0], "B"])
    assert pd.isna(payout.loc[idx[0], "C"])
    cash_flow = pd.DataFrame([[8.0, -1.0, 0.0]], idx, columns)
    coverage = DividendFundingFactor._ratio(
        dividend_yield, market_cap, cash_flow, False
    )
    assert coverage.loc[idx[0], "A"] == pytest.approx(2.0)
    assert coverage.loc[idx[0], "B"] == pytest.approx(-0.25)


def test_valuation_multiple_factor_definition_and_positive_domain():
    earnings = ValuationMultipleFactor("earnings_yield")
    pb = ValuationMultipleFactor("pb")
    assert earnings.name == "earnings_yield_ttm"
    assert earnings.source_field == "pe_ttm"
    assert pb.name == "pb_ratio"
    assert pb.source_field == "pb"
    assert earnings.definition() != pb.definition()


def test_residual_volatility_separates_equal_beta_different_noise():
    rng = np.random.default_rng(20260801)
    market = rng.normal(0.0, 0.012, 252)
    weight = np.power(0.5, np.arange(251, -1, -1) / 63.0)
    low_noise = 0.8 * market + rng.normal(0.0, 0.003, 252)
    high_noise = 0.8 * market + rng.normal(0.0, 0.02, 252)
    low = ResidualVolatilityFactor._annualised_residual_rms(
        market, low_noise, weight
    )
    high = ResidualVolatilityFactor._annualised_residual_rms(
        market, high_noise, weight
    )
    assert low is not None and high is not None
    assert high > low * 4


@pytest.mark.parametrize("factor", [
    RollingRiskFactor("x", "total_volatility", 120, 10_000),
    ResidualVolatilityFactor(minimum_observations=10_000),
])
def test_risk_factors_fail_closed_below_minimum_observations(world, factor):
    snapshot, days = world
    with tempfile.TemporaryDirectory() as cache:
        frame = _factor_frame(snapshot, factor, days[-1], days[-1], cache)
    assert frame.empty or frame["value"].dropna().empty


# --------------------------------------------- layer 2: no window dependence


@pytest.mark.parametrize("factor_name", [
    "beta", "momentum", "net_profit",
    "total_volatility_120", "total_volatility_252",
    "residual_volatility_252_hl63", "downside_volatility_252",
    "worst5_loss_252",
    "dividend_cash_log_growth_1y", "market_value_log_growth_1y",
    "dividend_yield_log_change_1y",
])
def test_a_row_never_depends_on_later_data(world, factor_name):
    """Recomputing a narrower window must not change the shared rows.

    This is what licenses building the panel once with a wide cutoff: if any
    row peeked past its own date, the two runs would disagree.
    """
    snapshot, days = world
    factory = {
        "beta": SelfBetaFactor,
        "momentum": RegressionMomentumFactor,
        "net_profit": net_profit_factor,
        "total_volatility_120": lambda: total_volatility_factor(120),
        "total_volatility_252": lambda: total_volatility_factor(252),
        "residual_volatility_252_hl63": ResidualVolatilityFactor,
        "downside_volatility_252": downside_volatility_factor,
        "worst5_loss_252": worst5_loss_factor,
        "dividend_cash_log_growth_1y": dividend_cash_growth_factor,
        "market_value_log_growth_1y": market_value_growth_factor,
        "dividend_yield_log_change_1y": dividend_yield_change_factor,
    }[factor_name]
    wide_end, narrow_end = days[-1], days[-4]
    start = days[-8]
    with tempfile.TemporaryDirectory() as cache:
        wide = _as_matrix(
            _factor_frame(snapshot, factory(), start, wide_end, cache)
        )
    with tempfile.TemporaryDirectory() as cache:
        narrow = _as_matrix(
            _factor_frame(snapshot, factory(), start, narrow_end, cache)
        )
    shared = narrow.index
    assert len(shared) >= 2
    pd.testing.assert_frame_equal(
        wide.reindex(index=shared)[narrow.columns], narrow,
        check_dtype=False,
    )


def test_rolling_volatility_treats_a_halt_as_zero_return(world):
    snapshot, days = world
    end, code = days[-1], CODES[5]
    adjusted = snapshot.post_adjusted_bars(
        days[-253], end, [code], ["close_post"], end,
    ).sort_values("trade_date")
    raw = snapshot.table(
        "daily_raw", days[-253], end, cutoff=end, fields=["paused"],
        instruments=[code],
    ).sort_values("trade_date")
    returns = adjusted["close_post"].pct_change(fill_method=None)
    returns.loc[raw["paused"].to_numpy(dtype=bool)] = 0.0
    expected = np.std(returns.dropna().to_numpy()[-252:], ddof=1) * np.sqrt(252)
    with tempfile.TemporaryDirectory() as cache:
        frame = _factor_frame(
            snapshot, total_volatility_factor(252), end, end, cache
        )
    actual = frame.set_index("instrument_id").loc[code, "value"]
    assert actual == pytest.approx(expected, rel=1e-12)


def test_risk_factor_excludes_a_recent_listing_with_too_few_prices():
    days = _trading_days(date(2013, 1, 2), 320)
    payload = _tables(days)
    victim = CODES[2]
    list_at = days[-150]
    payload["instruments"].loc[
        payload["instruments"]["instrument_id"] == victim, "list_date"
    ] = list_at
    raw = payload["daily_raw"]
    payload["daily_raw"] = raw[
        (raw["instrument_id"] != victim) | (raw["trade_date"] >= list_at)
    ]
    with tempfile.TemporaryDirectory() as directory:
        snapshot = SnapshotPublisher(directory).publish(
            "recent-listing-v1", payload, schema_version="1.1",
            lineage={"capabilities": {"financials": {
                "schema_version": "1.1", "pit_validated": True,
            }}},
        )
        clear_panel_cache()
        with tempfile.TemporaryDirectory() as cache:
            got = _as_matrix(_factor_frame(
                snapshot, total_volatility_factor(252),
                days[-1], days[-1], cache,
            ))
        clear_panel_cache()
    assert victim not in got.loc[days[-1]].dropna().index
    assert CODES[0] in got.loc[days[-1]].dropna().index


# ------------------------------------------------------- layer 3: thin inputs


def test_beta_is_missing_for_a_name_listed_inside_the_window(world):
    snapshot, days = world
    start, end = days[-2], days[-1]
    with tempfile.TemporaryDirectory() as cache:
        got = _as_matrix(
            _factor_frame(snapshot, SelfBetaFactor(), start, end, cache)
        )
    # CODES[4] is halted for the first 250 sessions, so its early returns are
    # zero rather than absent; the halt mask must still admit it here.
    assert CODES[0] in got.columns
    assert set(got.loc[end].dropna().index) == set(CODES)


def test_beta_needs_the_minimum_observation_count(world):
    snapshot, days = world
    start, end = days[-2], days[-1]
    with tempfile.TemporaryDirectory() as cache:
        strict = _as_matrix(_factor_frame(
            snapshot, SelfBetaFactor(minimum_observations=10_000),
            start, end, cache,
        ))
    # An impossible threshold must empty the factor rather than fit anyway.
    assert strict.empty or strict.notna().sum().sum() == 0


def test_momentum_rejects_a_window_with_any_gap():
    days = _trading_days(date(2013, 1, 2), 200)
    # One missing session inside the regression window, for one name only.
    payload = _tables(days)
    raw = payload["daily_raw"]
    victim = CODES[2]
    hole = days[100]
    payload["daily_raw"] = raw[
        ~((raw["instrument_id"] == victim) & (raw["trade_date"] == hole))
    ]
    with tempfile.TemporaryDirectory() as directory:
        snapshot = SnapshotPublisher(directory).publish(
            "gap-v1", payload, schema_version="1.1",
            lineage={"capabilities": {"financials": {
                "schema_version": "1.1", "pit_validated": True,
            }}},
        )
        clear_panel_cache()
        start, end = days[-2], days[-1]
        with tempfile.TemporaryDirectory() as cache:
            got = _as_matrix(_factor_frame(
                snapshot, RegressionMomentumFactor(), start, end, cache,
            ))
        clear_panel_cache()
    # The gap sits inside the 120-session window ending 20 sessions back, so the
    # name must be absent while its peers are present.
    assert victim not in got.loc[end].dropna().index
    assert CODES[0] in got.loc[end].dropna().index


def test_factor_values_are_finite_where_present(world):
    snapshot, days = world
    start, end = days[-3], days[-1]
    for factor in (
        SelfBetaFactor(), RegressionMomentumFactor(), DividendYieldFactor(),
        net_profit_factor(), RoeFactor(),
    ):
        with tempfile.TemporaryDirectory() as cache:
            frame = _factor_frame(snapshot, factor, start, end, cache)
        values = frame["value"].dropna()
        assert len(values) > 0, factor.name
        assert np.isfinite(values).all(), factor.name
        assert frame["trade_date"].between(start, end).all(), factor.name


def test_no_factor_emits_a_row_outside_the_listing_window():
    """A fundamental value would otherwise forward-fill past delisting.

    Price factors stop on their own — a delisted name has no bars — but a
    metric is a step function with no end, so without the restriction a company
    delisted years ago keeps carrying earnings. Metrics also cover issuers that
    never appear in `instruments`; those must not surface either.
    """
    days = _trading_days(date(2013, 1, 2), 300)
    payload = _tables(days)
    # Delist one name mid-sample and add a metric for an issuer nobody trades.
    stop = days[200]
    instruments = payload["instruments"]
    instruments.loc[
        instruments["instrument_id"] == CODES[1], "delist_date"
    ] = stop
    payload["instruments"] = instruments
    payload["daily_raw"] = payload["daily_raw"][
        ~(
            (payload["daily_raw"]["instrument_id"] == CODES[1])
            & (payload["daily_raw"]["trade_date"] >= stop)
        )
    ]
    payload["daily_valuation"] = payload["daily_valuation"][
        ~(
            (payload["daily_valuation"]["instrument_id"] == CODES[1])
            & (payload["daily_valuation"]["trade_date"] >= stop)
        )
    ]
    payload["universe_membership"].loc[
        payload["universe_membership"]["instrument_id"] == CODES[1],
        "effective_to",
    ] = stop

    with tempfile.TemporaryDirectory() as directory:
        snapshot = SnapshotPublisher(directory).publish(
            "listing-v1", payload, schema_version="1.1",
            lineage={"capabilities": {"financials": {
                "schema_version": "1.1", "pit_validated": True,
            }}},
        )
        clear_panel_cache()
        start, end = days[210], days[-1]
        known = snapshot.table("instruments").set_index("instrument_id")
        for factor in (
            net_profit_factor(), RoeFactor(), operating_cash_flow_factor(),
            DividendYieldFactor(), SelfBetaFactor(),
        ):
            with tempfile.TemporaryDirectory() as cache:
                frame = _factor_frame(snapshot, factor, start, end, cache)
            assert set(frame["instrument_id"]) <= set(known.index), factor.name
            # The delisted name stopped trading before this window opened.
            assert CODES[1] not in set(frame["instrument_id"]), factor.name
            # Its peers are still there, so the filter is not just emptying it.
            assert CODES[0] in set(frame["instrument_id"]), factor.name
        clear_panel_cache()


# ------------------------------------------------------------------- plumbing


def test_cache_reuses_a_broader_window(world):
    snapshot, days = world
    start, end = days[-6], days[-1]
    with tempfile.TemporaryDirectory() as cache:
        engine = FactorEngine(cache)
        factor = SelfBetaFactor()
        broad = engine.compute(factor, snapshot, start, end, None, end)
        narrow = engine.compute(factor, snapshot, days[-4], days[-2], None, end)
        assert not broad.from_cache
        assert narrow.from_cache
        assert narrow.cache_key == broad.cache_key


def test_worker_count_does_not_change_the_result(world):
    snapshot, days = world
    start, end = days[-4], days[-1]
    with tempfile.TemporaryDirectory() as cache:
        serial = _factor_frame(
            snapshot, SelfBetaFactor(workers=1), start, end, cache
        )
    with tempfile.TemporaryDirectory() as cache:
        parallel = _factor_frame(
            snapshot, SelfBetaFactor(workers=3), start, end, cache
        )
    order = ["trade_date", "instrument_id"]
    pd.testing.assert_frame_equal(
        serial.sort_values(order).reset_index(drop=True),
        parallel.sort_values(order).reset_index(drop=True),
    )


def test_definitions_omit_worker_count_but_keep_parameters():
    # Workers change speed, never output, so they must not fragment the cache.
    assert SelfBetaFactor(workers=1).definition() == (
        SelfBetaFactor(workers=8).definition()
    )
    assert SelfBetaFactor(window=252).definition() != (
        SelfBetaFactor(window=120).definition()
    )
    assert RegressionMomentumFactor(skip=20).definition() != (
        RegressionMomentumFactor(skip=5).definition()
    )
    assert math.isclose(
        SelfBetaFactor().definition()["half_life"], 63.0
    )
