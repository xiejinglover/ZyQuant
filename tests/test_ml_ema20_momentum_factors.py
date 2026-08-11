from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from strategies.ml_ema20_momentum_v1.factors import (
    EXCLUDED_CONSTANT_FEATURES,
    EXCLUDED_CROSS_SECTIONAL_FEATURES,
    FEATURE_NAMES,
    MomentumTechnicalFactor,
    clear_momentum_feature_cache,
    momentum_factor_catalog,
)
from zyquant.data import SnapshotPublisher
from zyquant.factors import FactorEngine

from tests.support import canonical_tables


SHARE = "600001.XSHG"
B_SHARE = "900901.XSHG"
BEIJING = "430001.XBEI"


def _snapshot(tmp_path):
    tables, _ = canonical_tables()
    days = [item.date() for item in pd.bdate_range("2023-01-03", periods=90)]
    instruments = pd.DataFrame([
        {
            "instrument_id": code,
            "symbol": code.split(".")[0],
            "exchange": code.split(".")[1],
            "asset_type": "stock",
            "list_date": date(2020, 1, 1),
            "delist_date": None,
            "lot_size": 100,
            "sell_delay_days": 1,
            "name": code,
            "currency": "CNY",
        }
        for code in (SHARE, B_SHARE, BEIJING)
    ])
    raw_rows = []
    for code_index, code in enumerate((SHARE, B_SHARE, BEIJING)):
        closes = 8.0 + code_index + np.exp(
            np.linspace(0, 0.25, len(days))
            + 0.015 * np.sin(np.arange(len(days)) / 3)
        )
        for index, (day, close) in enumerate(zip(days, closes, strict=True)):
            previous = closes[index - 1] if index else close
            paused = code == SHARE and index == 55
            volume = 0.0 if paused or (code == SHARE and index == 60) else 1_000_000 + index * 7_000
            raw_rows.append({
                "trade_date": day,
                "instrument_id": code,
                "open": float(close * (0.998 + 0.001 * (index % 3))),
                "high": float(close * 1.012),
                "low": float(close * 0.987),
                "close": float(close),
                "pre_close": float(previous),
                "volume": volume,
                "amount": float(volume * close),
                "paused": paused,
                "limit_up": float(previous * 1.1),
                "limit_down": float(previous * 0.9),
            })
    tables["instruments"] = instruments
    tables["trade_calendar"] = pd.DataFrame([
        {"trade_date": day, "exchange": exchange}
        for day in days for exchange in ("XSHG", "XBEI")
    ])
    tables["daily_raw"] = pd.DataFrame(raw_rows)
    tables["corporate_actions"] = tables["corporate_actions"].iloc[0:0]
    tables["universe_membership"] = pd.DataFrame([
        {
            "universe_id": "TEST",
            "instrument_id": code,
            "effective_from": days[0],
            "effective_to": None,
            "known_at": days[0],
        }
        for code in (SHARE, B_SHARE, BEIJING)
    ])
    tables["industry_membership"] = pd.DataFrame([
        {
            "classification": "TEST",
            "industry_id": "IND",
            "instrument_id": code,
            "effective_from": days[0],
            "effective_to": None,
            "known_at": days[0],
        }
        for code in (SHARE, B_SHARE, BEIJING)
    ])
    stock_rule = tables["market_rules"].query("asset_type == 'stock'").iloc[0].to_dict()
    stock_rule["effective_from"] = days[0]
    beijing_rule = dict(stock_rule)
    beijing_rule.update({
        "rule_id": "XBEI-stock-v1",
        "exchange": "XBEI",
    })
    tables["market_rules"] = pd.DataFrame([stock_rule, beijing_rule])
    return SnapshotPublisher(tmp_path).publish("ema-factor-test-v1", tables), days


def test_catalog_contains_exactly_49_raw_features():
    catalog = momentum_factor_catalog()
    assert len(catalog) == len(FEATURE_NAMES) == 49
    assert set(catalog) == {f"ml_ema20_{name}" for name in FEATURE_NAMES}
    assert not set(EXCLUDED_CROSS_SECTIONAL_FEATURES) & set(FEATURE_NAMES)
    assert not set(EXCLUDED_CONSTANT_FEATURES) & set(FEATURE_NAMES)
    assert {factor.definition()["bar_policy"] for factor in catalog.values()} == {
        "active_bar", "current_bar",
    }


def test_ret20_filters_paused_but_keeps_nonpaused_zero_volume(tmp_path):
    snapshot, days = _snapshot(tmp_path)
    factor = MomentumTechnicalFactor("ret_20d")
    frame = FactorEngine(tmp_path / "cache").compute(
        factor, snapshot, days[0], days[-1], None, days[-1]
    ).frame
    assert set(frame["instrument_id"]) == {SHARE}
    assert days[55] not in set(frame["trade_date"])
    active = snapshot.post_adjusted_bars(
        days[0], days[-1], [SHARE], ["close_post"], days[-1]
    ).merge(
        snapshot.raw_bars(days[0], days[-1], [SHARE], ["paused"], days[-1]),
        on=["trade_date", "instrument_id"], validate="one_to_one",
    )
    active = active.loc[~active["paused"]].reset_index(drop=True)
    expected = active.iloc[-1]["close_post"] / active.iloc[-21]["close_post"] - 1
    actual = frame.loc[frame["trade_date"] == days[-1], "value"].iloc[0]
    assert actual == pytest.approx(expected, rel=1e-12)
    assert days[60] in set(frame["trade_date"])


def test_weighted_momentum_matches_reference_formula(tmp_path):
    snapshot, days = _snapshot(tmp_path)
    engine = FactorEngine(tmp_path / "cache")
    results = {
        name: engine.compute(
            MomentumTechnicalFactor(name), snapshot, days[-1], days[-1], None, days[-1]
        ).frame.iloc[0]["value"]
        for name in ("annualized_returns", "r2", "slope", "score")
    }
    joined = snapshot.post_adjusted_bars(
        days[0], days[-1], [SHARE], ["close_post"], days[-1]
    ).merge(
        snapshot.raw_bars(days[0], days[-1], [SHARE], ["paused"], days[-1]),
        on=["trade_date", "instrument_id"], validate="one_to_one",
    )
    prices = joined.loc[~joined["paused"], "close_post"].to_numpy()[-21:]
    y = np.log(prices)
    x = np.arange(21, dtype=float)
    weights = np.linspace(1.0, 2.0, 21)
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    annualized = math.exp(slope * 250) - 1
    prediction = slope * x + intercept
    residual = np.sum(weights * (y - prediction) ** 2)
    total = np.sum(weights * (y - np.mean(y)) ** 2)
    r2 = 1 - residual / total if total > 0 else 0.0
    assert results["slope"] == pytest.approx(slope, rel=1e-11)
    assert results["annualized_returns"] == pytest.approx(annualized, rel=1e-11)
    assert results["r2"] == pytest.approx(r2, rel=1e-11)
    assert results["score"] == pytest.approx(annualized * r2, rel=1e-11)


def test_macd_and_state_factors_are_window_independent(tmp_path):
    snapshot, days = _snapshot(tmp_path)
    for name in ("macd_norm_12_26", "score_ratio", "score_diff", "decay_days"):
        clear_momentum_feature_cache()
        wide = FactorEngine(tmp_path / f"wide-{name}").compute(
            MomentumTechnicalFactor(name), snapshot, days[20], days[-1], None, days[-1]
        ).frame
        clear_momentum_feature_cache()
        narrow = FactorEngine(tmp_path / f"narrow-{name}").compute(
            MomentumTechnicalFactor(name), snapshot, days[-5], days[-1], None, days[-1]
        ).frame
        common = wide.merge(
            narrow, on=["trade_date", "instrument_id"], suffixes=("_wide", "_narrow")
        )
        assert np.allclose(
            common["value_wide"], common["value_narrow"], equal_nan=True,
            rtol=1e-12, atol=1e-12,
        )


def test_all_49_factors_are_finite_and_exclude_beijing_and_b_shares(tmp_path):
    snapshot, days = _snapshot(tmp_path)
    engine = FactorEngine(tmp_path / "all-cache")
    for factor in momentum_factor_catalog().values():
        result = engine.compute(
            factor, snapshot, days[-1], days[-1], None, days[-1]
        ).frame
        assert set(result["instrument_id"]) == {SHARE}
        values = pd.to_numeric(result["value"], errors="coerce").dropna()
        assert np.isfinite(values).all(), factor.name


def test_constant_price_has_zero_r2_without_division_warnings():
    from strategies.ml_ema20_momentum_v1.factors import _one_instrument

    count = 70
    frame = pd.DataFrame({
        "trade_date": [item.date() for item in pd.bdate_range("2024-01-02", periods=count)],
        "instrument_id": SHARE,
        "open_post": 10.0,
        "high_post": 10.0,
        "low_post": 10.0,
        "close_post": 10.0,
        "volume": 1_000_000.0,
        "amount": 10_000_000.0,
    })
    with np.errstate(all="raise"):
        result = _one_instrument(frame)
    assert result.iloc[-1]["r2"] == 0.0
    assert result.iloc[-1]["score"] == 0.0
    assert result.iloc[-1]["score_ratio"] == 1.0
