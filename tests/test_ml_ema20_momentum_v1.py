from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from strategies.ml_ema20_momentum_v1.execution import HoldLimitUpCloseExecutor
from strategies.ml_ema20_momentum_v1.prediction import PredictionBook
from strategies.ml_ema20_momentum_v1.report import _open_cohorts
from strategies.ml_ema20_momentum_v1.strategy import Ema20MomentumStrategy
from strategies.ml_ema20_momentum_v1.universe import (
    Ema20UniversePanel,
    build_ema20_universe,
)
from zyquant.backtest.types import MasterOrder
from zyquant.config import ExecutionConfig
from zyquant.core.exceptions import StrategyError
from zyquant.strategy import PortfolioView, PositionView, StrategyState


CODES = ("600001.XSHG", "600002.XSHG", "000001.XSHE", "000002.XSHE")


class FakeSnapshot:
    def __init__(self, closes: dict[str, list[float]] | None = None):
        self.days = [item.date() for item in pd.bdate_range("2024-01-02", periods=135)]
        self.metadata = SimpleNamespace(
            dataset_id="ema-test-v1",
            fingerprint="data-fp",
            as_of_date=self.days[-1],
        )
        values = closes or {
            code: [10.0] * 125 + [11.0] + [10.5] * 9 for code in CODES
        }
        self._post = pd.DataFrame([
            {"trade_date": day, "instrument_id": code, "close_post": close}
            for code in CODES
            for day, close in zip(self.days, values[code], strict=True)
        ])
        self._raw = pd.DataFrame([
            {
                "trade_date": day,
                "instrument_id": code,
                "close": close,
                "amount": 100_000_000.0,
                "volume": 10_000_000.0,
                "paused": False,
            }
            for code in CODES
            for day, close in zip(self.days, values[code], strict=True)
        ])
        self._instruments = pd.DataFrame([
            {
                "instrument_id": code,
                "symbol": code.split(".")[0],
                "exchange": code.split(".")[1],
                "asset_type": "stock",
                "list_date": date(2020, 1, 1),
                "delist_date": None,
            }
            for code in CODES
        ])
        self._special = pd.DataFrame([
            {
                "instrument_id": code,
                "name": f"测试{index}",
                "known_at": self.days[0],
                "effective_from": self.days[0],
                "effective_to": None,
            }
            for index, code in enumerate(CODES)
        ])

    def table(self, name, **kwargs):
        del kwargs
        return {
            "trade_calendar": pd.DataFrame({"trade_date": self.days}),
            "instruments": self._instruments,
            "special_treatment": self._special,
        }[name].copy()

    def post_adjusted_bars(self, start, end, instruments, fields, cutoff):
        del fields, cutoff
        return self._post[
            self._post["trade_date"].between(start, end)
            & self._post["instrument_id"].isin(instruments)
        ].copy()

    def raw_bars(self, start, end, instruments, fields, cutoff):
        del fields, cutoff
        return self._raw[
            self._raw["trade_date"].between(start, end)
            & self._raw["instrument_id"].isin(instruments)
        ].copy()


def prediction_frame(snapshot: FakeSnapshot, day: date) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "signal_date": day,
            "instrument_id": code,
            "score": score,
            "model_id": "synthetic",
            "model_version": "1",
            "feature_cutoff": day,
            "train_cutoff": day,
            "dataset_id": snapshot.metadata.dataset_id,
            "data_fingerprint": snapshot.metadata.fingerprint,
            "feature_set_id": "phase1-test",
        }
        for code, score in zip(CODES, (1.0, 1.0, 0.8, 0.7), strict=True)
    ])


def test_ema20_cross_is_recursive_filtered_and_causal():
    snapshot = FakeSnapshot()
    cross_day = snapshot.days[125]
    panel = build_ema20_universe(snapshot, snapshot.days[0], snapshot.days[-1])
    assert panel.eligible(snapshot.days[19]) == ()
    assert panel.eligible(cross_day) == tuple(sorted(CODES))
    before_120 = panel.diagnostics.loc[
        panel.diagnostics["signal_date"] == snapshot.days[118]
    ].iloc[0]
    at_120 = panel.diagnostics.loc[
        panel.diagnostics["signal_date"] == snapshot.days[119]
    ].iloc[0]
    assert before_120["excluded_insufficient_listing"] == len(CODES)
    assert at_120["excluded_insufficient_listing"] == 0

    changed = FakeSnapshot()
    changed._post.loc[changed._post["trade_date"] > cross_day, "close_post"] = 99.0
    changed_panel = build_ema20_universe(changed, changed.days[0], changed.days[-1])
    assert changed_panel.eligible(cross_day) == panel.eligible(cross_day)

    snapshot._raw.loc[
        (snapshot._raw["trade_date"] == cross_day)
        & (snapshot._raw["instrument_id"] == CODES[0]),
        "paused",
    ] = True
    snapshot._special.loc[snapshot._special["instrument_id"] == CODES[1], "name"] = "*ST测试"
    filtered = build_ema20_universe(snapshot, snapshot.days[0], snapshot.days[-1])
    assert CODES[0] not in filtered.eligible(cross_day)
    assert CODES[1] not in filtered.eligible(cross_day)


def test_low_price_does_not_exclude_an_eligible_cross():
    values = {
        code: ([2.0] * 125 + [2.2] + [2.1] * 9) if code == CODES[0]
        else [10.0] * 125 + [11.0] + [10.5] * 9
        for code in CODES
    }
    snapshot = FakeSnapshot(values)
    panel = build_ema20_universe(snapshot, snapshot.days[0], snapshot.days[-1])
    assert CODES[0] in panel.eligible(snapshot.days[125])


def test_prediction_protocol_and_stable_ranking(tmp_path):
    snapshot = FakeSnapshot()
    day = snapshot.days[125]
    source = tmp_path / "predictions.parquet"
    prediction_frame(snapshot, day).to_parquet(source, index=False)
    book = PredictionBook.load(source, snapshot)
    ranked = book.on(day, tuple(reversed(CODES)))
    assert ranked["instrument_id"].tolist()[:2] == [CODES[0], CODES[1]]

    duplicate = pd.concat([prediction_frame(snapshot, day)] * 2, ignore_index=True)
    duplicate.to_parquet(source, index=False)
    with pytest.raises(StrategyError, match="duplicate"):
        PredictionBook.load(source, snapshot)


def _context(day, state, *, cash=1_000_000.0, positions=()):
    return SimpleNamespace(
        signal_date=day,
        execution_date=day,
        portfolio=PortfolioView(1_000_000.0, cash, positions),
        state=state,
        data=None,
        factor_engine=None,
    )


def test_top3_full_slots_skip_held_and_schedule_t_plus_one_two(tmp_path):
    snapshot = FakeSnapshot()
    day = snapshot.days[125]
    source = tmp_path / "predictions.parquet"
    prediction_frame(snapshot, day).to_parquet(source, index=False)
    strategy = Ema20MomentumStrategy(prediction_path=source)
    strategy.bind_calendar(snapshot.days)
    strategy._universe = Ema20UniversePanel(
        {day: CODES}, pd.DataFrame([{
            "signal_date": day, "base_pool_count": 4,
        }]), "universe-fp",
    )
    strategy._predictions = PredictionBook.load(source, snapshot)
    held = (PositionView(CODES[0], 10_000, 10_000, 10.0),)
    decision = strategy.decide(_context(day, StrategyState(), positions=held))
    entries = [item for item in decision.scheduled_targets if item.execution_phase == "open"]
    exits = [item for item in decision.scheduled_targets if item.execution_phase == "close"]
    assert list(entries[0].weights) == [CODES[1], CODES[2], CODES[3]]
    assert set(entries[0].weights.values()) == {0.15}
    assert entries[0].session_offset == 1
    assert exits[0].session_offset == 2
    assert exits[0].diagnostics["liquidate_only_instruments"] == [
        CODES[1], CODES[2], CODES[3],
    ]

    low_cash = strategy.decide(_context(day, StrategyState(), cash=160_000.0))
    low_entry = [item for item in low_cash.scheduled_targets if item.execution_phase == "open"]
    assert len(low_entry[0].weights) == 1


def test_exit_state_retries_without_maximum_holding_period(tmp_path):
    snapshot = FakeSnapshot()
    day = snapshot.days[127]
    source = tmp_path / "predictions.parquet"
    prediction_frame(snapshot, day).to_parquet(source, index=False)
    strategy = Ema20MomentumStrategy(prediction_path=source)
    strategy.bind_calendar(snapshot.days)
    strategy._universe = Ema20UniversePanel(
        {day: ()}, pd.DataFrame([{
            "signal_date": day, "base_pool_count": 0,
        }]), "universe-fp",
    )
    strategy._predictions = PredictionBook.load(source, snapshot)
    cohort = "old"
    state = StrategyState(payload={"cohorts": {cohort: {
        "signal_date": snapshot.days[124].isoformat(),
        "entry_date": snapshot.days[125].isoformat(),
        "first_exit_date": snapshot.days[126].isoformat(),
        "symbols": [CODES[0]],
        "status": "exit_pending",
        "retry_count": 9999,
    }}})
    held = (PositionView(CODES[0], 10_000, 10_000, 10.0),)
    decision = strategy.decide(_context(day, state, cash=900_000, positions=held))
    assert decision.next_state.payload["cohorts"][cohort]["retry_count"] == 10000
    retry = [item for item in decision.scheduled_targets if item.cohort_id == cohort]
    assert len(retry) == 1 and retry[0].session_offset == 1
    assert retry[0].diagnostics["liquidate_only_instruments"] == [CODES[0]]


def test_open_cohort_report_ignores_stale_state_without_position() -> None:
    states = pd.DataFrame([{
        "date": date(2025, 1, 3),
        "state_json": '{"cohorts":{"old":{"signal_date":"2025-01-01",'
        '"symbols":["600001.XSHG"],"status":"exit_pending"}}}',
    }])
    positions = pd.DataFrame(columns=[
        "date", "instrument_id", "market_value",
    ])
    fills = pd.DataFrame(columns=[
        "side", "reject_reason", "execution_date", "instrument_id",
    ])
    assert _open_cohorts(states, positions, fills).empty


def test_hold_limit_up_close_executor_only_blocks_required_sells():
    executor = HoldLimitUpCloseExecutor(ExecutionConfig(
        max_participation=1.0,
        commission_bps=0,
        minimum_commission=0,
        stock_sell_tax_bps=0,
        slippage_bps=0,
        impact_coefficient_bps=0,
    ))
    order = MasterOrder(
        "o", date(2025, 1, 2), "close", CODES[0], "sell", 1000, 11.0, 100,
    )
    market = SimpleNamespace(
        paused=False, volume=1_000_000, amount=10_000_000,
        open=11.0, high=11.0, low=11.0, close=11.0,
        limit_up=11.0, limit_down=9.0,
    )
    assert executor.execute(order, market, "stock").reject_reason == "strategy_hold_limit_up"
    market.limit_up = float("nan")
    assert (
        executor.execute(order, market, "stock").reject_reason
        == "missing_limit_up_for_hold_rule"
    )
    market.limit_up = 12.0
    assert executor.execute(order, market, "stock").status == "filled"
