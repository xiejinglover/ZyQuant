"""Backtest engine scaling benchmark.

Run with: python benchmarks/benchmark_backtest.py
The benchmark reports a warning rather than failing because CI hardware varies.

What it guards: the engine used to rescan whole-range structures once per
trading day — the bar dictionary for that day's prices, and the corporate
action table for that day's events — which made a run quadratic in its length.
A single wall-clock number cannot catch that regressing, because a slower
machine and a quadratic engine look the same. Two lengths can: with per-day
work independent of the range, `seconds_per_day` is flat, so the ratio between
a long run and a short one stays near 1. It was about 2.4 before the fix.

The shipped sample dataset is nine sessions long and cannot show any of this,
so the benchmark publishes its own snapshot: wide enough that a day's
cross-section costs something, and carrying enough corporate actions that the
action path is actually exercised.

The strategy is deliberately trivial and never touches the snapshot. A pipeline
strategy would drag `StandardUniverseSelector` into the measurement, and that
selector re-reads the snapshot on every decision date over a window that grows
with the range — its own super-linear term, which would swamp the engine's.
"""

from __future__ import annotations

import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from zyquant.backtest import BacktestEngine, StrategyBinding
from zyquant.config import ExecutionConfig
from zyquant.core.hashing import hash_payload
from zyquant.data import SnapshotPublisher
from zyquant.strategy.schedule import EveryNTradingDays
from zyquant.strategy.types import StrategyDecision, TargetPortfolio

INSTRUMENTS = 400
SESSIONS = 760
ACTIONS = 4000
SHORT, LONG = 180, 720


def trading_days(start: date, count: int) -> list[date]:
    days, cursor = [], start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def tables(days: list[date], codes: list[str]) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(20260728)
    instruments = pd.DataFrame([
        {
            "instrument_id": code, "symbol": code.split(".")[0],
            "exchange": "XSHG", "asset_type": "stock",
            "list_date": days[0] - timedelta(days=2000), "delist_date": None,
            "lot_size": 100, "sell_delay_days": 1,
        }
        for code in codes
    ])
    paths = 20.0 * np.exp(
        np.cumsum(rng.normal(0.0003, 0.011, (len(codes), len(days))), axis=1)
    )
    previous = np.concatenate([paths[:, :1], paths[:, :-1]], axis=1)
    raw = pd.DataFrame({
        "trade_date": np.repeat(np.array(days, dtype=object), len(codes)),
        "instrument_id": np.tile(np.array(codes, dtype=object), len(days)),
        "open": (paths * 0.998).T.reshape(-1),
        "high": (paths * 1.01).T.reshape(-1),
        "low": (paths * 0.99).T.reshape(-1),
        "close": paths.T.reshape(-1),
        "pre_close": previous.T.reshape(-1),
        "volume": 40_000_000,
        "amount": (paths * 40_000_000).T.reshape(-1),
        "paused": False,
        "limit_up": (previous * 1.1).T.reshape(-1),
        "limit_down": (previous * 0.9).T.reshape(-1),
    })
    actions = pd.DataFrame([
        {
            "event_id": f"evt-{index:05d}",
            "instrument_id": codes[index % len(codes)],
            "event_type": "cash_dividend",
            "record_date": days[20 + index % (len(days) - 30)],
            "ex_date": days[21 + index % (len(days) - 30)],
            "pay_date": days[25 + index % (len(days) - 30)],
            "cash_per_share": 0.1, "share_ratio": 0.0, "status": "active",
            "announced_at": days[18 + index % (len(days) - 30)],
        }
        for index in range(ACTIONS)
    ])
    membership = pd.DataFrame([
        {
            "universe_id": "BENCH", "instrument_id": code,
            "effective_from": days[0], "effective_to": None, "known_at": days[0],
        }
        for code in codes
    ])
    industry = pd.DataFrame([
        {
            "classification": "BENCH", "industry_id": f"IND{index % 8}",
            "instrument_id": code, "effective_from": days[0],
            "effective_to": None, "known_at": days[0],
        }
        for index, code in enumerate(codes)
    ])
    rules = pd.DataFrame([{
        "rule_id": "XSHG-stock-v1", "exchange": "XSHG", "asset_type": "stock",
        "effective_from": days[0], "effective_to": None, "commission_bps": 2.5,
        "minimum_commission": 5.0, "sell_tax_bps": 5.0, "buy_tax_bps": 0.0,
        "transfer_fee_bps": 0.1, "currency": "CNY",
    }])
    return {
        "instruments": instruments, "trade_calendar": pd.DataFrame(
            [{"trade_date": day, "exchange": "XSHG"} for day in days]
        ),
        "daily_raw": raw, "corporate_actions": actions,
        "universe_membership": membership, "industry_membership": industry,
        "market_rules": rules,
    }


class _RotatingStrategy:
    """Minimal strategy: rebalances into a rotating slice of the universe.

    Satisfies the engine's whole contract — an id, a schedule, and `decide` —
    without reading the snapshot, so the timing reflects the engine alone.
    """

    strategy_id = "bench"

    def __init__(self, codes: list[str], holdings: int = 20):
        self.codes = codes
        self.holdings = holdings
        self.schedule = EveryNTradingDays(10)

    def decide(self, context):
        offset = (context.signal_date.toordinal() // 7) % len(self.codes)
        picked = [
            self.codes[(offset + step) % len(self.codes)]
            for step in range(self.holdings)
        ]
        weight = 0.95 / self.holdings
        weights = {code: weight for code in sorted(picked)}
        target = TargetPortfolio(
            self.strategy_id, context.signal_date, context.execution_date,
            context.execution_phase, weights, 1.0 - 0.95,
            "bench-universe", "bench-signal",
            hash_payload(context.state), hash_payload(context.state), {},
        )
        return StrategyDecision(target, context.state, None)


def measure(snapshot, codes, days, sessions: int) -> dict:
    window = days[:sessions]
    engine = BacktestEngine(ExecutionConfig(
        timing="next_open", max_participation=0.5, slippage_bps=10.0,
        impact_coefficient_bps=0.0, max_impact_bps=0.0,
    ))
    started = time.perf_counter()
    engine.run(
        snapshot, window[30], window[-1],
        [StrategyBinding(_RotatingStrategy(codes), 1.0)], 20_000_000.0, seed=7,
    )
    elapsed = time.perf_counter() - started
    return {"days": sessions - 30, "seconds": elapsed,
            "seconds_per_day": elapsed / (sessions - 30)}


def main():
    codes = [f"{600000 + index}.XSHG" for index in range(INSTRUMENTS)]
    days = trading_days(date(2022, 1, 3), SESSIONS)
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = SnapshotPublisher(Path(temporary)).publish(
            "benchmark-backtest-v1", tables(days, codes)
        )
        short = measure(snapshot, codes, days, SHORT)
        long = measure(snapshot, codes, days, LONG)
    ratio = long["seconds_per_day"] / short["seconds_per_day"]
    status = "PASS" if ratio <= 1.5 else "WARN"
    print({
        "instruments": INSTRUMENTS,
        "actions": ACTIONS,
        "short": short,
        "long": long,
        "seconds_per_day_ratio": ratio,
        "target": 1.5,
        "status": status,
    })


if __name__ == "__main__":
    main()
