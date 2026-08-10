from __future__ import annotations

import tempfile
import unittest

from zyquant.backtest import BacktestEngine, StrategyBinding
from zyquant.config import ExecutionConfig
from zyquant.data import SnapshotPublisher
from zyquant.portfolio import ConstraintEngine, PortfolioConstraints, TopKEqualWeightConstructor
from zyquant.strategy import DailySchedule, ExternalSignalGenerator, PipelineStrategy, StandardUniverseSelector
from zyquant.strategy.types import StrategyDecision, TargetPortfolio

from tests.support import CODE_A, CODE_B, canonical_tables, signal_frame


class BacktestTests(unittest.TestCase):
    def test_liquidate_only_target_does_not_rebalance_other_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("sample-v1", tables)

            class Strategy:
                strategy_id = "liquidation"
                schedule = DailySchedule()

                def decide(self, context):
                    target = None
                    if context.signal_date == days[0]:
                        target = TargetPortfolio(
                            self.strategy_id, context.signal_date,
                            context.execution_date, context.execution_phase,
                            {CODE_A: 0.4, CODE_B: 0.4}, 0.2,
                            "u", "entry", "before", "after", {},
                        )
                    elif context.signal_date == days[1]:
                        # B's deliberately different target weight must be
                        # ignored by a liquidation-only instruction for A.
                        target = TargetPortfolio(
                            self.strategy_id, context.signal_date,
                            context.execution_date, context.execution_phase,
                            {CODE_B: 0.01}, 0.99,
                            "u", "exit", "before", "after",
                            {"liquidate_only_instruments": [CODE_A]},
                        )
                    return StrategyDecision(target, context.state, None)

            result = BacktestEngine(ExecutionConfig(
                max_participation=1.0, commission_bps=0,
                minimum_commission=0, stock_sell_tax_bps=0,
                slippage_bps=0, impact_coefficient_bps=0,
            )).run(
                snapshot, days[0], days[-1],
                [StrategyBinding(Strategy(), 1.0)], 1_000_000,
            )
            fills = result.frames["fills"]
            liquidation = fills[fills["execution_date"] == days[2]]
            self.assertEqual(set(liquidation["instrument_id"]), {CODE_A})
            self.assertEqual(set(liquidation["side"]), {"sell"})
            demands = result.frames["sleeve_demands"]
            liquidation_demand = demands[
                (demands["instrument_id"] == CODE_A)
                & (demands["side"] == "sell")
            ].iloc[-1]
            self.assertEqual(liquidation_demand["lot_size"], 1)

    def test_daily_backtest_uses_materialized_signals_and_raw_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("sample-v1", tables)
            strategy = PipelineStrategy(
                "alpha", DailySchedule(),
                StandardUniverseSelector("TEST", median_amount_window=1),
                ExternalSignalGenerator(signal_frame(days)),
                TopKEqualWeightConstructor(1),
                ConstraintEngine(PortfolioConstraints()),
            )
            engine = BacktestEngine(ExecutionConfig(
                max_participation=1.0, commission_bps=0,
                minimum_commission=0, stock_sell_tax_bps=0,
                slippage_bps=0, impact_coefficient_bps=0,
            ))
            result = engine.run(
                snapshot, days[0], days[-1], [StrategyBinding(strategy, 1.0)],
                1_000_000, seed=7, compute_attribution=True,
            )
            self.assertGreater(result.metrics["fills"], 0)
            fills = result.frames["fills"]
            first = fills[fills["filled_quantity"] > 0].iloc[0]
            self.assertEqual(first["instrument_id"], CODE_A)
            raw = snapshot.raw_bars(first["execution_date"], first["execution_date"], [CODE_A], cutoff=first["execution_date"])
            self.assertAlmostEqual(first["price"], raw.iloc[0]["open"])
            nav = result.frames["nav"]
            self.assertTrue((nav["cash"] >= -1e-8).all())
            attribution = result.frames["attribution"]
            pivot = attribution.pivot_table(index="date", columns="component", values="pnl", aggfunc="sum").fillna(0)
            self.assertTrue(((pivot["gross_pnl"] + pivot["actual_cost"] - pivot["account_pnl"]).abs() < 1e-8).all())

    def test_multi_strategy_sleeves_net_opposing_rebalances(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("sample-v1", tables)
            first_signals = signal_frame(days)
            second_signals = first_signals.copy()
            second_signals["score"] = 1.0 - second_signals["score"]
            common = dict(
                schedule=DailySchedule(),
                universe_selector=StandardUniverseSelector("TEST", median_amount_window=1),
                constructor=TopKEqualWeightConstructor(1),
                constraint_engine=ConstraintEngine(PortfolioConstraints()),
            )
            first = PipelineStrategy(
                strategy_id="first",
                signal_generator=ExternalSignalGenerator(first_signals),
                **common,
            )
            second = PipelineStrategy(
                strategy_id="second",
                signal_generator=ExternalSignalGenerator(second_signals),
                **common,
            )
            result = BacktestEngine(ExecutionConfig(
                max_participation=1, commission_bps=0, minimum_commission=0,
                stock_sell_tax_bps=0, slippage_bps=0, impact_coefficient_bps=0,
            )).run(
                snapshot, days[0], days[-1],
                [StrategyBinding(first, 0.5), StrategyBinding(second, 0.5)],
                1_000_000,
            )
            self.assertFalse(result.frames["internal_crosses"].empty)
            daily = result.frames["nav"].groupby("date")["nav"].sum()
            self.assertTrue((daily > 0).all())

    def test_same_close_uses_previous_day_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("sample-v1", tables)
            strategy = PipelineStrategy(
                "same-close", DailySchedule(),
                StandardUniverseSelector("TEST", median_amount_window=1),
                ExternalSignalGenerator(signal_frame(days)),
                TopKEqualWeightConstructor(1),
                ConstraintEngine(PortfolioConstraints()),
            )
            result = BacktestEngine(ExecutionConfig(
                timing="same_close", max_participation=1, commission_bps=0,
                minimum_commission=0, stock_sell_tax_bps=0,
                slippage_bps=0, impact_coefficient_bps=0,
            )).run(snapshot, days[0], days[-1], [StrategyBinding(strategy, 1)], 100_000)
            targets = result.frames["targets"]
            self.assertTrue((targets["signal_date"] == targets["execution_date"]).all())
            fills = result.frames["fills"]
            first = fills[fills["filled_quantity"] > 0].iloc[0]
            raw = snapshot.raw_bars(
                first["execution_date"], first["execution_date"],
                [first["instrument_id"]], cutoff=first["execution_date"],
            )
            self.assertAlmostEqual(first["price"], raw.iloc[0]["close"])


if __name__ == "__main__":
    unittest.main()
