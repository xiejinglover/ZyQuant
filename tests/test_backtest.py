from __future__ import annotations

import tempfile
import unittest

from zyquant.backtest import BacktestEngine, StrategyBinding
from zyquant.config import ExecutionConfig
from zyquant.data import SnapshotPublisher
from zyquant.portfolio import ConstraintEngine, PortfolioConstraints, TopKEqualWeightConstructor
from zyquant.strategy import (
    DailySchedule, ExplicitDateSchedule, ExternalSignalGenerator,
    PipelineStrategy, ScheduledTargetPortfolio, StandardUniverseSelector,
)
from zyquant.strategy.types import StrategyDecision, TargetPortfolio

from tests.support import CODE_A, CODE_B, canonical_tables, signal_frame


class BacktestTests(unittest.TestCase):
    def test_scheduled_cohort_is_skipped_when_exit_is_outside_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("boundary-v1", tables)

            class Strategy:
                strategy_id = "boundary"
                schedule = ExplicitDateSchedule((days[3],))

                def decide(self, context):
                    common = dict(
                        strategy_id=self.strategy_id,
                        signal_date=context.signal_date,
                        cohort_id="last-cohort",
                        universe_fingerprint="u",
                        state_before_hash="before",
                        state_after_hash="after",
                    )
                    return StrategyDecision(
                        None, context.state, None,
                        scheduled_targets=(
                            ScheduledTargetPortfolio(
                                session_offset=1, execution_phase="open",
                                weights={CODE_B: 0.5}, cash_weight=0.5,
                                signal_fingerprint="entry", **common,
                            ),
                            ScheduledTargetPortfolio(
                                session_offset=2, execution_phase="close",
                                weights={}, cash_weight=1.0,
                                signal_fingerprint="exit", **common,
                            ),
                        ),
                    )

            result = BacktestEngine(ExecutionConfig()).run(
                snapshot, days[0], days[4],
                [StrategyBinding(Strategy(), 1.0)], 1_000_000,
            )
            self.assertTrue(result.frames["fills"].empty)
            residuals = result.frames["scheduled_target_residuals"]
            self.assertEqual(len(residuals), 1)
            self.assertEqual(residuals.iloc[0]["reason"], "outside_backtest_range")

    def test_scheduled_cohorts_enter_next_open_and_exit_following_close(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("cohort-v1", tables)

            class Strategy:
                strategy_id = "overnight"
                schedule = ExplicitDateSchedule((days[0], days[1]))

                def decide(self, context):
                    cohort = context.signal_date.isoformat()
                    common = dict(
                        strategy_id=self.strategy_id,
                        signal_date=context.signal_date,
                        cohort_id=cohort,
                        universe_fingerprint="u",
                        state_before_hash="before",
                        state_after_hash="after",
                    )
                    entry = ScheduledTargetPortfolio(
                        session_offset=1, execution_phase="open",
                        weights={CODE_B: 0.4}, cash_weight=0.6,
                        signal_fingerprint="entry", **common,
                    )
                    exit_ = ScheduledTargetPortfolio(
                        session_offset=2, execution_phase="close",
                        weights={}, cash_weight=1.0,
                        signal_fingerprint="exit", **common,
                    )
                    return StrategyDecision(
                        None, context.state, None,
                        scheduled_targets=(entry, exit_),
                    )

            result = BacktestEngine(ExecutionConfig(
                timing="next_open", max_participation=1.0,
                commission_bps=0, minimum_commission=0,
                stock_sell_tax_bps=0, slippage_bps=0,
                impact_coefficient_bps=0,
            )).run(
                snapshot, days[0], days[4],
                [StrategyBinding(Strategy(), 1.0)], 1_000_000,
            )
            fills = result.frames["fills"]
            executed = fills[fills["filled_quantity"] > 0]
            first_entry = executed[
                (executed["execution_date"] == days[1])
                & (executed["side"] == "buy")
            ].iloc[0]
            second_entry = executed[
                (executed["execution_date"] == days[2])
                & (executed["execution_phase"] == "open")
                & (executed["side"] == "buy")
            ].iloc[0]
            first_exit = executed[
                (executed["execution_date"] == days[2])
                & (executed["execution_phase"] == "close")
                & (executed["side"] == "sell")
            ].iloc[0]
            self.assertEqual(first_entry["execution_phase"], "open")
            self.assertEqual(first_entry["filled_quantity"], first_exit["filled_quantity"])
            self.assertGreater(second_entry["filled_quantity"], 0)

            allocations = result.frames["fill_allocations"]
            first_cohort = days[0].isoformat()
            close_allocations = allocations[
                (allocations["execution_date"] == days[2])
                & (allocations["side"] == "sell")
            ]
            self.assertEqual(set(close_allocations["cohort_id"]), {first_cohort})
            positions = result.frames["positions"]
            end_of_day = positions[
                (positions["date"] == days[2])
                & (positions["instrument_id"] == CODE_B)
            ]
            self.assertFalse(end_of_day.empty)
            self.assertEqual(
                int(end_of_day.iloc[0]["quantity"]),
                int(second_entry["filled_quantity"]),
            )

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
