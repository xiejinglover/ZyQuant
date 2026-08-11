from __future__ import annotations

import tempfile
import unittest
from datetime import date

from zyquant.backtest import BacktestEngine, StrategyBinding
from zyquant.backtest.types import PositionLot, SleeveAccount
from zyquant.config import ExecutionConfig
from zyquant.data import SnapshotPublisher
from zyquant.portfolio import (
    ConstraintEngine, PortfolioConstraints, TopKEqualWeightConstructor,
)
from zyquant.strategy import (
    DailySchedule, ExternalSignalGenerator, PipelineStrategy,
    StandardUniverseSelector,
)

from tests.support import CODE_A, CODE_B, canonical_tables, signal_frame


def strategy_for(signals, strategy_id="golden"):
    return PipelineStrategy(
        strategy_id, DailySchedule(),
        StandardUniverseSelector("TEST", median_amount_window=1),
        ExternalSignalGenerator(signals),
        TopKEqualWeightConstructor(1),
        ConstraintEngine(PortfolioConstraints()),
    )


class GoldenMarketTests(unittest.TestCase):
    def test_pause_rejection_then_recovery_and_historical_minimum_commission(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            mask = (
                (tables["daily_raw"]["trade_date"] == days[1])
                & (tables["daily_raw"]["instrument_id"] == CODE_A)
            )
            tables["daily_raw"].loc[mask, "paused"] = True
            snapshot = SnapshotPublisher(temporary).publish("paused-v1", tables)
            result = BacktestEngine(ExecutionConfig(
                max_participation=1, slippage_bps=0, impact_coefficient_bps=0,
            )).run(
                snapshot, days[0], days[-1],
                [StrategyBinding(strategy_for(signal_frame(days)), 1.0)],
                100_000,
            )
            first = result.frames["fills"].iloc[0]
            self.assertEqual(first["status"], "rejected")
            self.assertEqual(first["reject_reason"], "paused_or_zero_volume")
            paid = result.frames["fills"][
                result.frames["fills"]["filled_quantity"] > 0
            ]
            self.assertTrue((paid["commission"] >= 5.0).all())
            self.assertTrue(
                (
                    result.frames["demand_residuals"]["reason"]
                    == "paused_or_zero_volume"
                ).any()
            )

    def test_t_plus_one_lots_and_cash_dividend_accounting(self):
        account = SleeveAccount("stock", 0.0)
        account.add_lot(PositionLot(
            CODE_B, 100, date(2025, 1, 2), date(2025, 1, 3), 8.0
        ))
        self.assertEqual(account.sellable_quantity(CODE_B, date(2025, 1, 2)), 0)
        self.assertEqual(account.sellable_quantity(CODE_B, date(2025, 1, 3)), 100)

        cohort_account = SleeveAccount("cohorts", 0.0)
        cohort_account.add_lot(PositionLot(
            CODE_B, 100, date(2025, 1, 2), date(2025, 1, 3), 8.0,
            "first",
        ))
        cohort_account.add_lot(PositionLot(
            CODE_B, 100, date(2025, 1, 3), date(2025, 1, 6), 8.1,
            "second",
        ))
        cohort_account.adjust_shares(CODE_B, 30, date(2025, 1, 6))
        self.assertEqual(cohort_account.quantity(CODE_B, "first"), 115)
        self.assertEqual(cohort_account.quantity(CODE_B, "second"), 115)

        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            always_a = signal_frame(days)
            always_a["score"] = (
                always_a["instrument_id"] == CODE_A
            ).astype(float)
            snapshot = SnapshotPublisher(temporary).publish("dividend-v1", tables)
            result = BacktestEngine(ExecutionConfig(
                max_participation=1, commission_bps=0, minimum_commission=0,
                stock_sell_tax_bps=0, slippage_bps=0, impact_coefficient_bps=0,
            )).run(
                snapshot, days[0], days[-1],
                [StrategyBinding(strategy_for(always_a), 1.0)], 100_000,
            )
            actions = result.frames["corporate_actions"]
            self.assertIn("dividend_receivable", set(actions["type"]))
            self.assertIn("dividend_paid", set(actions["type"]))
            dividend_flows = result.frames["cashflows"][
                result.frames["cashflows"]["flow_type"] == "dividend_paid"
            ]
            sleeve = dividend_flows[
                dividend_flows["account_id"] == "golden"
            ]["amount"].sum()
            master = dividend_flows[
                dividend_flows["account_id"] == "__master__"
            ]["amount"].sum()
            self.assertAlmostEqual(sleeve, master)


if __name__ == "__main__":
    unittest.main()
