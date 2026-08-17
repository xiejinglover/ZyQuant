from __future__ import annotations

import unittest
from datetime import date

from zyquant.backtest.types import MasterOrder, SleeveDemand
from zyquant.portfolio.sleeve import (
    allocate_fill_demands, allocate_fill_quantities, net_sleeve_demands,
)


class SleeveTests(unittest.TestCase):
    def test_fill_allocation_keeps_same_strategy_cohorts_separate(self):
        order = MasterOrder(
            "order", date(2024, 1, 2), "close", "X", "sell", 300,
            10.0, 100,
        )
        demands = [
            SleeveDemand(
                "strategy", "X", "sell", 100, 10.0, 100,
                "demand-a", "close", 0, "cohort-a",
            ),
            SleeveDemand(
                "strategy", "X", "sell", 200, 10.0, 100,
                "demand-b", "close", 0, "cohort-b",
            ),
        ]
        allocation = allocate_fill_demands(order, 300, demands)
        self.assertEqual(allocation[("strategy", "cohort-a", "demand-a")], 100)
        self.assertEqual(allocation[("strategy", "cohort-b", "demand-b")], 200)

    def test_internal_cross_and_deterministic_allocation(self):
        day = date(2025, 1, 2)
        demands = [
            SleeveDemand("a", "X", "sell", 1000, 10.0, 100),
            SleeveDemand("b", "X", "buy", 600, 10.0, 100),
            SleeveDemand("c", "X", "buy", 800, 10.0, 100),
        ]
        crosses, orders, residuals = net_sleeve_demands(demands, day, "open")
        self.assertEqual(sum(item.quantity for item in crosses), 1000)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, "buy")
        self.assertEqual(orders[0].quantity, 400)
        allocation = allocate_fill_quantities(
            orders[0], 200, residuals[("X", "buy")]
        )
        self.assertEqual(sum(allocation.values()), 200)


if __name__ == "__main__":
    unittest.main()
