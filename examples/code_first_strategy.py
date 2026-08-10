"""Minimal code-first strategy loaded with module:factory syntax."""

from __future__ import annotations

from zyquant.core import hash_payload
from zyquant.strategy import (
    MonthlySchedule, StrategyDecision, StrategyState, TargetPortfolio,
)


class FirstNEqualWeightStrategy:
    def __init__(self, strategy_id: str = "first_n", stock_num: int = 10):
        self.strategy_id = strategy_id
        self.stock_num = int(stock_num)
        self.schedule = MonthlySchedule()

    def decide(self, context):
        instruments = context.data.table("instruments")
        selected = sorted(instruments["instrument_id"].astype(str))[:self.stock_num]
        weights = {
            code: 1.0 / len(selected) for code in selected
        } if selected else {}
        state_hash = hash_payload(context.state)
        target = TargetPortfolio(
            self.strategy_id,
            context.signal_date,
            context.execution_date,
            context.execution_phase,
            weights,
            0.0 if selected else 1.0,
            "all-instruments",
            "code-first",
            state_hash,
            state_hash,
        )
        return StrategyDecision(target, StrategyState(), None)


def create_strategy(**parameters):
    return FirstNEqualWeightStrategy(**parameters)
