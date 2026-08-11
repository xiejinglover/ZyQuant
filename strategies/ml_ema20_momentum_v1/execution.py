from __future__ import annotations

import pandas as pd

from zyquant.backtest.market import HistoricalCostModel, MarketExecutor
from zyquant.core.plugins import PluginMetadata


class HoldLimitUpCloseExecutor(MarketExecutor):
    """Keep close-phase sells while the security remains limit-up."""

    def __init__(self, config, cost_model=None, tolerance: float = 1e-8):
        super().__init__(config, cost_model or HistoricalCostModel())
        self.tolerance = float(tolerance)

    def execute(self, order, market_row, asset_type: str, rule=None):
        if order.side == "sell" and order.execution_phase == "close":
            if market_row is not None:
                paused = (
                    bool(market_row.paused)
                    if pd.notna(market_row.paused) else False
                )
                if not paused and float(market_row.volume) > 0:
                    if pd.isna(market_row.limit_up):
                        return self._rejected(
                            order, "missing_limit_up_for_hold_rule"
                        )
                    if (
                        float(market_row.close)
                        >= float(market_row.limit_up) - self.tolerance
                    ):
                        return self._rejected(order, "strategy_hold_limit_up")
        return super().execute(order, market_row, asset_type, rule)


def create_execution_model(**parameters):
    return HoldLimitUpCloseExecutor(**parameters)


create_execution_model.plugin_metadata = PluginMetadata(  # type: ignore[attr-defined]
    name="ml_ema20_hold_limit_up_v1",
    version="1.0.0",
    kind="execution_models",
    minimum_framework_version="2.0.0",
    deterministic=True,
)
