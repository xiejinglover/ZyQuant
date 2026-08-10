from .ledger import LEDGER_COLUMNS, enforce_ledger_schemas
from .types import (
    BacktestResult, CashFlow, Fill, FillAllocation, InternalCross,
    MasterAccount, MasterOrder, PositionLot, SleeveAccount, SleeveDemand,
)


def __getattr__(name):
    if name in {"BacktestEngine", "StrategyBinding"}:
        from .engine import BacktestEngine, StrategyBinding
        return {"BacktestEngine": BacktestEngine, "StrategyBinding": StrategyBinding}[name]
    if name in {
        "CostModel", "ExecutionModel", "HistoricalCostModel", "MarketExecutor",
    }:
        from .market import (
            CostModel, ExecutionModel, HistoricalCostModel, MarketExecutor,
        )
        return {
            "CostModel": CostModel,
            "ExecutionModel": ExecutionModel,
            "HistoricalCostModel": HistoricalCostModel,
            "MarketExecutor": MarketExecutor,
        }[name]
    raise AttributeError(name)


__all__ = [
    "BacktestEngine", "StrategyBinding", "CostModel", "ExecutionModel",
    "HistoricalCostModel", "MarketExecutor", "BacktestResult",
    "MasterAccount", "SleeveAccount", "PositionLot", "SleeveDemand",
    "InternalCross", "MasterOrder", "Fill", "FillAllocation", "CashFlow",
    "LEDGER_COLUMNS", "enforce_ledger_schemas",
]
