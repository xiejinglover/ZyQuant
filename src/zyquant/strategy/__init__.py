from .schedule import (
    DailySchedule, EveryNTradingDays, ExplicitDateSchedule, MonthlySchedule,
    WeeklySchedule,
)
from .signal import (
    ExternalSignalGenerator, FactorSignalGenerator, PredictionSignalGenerator,
)
from .types import (
    CandidateWeights, PortfolioConstructor,
    PortfolioView, PositionView, PreparableStrategy, RebalanceSchedule,
    SignalFrame, SignalGenerator, Strategy, StrategyContext, StrategyDecision,
    ScheduledTargetPortfolio, StrategyState, TargetPortfolio, UniverseSelector,
    UniverseSnapshot,
)
from .universe import StandardUniverseSelector


def __getattr__(name):
    if name in {"PipelineStrategy", "DirectTargetStrategy"}:
        from .pipeline import DirectTargetStrategy, PipelineStrategy
        return {
            "PipelineStrategy": PipelineStrategy,
            "DirectTargetStrategy": DirectTargetStrategy,
        }[name]
    raise AttributeError(name)


__all__ = [
    "DailySchedule", "EveryNTradingDays", "ExplicitDateSchedule",
    "MonthlySchedule", "WeeklySchedule", "ExternalSignalGenerator",
    "FactorSignalGenerator", "PredictionSignalGenerator",
    "CandidateWeights",
    "PortfolioConstructor", "PortfolioView", "PositionView", "RebalanceSchedule",
    "SignalFrame", "SignalGenerator", "StrategyContext", "StrategyDecision",
    "StrategyState", "Strategy", "PreparableStrategy", "TargetPortfolio",
    "ScheduledTargetPortfolio", "UniverseSelector",
    "UniverseSnapshot",
    "StandardUniverseSelector", "PipelineStrategy", "DirectTargetStrategy",
]
