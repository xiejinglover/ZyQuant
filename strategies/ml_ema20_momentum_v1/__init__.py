"""EMA20 cross momentum strategy driven by externally materialized predictions."""

from .execution import HoldLimitUpCloseExecutor, create_execution_model
from .factors import MomentumTechnicalFactor, momentum_factor_catalog
from .plugin import create_strategy
from .strategy import Ema20MomentumStrategy
from .universe import Ema20UniversePanel, build_ema20_universe

__all__ = [
    "Ema20MomentumStrategy", "Ema20UniversePanel",
    "HoldLimitUpCloseExecutor", "build_ema20_universe",
    "MomentumTechnicalFactor", "momentum_factor_catalog",
    "create_execution_model", "create_strategy",
]
