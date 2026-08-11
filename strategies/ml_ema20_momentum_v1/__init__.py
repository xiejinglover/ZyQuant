"""EMA20 cross momentum strategy driven by externally materialized predictions."""

from .execution import HoldLimitUpCloseExecutor, create_execution_model
from .dataset import (
    CROSS_SECTIONAL_FEATURES,
    LABEL_COLUMN,
    MODEL_FEATURES,
    rolling_year_folds,
)
from .factors import MomentumTechnicalFactor, momentum_factor_catalog
from .plugin import create_strategy
from .strategy import Ema20MomentumStrategy
from .universe import Ema20UniversePanel, build_ema20_universe

__all__ = [
    "Ema20MomentumStrategy", "Ema20UniversePanel",
    "HoldLimitUpCloseExecutor", "build_ema20_universe",
    "MomentumTechnicalFactor", "momentum_factor_catalog",
    "CROSS_SECTIONAL_FEATURES", "LABEL_COLUMN", "MODEL_FEATURES",
    "rolling_year_folds",
    "create_execution_model", "create_strategy",
]
