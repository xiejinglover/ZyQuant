from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class PortfolioProblem:
    instrument_ids: tuple[str, ...]
    expected_returns: np.ndarray
    current_weights: np.ndarray
    covariance: np.ndarray | None
    transaction_costs: np.ndarray | None
    constraints: tuple[Any, ...]
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.0


@dataclass(frozen=True)
class OptimizerResult:
    weights: Mapping[str, float]
    status: str
    diagnostics: Mapping[str, Any]


class PortfolioOptimizer(Protocol):
    def solve(self, problem: PortfolioProblem) -> OptimizerResult: ...

