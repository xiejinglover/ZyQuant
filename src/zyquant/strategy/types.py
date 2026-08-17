from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Mapping, Protocol

import numpy as np
import pandas as pd

from zyquant.data import DataSnapshot

if TYPE_CHECKING:
    from zyquant.factors import FactorService


JSONValue = Any


@dataclass(frozen=True)
class StrategyState:
    schema_version: str = "1.0"
    payload: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionView:
    instrument_id: str
    quantity: int
    sellable_quantity: int
    last_price: float
    cohort_quantities: Mapping[str, int] = field(default_factory=dict)
    cohort_sellable_quantities: Mapping[str, int] = field(default_factory=dict)
    position_status: str = "active"
    valuation_source: str = "market_close"
    last_observed_date: date | None = None
    stale_sessions: int = 0


@dataclass(frozen=True)
class PortfolioView:
    nav: float
    cash: float
    positions: tuple[PositionView, ...] = ()
    pending_orders: tuple[Any, ...] = ()

    @property
    def current_weights(self) -> dict[str, float]:
        if self.nav <= 0:
            return {}
        return {
            item.instrument_id: item.quantity * item.last_price / self.nav
            for item in self.positions if item.quantity and item.last_price > 0
        }


@dataclass(frozen=True)
class UniverseSnapshot:
    strategy_id: str
    signal_date: date
    eligible: tuple[str, ...]
    excluded: pd.DataFrame
    fingerprint: str


@dataclass(frozen=True)
class SignalFrame:
    frame: pd.DataFrame
    fingerprint: str


@dataclass(frozen=True)
class CandidateWeights:
    weights: Mapping[str, float]
    cash_weight: float
    constructor_id: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetPortfolio:
    strategy_id: str
    signal_date: date
    execution_date: date
    execution_phase: str
    weights: Mapping[str, float]
    cash_weight: float
    universe_fingerprint: str
    signal_fingerprint: str
    state_before_hash: str
    state_after_hash: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    cohort_id: str | None = None


@dataclass(frozen=True)
class ScheduledTargetPortfolio:
    """A cohort-scoped target to execute on a later session and phase.

    ``session_offset`` is counted from ``signal_date`` (1 means the next
    trading session).  Targets sharing a cohort operate only on lots carrying
    that cohort id, which allows an older cohort to exit at the close without
    touching a new cohort entered at the same day's open.
    """

    strategy_id: str
    signal_date: date
    session_offset: int
    execution_phase: str
    cohort_id: str
    weights: Mapping[str, float]
    cash_weight: float
    universe_fingerprint: str
    signal_fingerprint: str
    state_before_hash: str
    state_after_hash: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyContext:
    strategy_id: str
    signal_date: date
    cutoff: date
    execution_date: date
    execution_phase: str
    data: DataSnapshot
    universe: UniverseSnapshot | None
    portfolio: PortfolioView
    previous_target: TargetPortfolio | None
    state: StrategyState
    parameters: Mapping[str, Any]
    rng: np.random.Generator
    factor_engine: "FactorService | None" = None
    is_bootstrap: bool = False

    @property
    def factor_service(self) -> "FactorService | None":
        """Typed alias; ``factor_engine`` remains for 1.x strategy source."""
        return self.factor_engine


@dataclass(frozen=True)
class StrategyDecision:
    target: TargetPortfolio | None
    next_state: StrategyState
    signals: SignalFrame | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    scheduled_targets: tuple[ScheduledTargetPortfolio, ...] = ()


class RebalanceSchedule(Protocol):
    def decision_dates(self, calendar: list[date]) -> list[date]: ...


class UniverseSelector(Protocol):
    def select(self, context: StrategyContext) -> UniverseSnapshot: ...


class SignalGenerator(Protocol):
    def generate(self, context: StrategyContext, universe: UniverseSnapshot) -> SignalFrame: ...


class PortfolioConstructor(Protocol):
    def construct(
        self,
        signals: SignalFrame,
        context: StrategyContext,
        state: StrategyState,
    ) -> tuple[CandidateWeights, StrategyState]: ...


class Strategy(Protocol):
    strategy_id: str
    schedule: RebalanceSchedule

    def decide(self, context: StrategyContext) -> StrategyDecision: ...


class PreparableStrategy(Protocol):
    def prepare_run(
        self,
        snapshot: DataSnapshot,
        factor_service: "FactorService | None",
        start: date,
        end: date,
    ) -> None: ...
