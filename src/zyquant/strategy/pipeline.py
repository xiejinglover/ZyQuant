from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from zyquant.core.exceptions import (
    ConstraintError, ConstraintProtocolError, StrategyError,
)
from zyquant.core.hashing import hash_payload
from zyquant.portfolio.constraints import ConstraintPolicy

from .types import (
    CandidateWeights, PortfolioConstructor, RebalanceSchedule, SignalGenerator,
    StrategyContext, StrategyDecision, TargetPortfolio, UniverseSelector,
)


@dataclass
class PipelineStrategy:
    strategy_id: str
    schedule: RebalanceSchedule
    universe_selector: UniverseSelector
    signal_generator: SignalGenerator
    constructor: PortfolioConstructor
    constraint_engine: ConstraintPolicy
    no_candidate_policy: str = "hold_previous"
    constraint_failure_policy: str = "fail"

    def decide(self, context: StrategyContext) -> StrategyDecision:
        universe = self.universe_selector.select(context)
        bound = replace(context, universe=universe)
        signals = self.signal_generator.generate(bound, universe)
        if signals.frame.empty:
            return self._fallback(bound, universe, signals, "no_signals")
        candidate, next_state = self.constructor.construct(signals, bound, context.state)
        if not candidate.weights:
            return self._fallback(bound, universe, signals, "no_candidates", next_state)
        industries = self._industry_map(bound)
        asset_types = self._asset_type_map(bound)
        try:
            constrained, report = self.constraint_engine.apply(
                candidate, set(universe.eligible), context.portfolio.current_weights,
                industries, asset_types,
            )
        except ConstraintProtocolError:
            raise
        except ConstraintError as exc:
            if self.constraint_failure_policy == "fail":
                raise
            return self._fallback(
                bound, universe, signals,
                f"constraint_infeasible:{exc}", next_state,
                policy=self.constraint_failure_policy,
            )
        before_hash = hash_payload(context.state)
        after_hash = hash_payload(next_state)
        target = TargetPortfolio(
            self.strategy_id, context.signal_date, context.execution_date,
            context.execution_phase, dict(sorted(constrained.weights.items())),
            constrained.cash_weight, universe.fingerprint, signals.fingerprint,
            before_hash, after_hash,
            {"constructor": candidate.constructor_id, "constraint_report": report},
        )
        self._validate_target(target, set(universe.eligible))
        return StrategyDecision(
            target, next_state, signals,
            {
                "universe_fingerprint": universe.fingerprint,
                "eligible": list(universe.eligible),
                "excluded": universe.excluded.to_dict("records"),
                "constraint_report": report,
            },
        )

    def _fallback(
        self, context, universe, signals, reason, state=None, policy=None
    ):
        state = state or context.state
        selected_policy = policy or self.no_candidate_policy
        if selected_policy == "fail":
            raise StrategyError(f"{self.strategy_id}: {reason}")
        if selected_policy in {"hold_previous", "skip"} and context.previous_target:
            weights = context.previous_target.weights
            cash = context.previous_target.cash_weight
        elif selected_policy == "skip":
            return StrategyDecision(
                None, state, signals,
                {
                    "reason": reason,
                    "universe_fingerprint": universe.fingerprint,
                    "eligible": list(universe.eligible),
                    "excluded": universe.excluded.to_dict("records"),
                },
            )
        else:
            weights, cash = {}, 1.0
        target = TargetPortfolio(
            self.strategy_id, context.signal_date, context.execution_date,
            context.execution_phase, weights, cash, universe.fingerprint,
            signals.fingerprint if signals else "none", hash_payload(context.state),
            hash_payload(state), {"fallback": reason},
        )
        return StrategyDecision(
            target, state, signals,
            {
                "reason": reason,
                "universe_fingerprint": universe.fingerprint,
                "eligible": list(universe.eligible),
                "excluded": universe.excluded.to_dict("records"),
            },
        )

    @staticmethod
    def _industry_map(context):
        frame = context.data.table(
            "industry_membership", cutoff=context.cutoff
        )
        if frame.empty:
            return {}
        frame = frame[
            (frame["effective_from"] <= context.signal_date)
            & (frame["effective_to"].isna() | (frame["effective_to"] >= context.signal_date))
            & (frame["known_at"] <= context.cutoff)
        ]
        return dict(zip(frame["instrument_id"].astype(str), frame["industry_id"].astype(str)))

    @staticmethod
    def _asset_type_map(context):
        frame = context.data.table("instruments")
        return dict(zip(
            frame["instrument_id"].astype(str), frame["asset_type"].astype(str)
        ))

    @staticmethod
    def _validate_target(target, eligible):
        if set(target.weights) - eligible:
            raise StrategyError("target contains instruments outside universe")
        if any(value < 0 for value in target.weights.values()):
            raise StrategyError("target contains negative weights")
        if abs(sum(target.weights.values()) + target.cash_weight - 1.0) > 1e-12:
            raise StrategyError("target weights and cash must sum to one")


@dataclass
class DirectTargetStrategy:
    strategy_id: str
    schedule: RebalanceSchedule
    target_generator: Callable[[StrategyContext], CandidateWeights]
    universe_selector: UniverseSelector
    constraint_engine: ConstraintPolicy

    def decide(self, context: StrategyContext) -> StrategyDecision:
        universe = self.universe_selector.select(context)
        bound = replace(context, universe=universe)
        candidate = self.target_generator(bound)
        constrained, report = self.constraint_engine.apply(
            candidate, set(universe.eligible), context.portfolio.current_weights,
            PipelineStrategy._industry_map(bound),
            PipelineStrategy._asset_type_map(bound),
        )
        target = TargetPortfolio(
            self.strategy_id, context.signal_date, context.execution_date,
            context.execution_phase, dict(sorted(constrained.weights.items())),
            constrained.cash_weight, universe.fingerprint, "direct",
            hash_payload(context.state), hash_payload(context.state),
            {"constraint_report": report},
        )
        PipelineStrategy._validate_target(target, set(universe.eligible))
        return StrategyDecision(
            target, context.state, None,
            {
                "universe_fingerprint": universe.fingerprint,
                "eligible": list(universe.eligible),
                "excluded": universe.excluded.to_dict("records"),
                "constraint_report": report,
            },
        )
