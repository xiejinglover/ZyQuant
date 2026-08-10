from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np

from zyquant.core.exceptions import ConstraintError, ConstraintProtocolError
from zyquant.strategy.types import CandidateWeights


@dataclass(frozen=True)
class PortfolioConstraints:
    min_cash_weight: float = 0.0
    max_cash_weight: float = 1.0
    max_instrument_weight: float = 1.0
    max_asset_weights: Mapping[str, float] = field(default_factory=dict)
    max_industry_weight: float | None = None
    max_one_way_turnover: float | None = None
    min_target_weight: float = 0.0
    min_holdings: int = 0
    max_holdings: int | None = None


@dataclass(frozen=True)
class ConstraintReport:
    before: Mapping[str, float]
    after: Mapping[str, float]
    cash_weight: float
    one_way_turnover: float
    diagnostics: Mapping[str, object] = field(default_factory=dict)


class ConstraintPolicy(Protocol):
    def apply(
        self,
        candidate: CandidateWeights,
        eligible: set[str],
        current_weights: Mapping[str, float],
        industries: Mapping[str, str],
        asset_types: Mapping[str, str],
    ) -> tuple[CandidateWeights, ConstraintReport]: ...


class ConstraintEngine:
    def __init__(self, constraints: PortfolioConstraints):
        if not 0 <= constraints.min_cash_weight <= constraints.max_cash_weight <= 1:
            raise ValueError("cash constraints must satisfy 0 <= min <= max <= 1")
        if not 0 < constraints.max_instrument_weight <= 1:
            raise ValueError("max_instrument_weight must be in (0, 1]")
        if constraints.max_industry_weight is not None and not (
            0 < constraints.max_industry_weight <= 1
        ):
            raise ValueError("max_industry_weight must be in (0, 1]")
        if constraints.max_one_way_turnover is not None and (
            constraints.max_one_way_turnover <= 0
        ):
            raise ValueError("max_one_way_turnover must be positive")
        if constraints.max_holdings is not None and (
            constraints.max_holdings < constraints.min_holdings
        ):
            raise ValueError("max_holdings must not be below min_holdings")
        if any(not 0 <= value <= 1 for value in constraints.max_asset_weights.values()):
            raise ValueError("asset caps must be between zero and one")
        self.constraints = constraints

    def apply(
        self,
        candidate: CandidateWeights,
        eligible: set[str],
        current_weights: Mapping[str, float],
        industries: Mapping[str, str] | None = None,
        asset_types: Mapping[str, str] | None = None,
    ) -> tuple[CandidateWeights, ConstraintReport]:
        before = {str(key): float(value) for key, value in candidate.weights.items()}
        cash_before = float(candidate.cash_weight)
        self._validate_protocol(before, cash_before, eligible)
        industries = industries or {}
        asset_types = asset_types or {}
        if self.constraints.max_industry_weight is not None:
            missing = set(before) - set(industries)
            if missing:
                raise ConstraintError(f"missing industry classification: {sorted(missing)}")
        if self.constraints.max_asset_weights:
            missing = set(before) - set(asset_types)
            if missing:
                raise ConstraintError(f"missing asset type: {sorted(missing)}")

        changes: list[dict[str, object]] = []
        diagnostics: dict[str, object] = {"changes": changes}
        weights = {code: value for code, value in before.items() if value > 0}
        budget = 1.0 - self.constraints.min_cash_weight
        invested = sum(weights.values())
        if invested > budget + 1e-12:
            scale = budget / invested
            weights = {code: value * scale for code, value in weights.items()}
            changes.append({
                "constraint": "min_cash", "action": "scale", "factor": scale,
            })
        target_invested = sum(weights.values())
        weights = self._redistribute_with_caps(
            weights, target_invested, industries, asset_types
        )

        if self.constraints.max_holdings is not None and (
            len(weights) > self.constraints.max_holdings
        ):
            keep = {
                code for code, _ in sorted(
                    weights.items(), key=lambda item: (-item[1], item[0])
                )[: self.constraints.max_holdings]
            }
            removed = {code: value for code, value in weights.items() if code not in keep}
            weights = {code: value for code, value in weights.items() if code in keep}
            changes.append({
                "constraint": "max_holdings", "removed": removed,
            })

        removed_small = {
            code: value for code, value in weights.items()
            if value < self.constraints.min_target_weight
        }
        if removed_small:
            weights = {
                code: value for code, value in weights.items()
                if value >= self.constraints.min_target_weight
            }
            changes.append({
                "constraint": "min_target_weight", "removed": removed_small,
            })

        turnover = self._turnover(weights, current_weights)
        limit = self.constraints.max_one_way_turnover
        if limit is not None and turnover > limit:
            alpha = limit / turnover
            union = set(weights) | set(current_weights)
            interpolated = {
                code: float(current_weights.get(code, 0.0))
                + alpha * (
                    weights.get(code, 0.0) - float(current_weights.get(code, 0.0))
                )
                for code in union
                if code in eligible
            }
            interpolated = {
                code: value for code, value in interpolated.items() if value > 1e-15
            }
            if self._hard_valid(interpolated, industries, asset_types):
                weights = interpolated
                turnover = self._turnover(weights, current_weights)
                changes.append({
                    "constraint": "turnover", "action": "interpolate", "alpha": alpha,
                })
            else:
                diagnostics["turnover_constraint_overridden"] = (
                    "hard_constraint_repair"
                )

        if not self._hard_valid(weights, industries, asset_types):
            raise ConstraintError("candidate portfolio cannot satisfy hard constraints")
        if len(weights) < self.constraints.min_holdings:
            diagnostics["min_holdings_unmet"] = {
                "required": self.constraints.min_holdings, "actual": len(weights),
            }
        cash = 1.0 - sum(weights.values())
        if cash > self.constraints.max_cash_weight + 1e-12:
            raise ConstraintError(
                "hard constraints leave cash above max_cash_weight; portfolio is infeasible"
            )
        result = CandidateWeights(
            dict(sorted(weights.items())), cash, candidate.constructor_id,
            candidate.diagnostics,
        )
        return result, ConstraintReport(before, result.weights, cash, turnover, diagnostics)

    @staticmethod
    def _validate_protocol(weights, cash, eligible):
        unknown = set(weights) - eligible
        if unknown:
            raise ConstraintProtocolError(
                f"candidate contains instruments outside universe: {sorted(unknown)}"
            )
        if not np.isfinite(cash) or cash < -1e-12 or cash > 1 + 1e-12:
            raise ConstraintProtocolError(
                "candidate cash weight must be finite and within [0, 1]"
            )
        invalid = {
            code: value for code, value in weights.items()
            if not np.isfinite(value) or value < 0
        }
        if invalid:
            raise ConstraintProtocolError(f"candidate contains invalid weights: {invalid}")
        if abs(sum(weights.values()) + cash - 1.0) > 1e-10:
            raise ConstraintProtocolError("candidate weights and cash must sum to one")

    def _redistribute_with_caps(self, weights, target, industries, asset_types):
        result = {code: 0.0 for code in weights}
        remaining = target
        active = set(weights)
        for _ in range(max(1, len(weights) * 4)):
            if remaining <= 1e-15 or not active:
                break
            base_total = sum(weights[code] for code in active)
            progressed = False
            for code in sorted(active):
                proposed = remaining * weights[code] / base_total
                room = self.constraints.max_instrument_weight - result[code]
                if self.constraints.max_industry_weight is not None:
                    group = industries[code]
                    group_total = sum(
                        value for item, value in result.items()
                        if industries.get(item) == group
                    )
                    room = min(room, self.constraints.max_industry_weight - group_total)
                asset = asset_types.get(code)
                if asset in self.constraints.max_asset_weights:
                    asset_total = sum(
                        value for item, value in result.items()
                        if asset_types.get(item) == asset
                    )
                    room = min(
                        room, self.constraints.max_asset_weights[asset] - asset_total
                    )
                added = min(max(0.0, room), proposed)
                if added > 0:
                    result[code] += added
                    remaining -= added
                    progressed = True
                if room <= proposed + 1e-15:
                    active.discard(code)
            if not progressed:
                break
        return {code: value for code, value in result.items() if value > 1e-15}

    def _hard_valid(self, weights, industries, asset_types):
        if any(
            value < -1e-12
            or value > self.constraints.max_instrument_weight + 1e-12
            for value in weights.values()
        ):
            return False
        if sum(weights.values()) > 1 - self.constraints.min_cash_weight + 1e-12:
            return False
        if self.constraints.max_industry_weight is not None:
            totals: dict[str, float] = {}
            for code, value in weights.items():
                group = industries.get(code)
                if group is None:
                    return False
                totals[group] = totals.get(group, 0.0) + value
            if any(
                value > self.constraints.max_industry_weight + 1e-12
                for value in totals.values()
            ):
                return False
        for asset, cap in self.constraints.max_asset_weights.items():
            total = sum(
                value for code, value in weights.items()
                if asset_types.get(code) == asset
            )
            if total > cap + 1e-12:
                return False
        return True

    @staticmethod
    def _turnover(target, current):
        return 0.5 * sum(
            abs(target.get(code, 0.0) - current.get(code, 0.0))
            for code in set(target) | set(current)
        )
