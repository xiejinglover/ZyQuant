from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from zyquant.core.exceptions import StrategyError
from zyquant.strategy.types import CandidateWeights, StrategyState


@dataclass(frozen=True)
class TopKEqualWeightConstructor:
    top_k: int
    investment_weight: float = 1.0

    def construct(self, signals, context, state):
        selected = signals.frame.head(self.top_k)["instrument_id"].tolist()
        if not selected:
            return CandidateWeights({}, 1.0, "topk_equal"), state
        weight = self.investment_weight / len(selected)
        return CandidateWeights(
            {code: weight for code in selected}, 1 - self.investment_weight,
            "topk_equal", {"selected": selected},
        ), state


@dataclass(frozen=True)
class TopKDropoutConstructor:
    top_k: int
    buffer: int = 5
    max_replacements: int | None = None
    investment_weight: float = 1.0

    def construct(self, signals, context, state):
        ranked = signals.frame["instrument_id"].astype(str).tolist()
        previous = [str(item) for item in state.payload.get("selected", [])]
        retained = [code for code in previous if code in ranked[: self.top_k + self.buffer]]
        limit = self.max_replacements if self.max_replacements is not None else self.top_k
        additions = []
        for code in ranked:
            if code not in retained and len(additions) < min(limit, self.top_k - len(retained)):
                additions.append(code)
        selected = (retained + additions)[: self.top_k]
        if len(selected) < self.top_k:
            for code in ranked:
                if code not in selected:
                    selected.append(code)
                if len(selected) == self.top_k:
                    break
        next_state = StrategyState(state.schema_version, {**state.payload, "selected": selected})
        weight = self.investment_weight / len(selected) if selected else 0.0
        return CandidateWeights(
            {code: weight for code in selected}, 1 - self.investment_weight if selected else 1.0,
            "topk_dropout", {"retained": retained, "added": additions},
        ), next_state


@dataclass(frozen=True)
class ScoreWeightedConstructor:
    top_k: int | None = None
    method: str = "positive"
    temperature: float = 1.0
    investment_weight: float = 1.0

    def construct(self, signals, context, state):
        frame = signals.frame.head(self.top_k).copy() if self.top_k else signals.frame.copy()
        scores = frame["score"].to_numpy(dtype=float)
        if self.method == "positive":
            raw = np.maximum(scores, 0.0)
        elif self.method == "rank":
            raw = np.arange(len(scores), 0, -1, dtype=float)
        elif self.method == "softmax":
            shifted = (scores - scores.max()) / self.temperature
            raw = np.exp(shifted)
        else:
            raise StrategyError(f"unsupported score weighting method: {self.method}")
        if not len(raw) or raw.sum() <= 0:
            return CandidateWeights({}, 1.0, "score_weighted"), state
        normalized = raw / raw.sum() * self.investment_weight
        weights = dict(zip(frame["instrument_id"].astype(str), normalized, strict=True))
        return CandidateWeights(weights, 1 - self.investment_weight, "score_weighted"), state


@dataclass(frozen=True)
class RiskParityConstructor:
    lookback: int = 60
    top_k: int | None = None
    investment_weight: float = 1.0

    def construct(self, signals, context, state):
        codes = signals.frame.head(self.top_k)["instrument_id"].astype(str).tolist() if self.top_k else signals.frame["instrument_id"].astype(str).tolist()
        observation_date = min(context.signal_date, context.cutoff)
        start = observation_date.replace(year=max(1900, observation_date.year - 2))
        bars = context.data.post_adjusted_bars(
            start, observation_date, codes, ["close_post"], context.cutoff
        )
        matrix = bars.pivot(index="trade_date", columns="instrument_id", values="close_post").pct_change().tail(self.lookback)
        matrix = matrix.dropna(axis=1, thresh=max(2, self.lookback // 2)).dropna()
        if matrix.shape[0] < 2 or matrix.shape[1] == 0:
            return CandidateWeights({}, 1.0, "risk_parity", {"reason": "no_covariance"}), state
        sample = matrix.cov().to_numpy(dtype=float)
        diagonal = np.diag(np.diag(sample))
        covariance = 0.9 * sample + 0.1 * diagonal
        weights = np.full(covariance.shape[0], 1.0 / covariance.shape[0])
        target = 1.0 / covariance.shape[0]
        for _ in range(1000):
            marginal = covariance @ weights
            variance = float(weights @ marginal)
            if variance <= 0 or not np.isfinite(variance):
                return CandidateWeights(
                    {}, 1.0, "risk_parity", {"reason": "invalid_covariance"}
                ), state
            contributions = weights * marginal / variance
            error = float(np.max(np.abs(contributions - target)))
            if error < 1e-8:
                break
            weights *= np.clip(target / np.maximum(contributions, 1e-12), 0.2, 5.0)
            weights = np.maximum(weights, 1e-12)
            weights /= weights.sum()
        weights *= self.investment_weight
        result = dict(zip(matrix.columns.astype(str), weights, strict=True))
        return CandidateWeights(
            result, 1 - self.investment_weight, "risk_parity",
            {"iterations_converged": error < 1e-8, "risk_contribution_error": error},
        ), state


@dataclass(frozen=True)
class ExternalTargetWeightsConstructor:
    targets: Mapping[str, float] | None = None
    source_id: str = "external"
    source_hash: str | None = None

    def construct(self, signals, context, state):
        weights = dict(self.targets or {})
        total = sum(float(value) for value in weights.values())
        if total > 1 + 1e-12:
            raise StrategyError("external target weights exceed one")
        return CandidateWeights(
            weights, 1.0 - total, "external_target",
            {"source_id": self.source_id, "source_hash": self.source_hash},
        ), state
