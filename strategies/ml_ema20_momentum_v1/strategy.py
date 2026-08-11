from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from zyquant.core.hashing import hash_payload
from zyquant.strategy import (
    DailySchedule, ScheduledTargetPortfolio, SignalFrame, StrategyDecision,
    StrategyState,
)

from .prediction import PredictionBook
from .universe import Ema20UniversePanel, build_ema20_universe


class _RunWindowSchedule:
    def __init__(self, owner: "Ema20MomentumStrategy"):
        self.owner = owner
        self.inner = DailySchedule()

    def decision_dates(self, calendar: Sequence[date]) -> list[date]:
        self.owner.bind_calendar(calendar)
        return self.inner.decision_dates(list(calendar))


class Ema20MomentumStrategy:
    """Daily EMA20 event strategy with cohort-scoped open/close legs."""

    def __init__(
        self,
        *,
        prediction_path: str | Path,
        strategy_id: str = "ml_ema20_momentum_v1",
        top_k: int = 3,
        instrument_weight: float = 0.15,
        cash_buffer: float = 0.01,
        ema_span: int = 20,
        minimum_listed_sessions: int = 120,
        minimum_price: float | None = None,
        minimum_median_amount: float = 50_000_000.0,
    ):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 < instrument_weight <= 1:
            raise ValueError("instrument_weight must be in (0, 1]")
        if not 0 <= cash_buffer < 1:
            raise ValueError("cash_buffer must be in [0, 1)")
        if top_k * instrument_weight > 1.0 - cash_buffer + 1e-12:
            raise ValueError("top_k weights exceed investable capital")
        self.strategy_id = strategy_id
        self.prediction_path = Path(prediction_path)
        self.top_k = int(top_k)
        self.instrument_weight = float(instrument_weight)
        self.cash_buffer = float(cash_buffer)
        self.ema_span = int(ema_span)
        self.minimum_listed_sessions = int(minimum_listed_sessions)
        self.minimum_price = (
            float(minimum_price) if minimum_price is not None else None
        )
        self.minimum_median_amount = float(minimum_median_amount)
        self.schedule = _RunWindowSchedule(self)
        self._calendar: list[date] = []
        self._calendar_index: dict[date, int] = {}
        self._universe: Ema20UniversePanel | None = None
        self._predictions: PredictionBook | None = None
        self.planned: list[dict[str, Any]] = []

    @property
    def factor_provenance(self) -> dict[str, Any]:
        if self._universe is None or self._predictions is None:
            return {}
        return {
            "profile": "ml_ema20_momentum_v1_phase1",
            "ema_universe_fingerprint": self._universe.fingerprint,
            "prediction_fingerprint": self._predictions.fingerprint,
            "prediction_path": str(self._predictions.source_path),
        }

    def bind_calendar(self, calendar: Sequence[date]) -> None:
        self._calendar = list(calendar)
        self._calendar_index = {
            day: index for index, day in enumerate(self._calendar)
        }

    def prepare_run(self, snapshot, factor_engine, start: date, end: date) -> None:
        del factor_engine
        if not self._calendar:
            calendar = sorted(set(snapshot.table("trade_calendar")["trade_date"]))
            self.bind_calendar([day for day in calendar if start <= day <= end])
        self._universe = build_ema20_universe(
            snapshot, start, end,
            ema_span=self.ema_span,
            minimum_listed_sessions=self.minimum_listed_sessions,
            minimum_price=self.minimum_price,
            minimum_median_amount=self.minimum_median_amount,
        )
        self._predictions = PredictionBook.load(self.prediction_path, snapshot)

    def _next_day_exists(self, day: date, offset: int) -> bool:
        index = self._calendar_index.get(day)
        return index is not None and index + offset < len(self._calendar)

    @staticmethod
    def _held_codes(context) -> set[str]:
        return {
            str(position.instrument_id)
            for position in context.portfolio.positions
            if position.quantity > 0
        }

    def decide(self, context) -> StrategyDecision:
        if self._universe is None or self._predictions is None:
            fallback_start = self._calendar[0] if self._calendar else context.signal_date
            fallback_end = self._calendar[-1] if self._calendar else context.execution_date
            self.prepare_run(
                context.data, context.factor_engine,
                fallback_start, fallback_end,
            )
        assert self._universe is not None and self._predictions is not None
        day = context.signal_date
        held = self._held_codes(context)
        payload = dict(context.state.payload)
        raw_cohorts = dict(payload.get("cohorts", {}))
        cohorts: dict[str, dict[str, Any]] = {}
        scheduled: list[ScheduledTargetPortfolio] = []

        for cohort_id, raw in sorted(raw_cohorts.items()):
            item = dict(raw)
            symbols = [str(code) for code in item.get("symbols", [])]
            entry_date = date.fromisoformat(str(item["entry_date"]))
            first_exit_date = date.fromisoformat(str(item["first_exit_date"]))
            actual = sorted(set(symbols) & held)
            if day < entry_date:
                item["status"] = "entry_pending"
                cohorts[cohort_id] = item
                continue
            if day < first_exit_date:
                if actual:
                    item["symbols"] = actual
                    item["status"] = "holding"
                    cohorts[cohort_id] = item
                continue
            if not actual:
                continue
            item["symbols"] = actual
            item["status"] = "exit_pending"
            item["retry_count"] = int(item.get("retry_count", 0)) + 1
            cohorts[cohort_id] = item
            if self._next_day_exists(day, 1):
                scheduled.append(self._exit_target(
                    context, cohort_id, session_offset=1,
                    signal_fingerprint=f"retry-{item['retry_count']}",
                ))

        eligible = self._universe.eligible(day)
        ranked = self._predictions.on(day, eligible)
        held_ranked = ranked["instrument_id"].isin(held)
        available = ranked.loc[~held_ranked].copy()
        nav = float(context.portfolio.nav)
        cash_ratio = float(context.portfolio.cash) / nav if nav > 0 else 0.0
        slots = max(0, math.floor(
            (cash_ratio - self.cash_buffer + 1e-12) / self.instrument_weight
        ))
        slots = min(self.top_k, slots)
        selected = available.head(slots)["instrument_id"].astype(str).tolist()
        reason = None
        if not self._next_day_exists(day, 2):
            selected = []
            reason = "insufficient_future_sessions"
        elif slots == 0:
            reason = "insufficient_cash_for_full_slot"
        elif available.empty:
            reason = "no_scored_unheld_candidates"
        elif len(selected) < self.top_k:
            reason = "partial_candidate_or_cash_slots"

        if selected:
            cohort_id = f"{self.strategy_id}:{day.isoformat()}"
            entry_day = self._calendar[self._calendar_index[day] + 1]
            exit_day = self._calendar[self._calendar_index[day] + 2]
            cohorts[cohort_id] = {
                "signal_date": day.isoformat(),
                "entry_date": entry_day.isoformat(),
                "first_exit_date": exit_day.isoformat(),
                "original_candidates": selected,
                "symbols": selected,
                "status": "entry_pending",
                "retry_count": 0,
            }

        universe_day = self._universe.diagnostics[
            self._universe.diagnostics["signal_date"] == day
        ]
        universe_counts = {
            str(column): int(universe_day.iloc[0][column])
            for column in universe_day.columns
            if column != "signal_date" and not universe_day.empty
        }
        decision_diagnostics = {
            "signal_date": day.isoformat(),
            "base_pool_count": self._diagnostic_value(day, "base_pool_count"),
            "cross_count": len(eligible),
            "prediction_count": len(ranked),
            "prediction_coverage": (
                len(ranked) / len(eligible) if eligible else 0.0
            ),
            "held_candidates_skipped": int(held_ranked.sum()),
            "held_candidates": ranked.loc[
                held_ranked, "instrument_id"
            ].astype(str).tolist(),
            "cash_ratio": cash_ratio,
            "available_slots": slots,
            "selected": selected,
            "empty_slot_reason": reason,
            "active_cohorts": len(cohorts),
            "universe_exclusion_counts": universe_counts,
            "universe_fingerprint": self._universe.fingerprint,
            "prediction_fingerprint": self._predictions.fingerprint,
        }
        payload["cohorts"] = cohorts
        payload["last_decision"] = decision_diagnostics
        next_state = StrategyState(context.state.schema_version, payload)
        before_hash = hash_payload(context.state)
        after_hash = hash_payload(next_state)

        if selected:
            cohort_id = f"{self.strategy_id}:{day.isoformat()}"
            weights = {code: self.instrument_weight for code in selected}
            common = dict(
                strategy_id=self.strategy_id,
                signal_date=day,
                cohort_id=cohort_id,
                universe_fingerprint=self._universe.fingerprint,
                state_before_hash=before_hash,
                state_after_hash=after_hash,
                diagnostics={"kind": "entry", **decision_diagnostics},
            )
            scheduled.extend([
                ScheduledTargetPortfolio(
                    session_offset=1, execution_phase="open",
                    weights=weights, cash_weight=1.0 - sum(weights.values()),
                    signal_fingerprint=self._predictions.fingerprint,
                    **common,
                ),
                ScheduledTargetPortfolio(
                    session_offset=2, execution_phase="close",
                    weights={}, cash_weight=1.0,
                    signal_fingerprint=f"{self._predictions.fingerprint}:exit",
                    **{**common, "diagnostics": {
                        "kind": "initial_exit", **decision_diagnostics,
                    }},
                ),
            ])

        signal_records = ranked[[
            "signal_date", "instrument_id", "score",
        ]].copy()
        signal_records["source_id"] = (
            ranked["model_id"].astype(str).to_numpy()
            if not ranked.empty else pd.Series(dtype=str)
        )
        signal_records["source_version"] = (
            ranked["model_version"].astype(str).to_numpy()
            if not ranked.empty else pd.Series(dtype=str)
        )
        signals = SignalFrame(
            signal_records,
            hash_payload(signal_records.to_dict("records")),
        )
        self.planned.append(decision_diagnostics)
        return StrategyDecision(
            None, next_state, signals, decision_diagnostics,
            tuple(scheduled),
        )

    def _diagnostic_value(self, day: date, column: str) -> int:
        assert self._universe is not None
        rows = self._universe.diagnostics[
            self._universe.diagnostics["signal_date"] == day
        ]
        return int(rows[column].iloc[0]) if not rows.empty else 0

    def _exit_target(
        self, context, cohort_id: str, *, session_offset: int,
        signal_fingerprint: str,
    ) -> ScheduledTargetPortfolio:
        state_hash = hash_payload(context.state)
        assert self._universe is not None
        return ScheduledTargetPortfolio(
            self.strategy_id, context.signal_date, session_offset, "close",
            cohort_id, {}, 1.0, self._universe.fingerprint,
            signal_fingerprint, state_hash, state_hash,
            {"kind": "retry_exit", "cohort_id": cohort_id},
        )
