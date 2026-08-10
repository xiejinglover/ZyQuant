from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from zyquant.core.hashing import hash_payload

from .types import StrategyContext, UniverseSnapshot


@dataclass(frozen=True)
class StandardUniverseSelector:
    universe_id: str | None = None
    asset_types: tuple[str, ...] = ("stock", "etf")
    minimum_listed_days: int = 0
    median_amount_window: int = 20
    minimum_median_amount: float = 0.0

    def select(self, context: StrategyContext) -> UniverseSnapshot:
        day = context.signal_date
        instruments = context.data.table("instruments")
        candidates = instruments[instruments["asset_type"].isin(self.asset_types)].copy()
        if self.universe_id:
            membership = context.data.table(
                "universe_membership", cutoff=context.cutoff
            )
            membership = membership[
                (membership["universe_id"] == self.universe_id)
                & (membership["effective_from"] <= day)
                & (membership["effective_to"].isna() | (membership["effective_to"] >= day))
                & (membership["known_at"] <= context.cutoff)
            ]
            candidates = candidates[candidates["instrument_id"].isin(membership["instrument_id"])]
        reasons: list[dict[str, str]] = []
        eligible: list[str] = []
        all_codes = set(candidates["instrument_id"].astype(str))
        start = max(date(1900, 1, 1), context.data.metadata.as_of_date - timedelta(days=20 * 366))
        history_end = min(day, context.cutoff)
        raw = context.data.raw_bars(
            start, history_end, sorted(all_codes), ["amount"], context.cutoff
        )
        for row in candidates.itertuples(index=False):
            code = str(row.instrument_id)
            if row.list_date and (day - row.list_date).days < self.minimum_listed_days:
                reasons.append({"instrument_id": code, "reason_code": "listed_days"})
                continue
            if getattr(row, "delist_date", None) and row.delist_date < day:
                reasons.append({"instrument_id": code, "reason_code": "delisted"})
                continue
            history = raw[raw["instrument_id"] == code].sort_values("trade_date")
            if len(history) < max(1, self.median_amount_window):
                reasons.append({"instrument_id": code, "reason_code": "history"})
                continue
            median = history["amount"].tail(self.median_amount_window).median()
            if pd.isna(median) or median < self.minimum_median_amount:
                reasons.append({"instrument_id": code, "reason_code": "liquidity"})
                continue
            eligible.append(code)
        eligible.sort()
        fingerprint = hash_payload({"date": day, "eligible": eligible, "selector": self.__dict__})
        return UniverseSnapshot(
            context.strategy_id,
            day,
            tuple(eligible),
            pd.DataFrame(reasons, columns=["instrument_id", "reason_code"]),
            fingerprint,
        )
