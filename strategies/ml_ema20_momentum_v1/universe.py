from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from zyquant.core.exceptions import DataContractError
from zyquant.core.hashing import hash_payload


ST_MARKERS = ("ST", "*ST", "退")


@dataclass(frozen=True)
class Ema20UniversePanel:
    by_date: dict[date, tuple[str, ...]]
    diagnostics: pd.DataFrame
    fingerprint: str

    def eligible(self, day: date) -> tuple[str, ...]:
        return self.by_date.get(day, ())


def _st_codes(frame: pd.DataFrame, day: date) -> set[str]:
    visible = frame[
        (frame["known_at"] <= day)
        & (frame["effective_from"] <= day)
        & (frame["effective_to"].isna() | (frame["effective_to"] > day))
    ]
    if visible.empty:
        return set()
    current = visible.sort_values(
        ["instrument_id", "effective_from"], kind="mergesort"
    ).groupby("instrument_id", sort=False).tail(1)
    marked = current["name"].astype(str).apply(
        lambda value: any(token in value.upper() for token in ST_MARKERS)
    )
    return set(current.loc[marked, "instrument_id"].astype(str))


def build_ema20_universe(
    snapshot: Any,
    start: date,
    end: date,
    *,
    ema_span: int = 20,
    minimum_listed_sessions: int = 120,
    minimum_price: float | None = None,
    minimum_median_amount: float = 50_000_000.0,
) -> Ema20UniversePanel:
    """Build a sparse PIT universe without registering framework factors."""
    if ema_span < 2 or minimum_listed_sessions < 0:
        raise ValueError("invalid EMA universe parameters")
    if minimum_price is not None and minimum_price < 0:
        raise ValueError("minimum_price must be non-negative or None")
    calendar = sorted(set(snapshot.table("trade_calendar")["trade_date"]))
    run_days = [day for day in calendar if start <= day <= end]
    if not run_days:
        raise ValueError("EMA universe run range has no trading sessions")
    instruments = snapshot.table("instruments").copy()
    required_instrument = {
        "instrument_id", "symbol", "exchange", "asset_type", "list_date",
        "delist_date",
    }
    if required_instrument - set(instruments):
        raise DataContractError("instruments table cannot build EMA universe")
    symbols = instruments["symbol"].astype(str)
    not_b_share = ~(
        ((instruments["exchange"] == "XSHG") & symbols.str.startswith("900"))
        | ((instruments["exchange"] == "XSHE") & symbols.str.startswith("200"))
    )
    currency_ok = (
        instruments["currency"].isna() | (instruments["currency"] == "CNY")
        if "currency" in instruments else True
    )
    stocks = instruments[
        instruments["exchange"].isin(["XSHG", "XSHE"])
        & (instruments["asset_type"] == "stock")
        & not_b_share
        & currency_ok
    ].copy()
    stocks["instrument_id"] = stocks["instrument_id"].astype(str)
    codes = stocks["instrument_id"].tolist()
    history_start = calendar[0]
    post = snapshot.post_adjusted_bars(
        history_start, end, codes, ["close_post"], cutoff=end,
    )
    raw = snapshot.raw_bars(
        history_start, end, codes,
        ["close", "amount", "volume", "paused"], cutoff=end,
    )
    if post.empty or raw.empty:
        raise DataContractError("EMA universe requires adjusted and raw bars")
    bars = post.merge(
        raw, on=["trade_date", "instrument_id"], how="inner",
        validate="one_to_one",
    ).sort_values(["instrument_id", "trade_date"], kind="mergesort")
    grouped = bars.groupby("instrument_id", sort=False)
    bars["ema"] = grouped["close_post"].transform(
        lambda values: values.ewm(
            span=ema_span, adjust=False, min_periods=ema_span,
        ).mean()
    )
    bars["previous_close"] = grouped["close_post"].shift(1)
    bars["previous_ema"] = grouped["ema"].shift(1)
    bars["median_amount"] = grouped["amount"].transform(
        lambda values: values.rolling(20, min_periods=20).median()
    )
    meta = stocks.set_index("instrument_id")
    bars["list_date"] = bars["instrument_id"].map(meta["list_date"])
    bars["delist_date"] = bars["instrument_id"].map(meta["delist_date"])
    calendar_array = np.asarray(calendar, dtype="datetime64[D]")
    trade_values = pd.to_datetime(bars["trade_date"]).to_numpy(dtype="datetime64[D]")
    list_values = pd.to_datetime(bars["list_date"]).to_numpy(dtype="datetime64[D]")
    bars["listed_sessions"] = (
        np.searchsorted(calendar_array, trade_values, side="right")
        - np.searchsorted(calendar_array, list_values, side="left")
    )
    bars = bars[bars["trade_date"].isin(run_days)].copy()
    names = snapshot.table("special_treatment", cutoff=end)
    if names.empty:
        raise DataContractError(
            "EMA universe requires point-in-time special_treatment data"
        )
    for column in ("known_at", "effective_from", "effective_to"):
        converted = pd.to_datetime(names[column], errors="coerce")
        names[column] = converted.map(
            lambda value: value.date() if pd.notna(value) else None
        ).astype(object)
    names["instrument_id"] = names["instrument_id"].astype(str)

    by_date: dict[date, tuple[str, ...]] = {}
    diagnostics = []
    for day, daily in bars.groupby("trade_date", sort=True):
        alive = daily["delist_date"].isna() | (daily["delist_date"] > day)
        seasoned = daily["listed_sessions"] >= minimum_listed_sessions
        not_st = ~daily["instrument_id"].isin(_st_codes(names, day))
        not_paused = ~daily["paused"].fillna(True).astype(bool)
        has_volume = pd.to_numeric(daily["volume"], errors="coerce").fillna(0) > 0
        price_ok = (
            daily["close"] >= minimum_price
            if minimum_price is not None
            else pd.Series(True, index=daily.index)
        )
        tradable = (
            not_paused & has_volume & price_ok
            & (daily["median_amount"] >= minimum_median_amount)
        )
        ready = daily[[
            "close_post", "ema", "previous_close", "previous_ema",
        ]].notna().all(axis=1)
        crossed = (
            (daily["previous_close"] <= daily["previous_ema"])
            & (daily["close_post"] > daily["ema"])
        )
        eligible = daily.loc[
            alive & seasoned & not_st & tradable & ready & crossed,
            "instrument_id",
        ].astype(str).sort_values(kind="mergesort")
        values = tuple(eligible)
        by_date[day] = values
        diagnostics.append({
            "signal_date": day,
            "stock_rows": int(len(daily)),
            "excluded_not_alive": int((~alive).sum()),
            "excluded_insufficient_listing": int((alive & ~seasoned).sum()),
            "excluded_st": int((alive & seasoned & ~not_st).sum()),
            "excluded_paused": int(
                (alive & seasoned & not_st & daily["paused"].fillna(True).astype(bool)).sum()
            ),
            "excluded_zero_volume": int(
                (alive & seasoned & not_st & not_paused & ~has_volume).sum()
            ),
            "excluded_price": int((
                alive & seasoned & not_st & not_paused & has_volume & ~price_ok
            ).sum()),
            "excluded_liquidity_or_warmup": int(
                (alive & seasoned & not_st & not_paused & has_volume & price_ok
                 & ~(daily["median_amount"] >= minimum_median_amount)).sum()
            ),
            "base_pool_count": int(
                (alive & seasoned & not_st & tradable & ready).sum()
            ),
            "cross_count": len(values),
        })
    diagnostic_frame = pd.DataFrame(diagnostics)
    fingerprint = hash_payload({
        "dataset": snapshot.metadata.fingerprint,
        "definition": {
            "ema_span": ema_span, "adjust": False,
            "minimum_listed_sessions": minimum_listed_sessions,
            "minimum_price": minimum_price,
            "minimum_median_amount": minimum_median_amount,
        },
        "eligible": [
            (str(day), list(codes_for_day))
            for day, codes_for_day in sorted(by_date.items())
        ],
    })
    return Ema20UniversePanel(by_date, diagnostic_frame, fingerprint)
