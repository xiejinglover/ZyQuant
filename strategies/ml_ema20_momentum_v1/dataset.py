from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from zyquant.core.exceptions import DataContractError

from .factors import FEATURE_NAMES
from .universe import Ema20UniversePanel


CROSS_SECTIONAL_FEATURES = (
    "score_rank_pct",
    "ret_5d_rank_pct",
    "ret_20d_rank_pct",
    "is_new_top1",
)
MODEL_FEATURES = (*FEATURE_NAMES, *CROSS_SECTIONAL_FEATURES)
LABEL_COLUMN = "next_open_to_following_close_return"
LABEL_DEFINITION = {
    "entry_offset": 1,
    "horizon": 1,
    "entry_price_field": "open_post",
    "exit_price_field": "close_post",
    "formula": "close_post[T+2] / open_post[T+1] - 1",
}


@dataclass(frozen=True)
class RollingYearFold:
    fold_id: str
    train_years: tuple[int, int, int]
    test_year: int
    train: pd.DataFrame
    test: pd.DataFrame


def candidate_keys(panel: Ema20UniversePanel) -> pd.DataFrame:
    rows = [
        {"signal_date": day, "instrument_id": instrument_id}
        for day, instruments in sorted(panel.by_date.items())
        for instrument_id in instruments
    ]
    result = pd.DataFrame(rows, columns=["signal_date", "instrument_id"])
    if result.empty:
        raise DataContractError("EMA20 event pool produced no dataset candidates")
    if result.duplicated(["signal_date", "instrument_id"]).any():
        raise DataContractError("EMA20 event pool contains duplicate candidate keys")
    return result.sort_values(
        ["signal_date", "instrument_id"], kind="mergesort", ignore_index=True
    )


def attach_factor(
    panel: pd.DataFrame,
    factor_frame: pd.DataFrame,
    feature_name: str,
) -> pd.DataFrame:
    required = {"trade_date", "instrument_id", "value"}
    if required - set(factor_frame):
        raise DataContractError(f"factor {feature_name} has an invalid schema")
    values = factor_frame[["trade_date", "instrument_id", "value"]].rename(
        columns={"trade_date": "signal_date", "value": feature_name}
    )
    if values.duplicated(["signal_date", "instrument_id"]).any():
        raise DataContractError(f"factor {feature_name} contains duplicate keys")
    return panel.merge(
        values,
        on=["signal_date", "instrument_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["score_rank_pct"] = result.groupby("signal_date")["score"].rank(
        pct=True
    )
    result["ret_5d_rank_pct"] = result.groupby("signal_date")[
        "short_term_return_5d"
    ].rank(pct=True)
    result["ret_20d_rank_pct"] = result.groupby("signal_date")["ret_20d"].rank(
        pct=True
    )
    result["is_new_top1"] = 0.0
    previous_top1: str | None = None
    for _, group in result.groupby("signal_date", sort=True):
        scores = pd.to_numeric(group["score"], errors="coerce")
        finite = group[np.isfinite(scores)]
        if finite.empty:
            continue
        winner = finite.sort_values(
            ["score", "instrument_id"],
            ascending=[False, True],
            kind="mergesort",
        ).index[0]
        instrument_id = str(result.at[winner, "instrument_id"])
        result.at[winner, "is_new_top1"] = float(instrument_id != previous_top1)
        previous_top1 = instrument_id
    return result


def attach_executable_labels(
    panel: pd.DataFrame,
    calendar: Sequence[date],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    days = sorted(set(calendar))
    next_one = {days[index]: days[index + 1] for index in range(len(days) - 1)}
    next_two = {days[index]: days[index + 2] for index in range(len(days) - 2)}
    result = panel.copy()
    result["label_start_date"] = result["signal_date"].map(next_one)
    result["label_end_date"] = result["signal_date"].map(next_two)

    required = {"trade_date", "instrument_id", "open_post", "close_post"}
    if required - set(prices):
        raise DataContractError("post-adjusted prices cannot build executable labels")
    if prices.duplicated(["trade_date", "instrument_id"]).any():
        raise DataContractError("post-adjusted label prices contain duplicate keys")
    entry = prices[["trade_date", "instrument_id", "open_post"]].rename(
        columns={"trade_date": "label_start_date", "open_post": "entry_price"}
    )
    exit_ = prices[["trade_date", "instrument_id", "close_post"]].rename(
        columns={"trade_date": "label_end_date", "close_post": "exit_price"}
    )
    result = result.merge(
        entry,
        on=["label_start_date", "instrument_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        exit_,
        on=["label_end_date", "instrument_id"],
        how="left",
        validate="many_to_one",
    )
    entry_values = pd.to_numeric(result["entry_price"], errors="coerce").to_numpy()
    exit_values = pd.to_numeric(result["exit_price"], errors="coerce").to_numpy()
    valid = (
        np.isfinite(entry_values)
        & np.isfinite(exit_values)
        & (entry_values > 0)
        & (exit_values > 0)
    )
    labels = np.full(len(result), np.nan, dtype=float)
    np.divide(exit_values, entry_values, out=labels, where=valid)
    labels[valid] -= 1.0
    result[LABEL_COLUMN] = labels
    result["label_valid"] = valid & np.isfinite(labels)
    result["label_status"] = np.select(
        [
            result["label_start_date"].isna(),
            result["label_end_date"].isna(),
            ~np.isfinite(entry_values),
            ~np.isfinite(exit_values),
            entry_values <= 0,
            exit_values <= 0,
            ~np.isfinite(labels),
        ],
        [
            "missing_entry_date",
            "missing_exit_date",
            "missing_entry_price",
            "missing_exit_price",
            "nonpositive_entry_price",
            "nonpositive_exit_price",
            "nonfinite_label",
        ],
        default="valid",
    )
    return result


def rolling_year_folds(
    panel: pd.DataFrame,
    *,
    first_year: int,
    last_year: int,
    training_years: int = 3,
) -> tuple[RollingYearFold, ...]:
    if training_years != 3:
        raise ValueError("ml_ema20_momentum_v1 currently requires three training years")
    usable = panel[panel["label_valid"]].copy()
    folds: list[RollingYearFold] = []
    for test_year in range(first_year + training_years, last_year + 1):
        years = tuple(range(test_year - training_years, test_year))
        train = usable[usable["signal_date"].map(lambda value: value.year in years)]
        test = usable[usable["signal_date"].map(lambda value: value.year == test_year)]
        if train.empty or test.empty:
            continue
        # The model is frozen before the calendar test year begins.  Do not
        # admit a prior-year signal whose outcome only becomes known during
        # the test year, even when the year's first event occurs later.
        test_boundary = date(test_year, 1, 1)
        train = train[train["label_end_date"] < test_boundary].copy()
        if train.empty:
            continue
        columns = [
            "signal_date", "instrument_id", "label_start_date", "label_end_date",
            *MODEL_FEATURES, LABEL_COLUMN,
        ]
        folds.append(RollingYearFold(
            fold_id=f"train_{years[0]}_{years[-1]}__test_{test_year}",
            train_years=(years[0], years[1], years[2]),
            test_year=test_year,
            train=train[columns].reset_index(drop=True),
            test=test[columns].reset_index(drop=True),
        ))
    return tuple(folds)


def numeric_quality(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        result[column] = {
            "rows": int(len(values)),
            "nan": int(np.isnan(values).sum()),
            "positive_inf": int(np.isposinf(values).sum()),
            "negative_inf": int(np.isneginf(values).sum()),
            "finite": int(np.isfinite(values).sum()),
            "minimum": float(finite.min()) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None,
            "mean": float(finite.mean()) if len(finite) else None,
            "std_ddof0": float(finite.std(ddof=0)) if len(finite) else None,
        }
    return result


def frame_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    normalized = frame[list(columns)].copy()
    hashed = pd.util.hash_pandas_object(normalized, index=False).to_numpy(
        dtype=np.uint64
    )
    return hashlib.sha256(hashed.tobytes()).hexdigest()
