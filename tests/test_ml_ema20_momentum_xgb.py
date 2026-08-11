from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from strategies.ml_ema20_momentum_v1.dataset import LABEL_COLUMN, MODEL_FEATURES
from strategies.ml_ema20_momentum_v1.report import _yearly_performance
from strategies.ml_ema20_momentum_v1.xgb_training import (
    BINARY_FEATURES,
    make_relevance,
    ndcg_at_3,
    prepare_fold,
    preprocess_daily,
)


def _frame(days: list[date], members: int = 5) -> pd.DataFrame:
    rows = []
    for day_index, day in enumerate(days):
        for member in range(members):
            row = {
                "signal_date": day,
                "instrument_id": f"{member:06d}.XSHE",
                "label_start_date": day + timedelta(days=1),
                "label_end_date": day + timedelta(days=2),
                LABEL_COLUMN: float(member - 2) / 100 + day_index / 10000,
            }
            row.update({feature: float(member + day_index) for feature in MODEL_FEATURES})
            for feature in BINARY_FEATURES:
                row[feature] = float(member % 2)
            rows.append(row)
    return pd.DataFrame(rows)


def test_daily_preprocessing_is_bounded_and_does_not_mix_dates() -> None:
    days = [date(2020, 1, 2), date(2020, 1, 3)]
    frame = _frame(days)
    frame.loc[frame["signal_date"] == days[1], "annualized_returns"] += 1_000_000
    transformed = preprocess_daily(frame)
    for _, group in transformed.groupby("signal_date"):
        assert group["annualized_returns"].between(0, 1).all()
        assert group["annualized_returns"].tolist() == pytest.approx(
            [0.2, 0.4, 0.6, 0.8, 1.0]
        )
    assert transformed[list(BINARY_FEATURES)].isin([0.0, 1.0]).all(axis=None)


def test_relevance_uses_same_day_grades_and_preserves_ties() -> None:
    frame = _frame([date(2020, 1, 2)], members=5)
    frame.loc[1, LABEL_COLUMN] = frame.loc[2, LABEL_COLUMN]
    relevance = make_relevance(frame)
    assert relevance.between(0, 4).all()
    assert relevance.iloc[1] == relevance.iloc[2]
    assert relevance.iloc[0] == 0
    assert relevance.iloc[-1] == 4


def test_prepare_fold_purges_outcomes_and_two_embargo_dates() -> None:
    days = [date(2020, 1, 1) + timedelta(days=value) for value in range(150)]
    train = _frame(days)
    test = _frame([date(2021, 1, 4)])
    fold = prepare_fold(train, test, validation_sessions=126, embargo_sessions=2)
    assert len(fold.embargo_dates) == 2
    assert not fold.fit["signal_date"].isin(fold.embargo_dates).any()
    assert (fold.fit["label_end_date"] < fold.validation_start).all()
    assert set(fold.validation["signal_date"]) == set(days[-126:])


def test_ndcg_is_one_for_perfect_order() -> None:
    frame = _frame([date(2020, 1, 2)])
    frame["relevance"] = make_relevance(frame)
    frame["score"] = frame[LABEL_COLUMN]
    assert ndcg_at_3(frame) == pytest.approx(1.0)


def test_yearly_report_and_best_day_removal() -> None:
    nav = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2021-01-04"]),
        "nav": [100.0, 110.0, 99.0],
    })
    yearly, trimmed = _yearly_performance(nav)
    assert yearly["year"].tolist() == [2020, 2021]
    assert yearly.loc[0, "return"] == pytest.approx(0.1)
    assert "return_without_best_10_days" in trimmed


def test_preprocessing_rejects_nonfinite_inputs() -> None:
    frame = _frame([date(2020, 1, 2)])
    frame.loc[0, "score"] = np.inf
    with pytest.raises(Exception, match="finite"):
        preprocess_daily(frame)
