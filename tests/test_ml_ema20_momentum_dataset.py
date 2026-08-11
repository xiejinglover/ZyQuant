from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from strategies.ml_ema20_momentum_v1.dataset import (
    LABEL_COLUMN,
    MODEL_FEATURES,
    add_cross_sectional_features,
    attach_executable_labels,
    numeric_quality,
    rolling_year_folds,
)


def test_labels_use_next_calendar_open_and_following_calendar_close() -> None:
    days = [date(2020, 1, 2) + timedelta(days=value) for value in range(5)]
    panel = pd.DataFrame({
        "signal_date": [days[0]], "instrument_id": ["000001.XSHE"],
    })
    prices = pd.DataFrame({
        "trade_date": days,
        "instrument_id": ["000001.XSHE"] * len(days),
        "open_post": [10.0, 11.0, 99.0, 13.0, 14.0],
        "close_post": [10.5, 11.5, 12.1, 13.5, 14.5],
    })
    result = attach_executable_labels(panel, days, prices)
    assert result.loc[0, "label_start_date"] == days[1]
    assert result.loc[0, "label_end_date"] == days[2]
    assert result.loc[0, LABEL_COLUMN] == pytest.approx(0.1)
    assert bool(result.loc[0, "label_valid"])


def test_missing_future_price_is_preserved_and_audited() -> None:
    days = [date(2020, 1, 2) + timedelta(days=value) for value in range(3)]
    panel = pd.DataFrame({
        "signal_date": [days[0]], "instrument_id": ["000001.XSHE"],
    })
    prices = pd.DataFrame({
        "trade_date": days[:2],
        "instrument_id": ["000001.XSHE"] * 2,
        "open_post": [10.0, 11.0], "close_post": [10.0, 11.0],
    })
    result = attach_executable_labels(panel, days, prices)
    assert np.isnan(result.loc[0, LABEL_COLUMN])
    assert not bool(result.loc[0, "label_valid"])
    assert result.loc[0, "label_status"] == "missing_exit_price"


def test_cross_sectional_features_are_only_ranked_inside_event_pool() -> None:
    panel = pd.DataFrame({
        "signal_date": [date(2020, 1, 2)] * 2 + [date(2020, 1, 3)],
        "instrument_id": ["B.XSHE", "A.XSHE", "B.XSHE"],
        "score": [2.0, 2.0, 3.0],
        "short_term_return_5d": [0.2, 0.1, 0.3],
        "ret_20d": [0.1, 0.2, 0.3],
    })
    result = add_cross_sectional_features(panel)
    assert result.loc[0, "score_rank_pct"] == 0.75
    assert result.loc[1, "score_rank_pct"] == 0.75
    assert result.loc[0, "ret_5d_rank_pct"] == 1.0
    assert result.loc[1, "ret_20d_rank_pct"] == 1.0
    assert result.loc[1, "is_new_top1"] == 1.0
    assert result.loc[2, "is_new_top1"] == 1.0


def test_rolling_folds_use_prior_three_years_and_purge_label_end() -> None:
    rows = []
    for year in range(2010, 2015):
        signal = date(year, 12, 30)
        rows.append({
            "signal_date": signal,
            "instrument_id": "000001.XSHE",
            "label_start_date": signal,
            "label_end_date": date(year + 1, 1, 2),
            "label_valid": True,
            LABEL_COLUMN: 0.01,
        })
    panel = pd.DataFrame(rows)
    for feature in MODEL_FEATURES:
        panel[feature] = 1.0
    folds = rolling_year_folds(panel, first_year=2010, last_year=2014)
    assert [fold.test_year for fold in folds] == [2013, 2014]
    assert folds[0].train_years == (2010, 2011, 2012)
    assert set(folds[0].train["signal_date"].map(lambda value: value.year)) == {2010, 2011}
    assert (folds[0].train["label_end_date"] < folds[0].test["signal_date"].min()).all()


def test_numeric_quality_counts_nan_and_signed_infinity_separately() -> None:
    frame = pd.DataFrame({"value": [1.0, np.nan, np.inf, -np.inf]})
    quality = numeric_quality(frame, ["value"])["value"]
    assert quality["nan"] == 1
    assert quality["positive_inf"] == 1
    assert quality["negative_inf"] == 1
    assert quality["finite"] == 1
