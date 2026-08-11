from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from strategies.ml_ema20_momentum_v1.dataset import LABEL_COLUMN, MODEL_FEATURES
from strategies.ml_ema20_momentum_v1.feature_search import (
    DEVELOPMENT_YEARS,
    _annual_metrics,
    generate_feature_trials,
    search_objective,
)
from strategies.ml_ema20_momentum_v1.xgb_training import preprocess_daily


def test_random_feature_trials_are_reproducible_unique_and_bounded() -> None:
    first = generate_feature_trials(500, seed=17)
    second = generate_feature_trials(500, seed=17)
    assert first == second
    assert len({item.features for item in first}) == 500
    assert all(5 <= len(item.features) <= 30 for item in first)
    assert all(set(item.features) <= set(MODEL_FEATURES) for item in first)


def test_search_objective_penalizes_unstable_annual_excess() -> None:
    stable = [{
        "top3_mean_return": 0.02,
        "pool_mean_return": 0.01,
    } for _ in DEVELOPMENT_YEARS]
    unstable = [
        {
            "top3_mean_return": 0.03 if index % 2 else 0.01,
            "pool_mean_return": 0.01,
        }
        for index, _ in enumerate(DEVELOPMENT_YEARS)
    ]
    assert search_objective(stable) > search_objective(unstable)


def test_annual_metrics_uses_stable_top3_ranking() -> None:
    rows = []
    scores = []
    for day in (date(2020, 1, 2), date(2020, 1, 3)):
        for member in range(5):
            rows.append({
                "signal_date": day,
                "instrument_id": f"{member:06d}.XSHE",
                LABEL_COLUMN: member / 100.0,
                "relevance": member,
            })
            scores.append(float(member))
    metrics = _annual_metrics(pd.DataFrame(rows), np.asarray(scores))
    assert metrics["ndcg_at_3"] == pytest.approx(1.0)
    assert metrics["top1_mean_return"] == pytest.approx(0.04)
    assert metrics["top3_mean_return"] == pytest.approx(0.03)


def test_subset_preprocessing_only_requires_selected_columns() -> None:
    frame = pd.DataFrame({
        "signal_date": [date(2020, 1, 2)] * 3,
        "ret_1d": [1.0, 2.0, 3.0],
        "ret_3d": [np.inf, np.inf, np.inf],
    })
    result = preprocess_daily(frame, ("ret_1d",))
    assert result["ret_1d"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
