from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zyquant.core.exceptions import StrategyError
from zyquant.core.hashing import hash_file

from .dataset import LABEL_COLUMN, MODEL_FEATURES


BINARY_FEATURES = (
    "has_recent_drop",
    "is_decaying",
    "over_return_cap",
    "is_high_level_volume_spike",
    "is_new_top1",
)
CONTINUOUS_FEATURES = tuple(
    feature for feature in MODEL_FEATURES if feature not in BINARY_FEATURES
)
FEATURE_SET_ID = "ml_ema20_53_daily_rank_v1"
MODEL_VERSION = "1"
SEED = 20260722
XGB_PARAMETERS: dict[str, Any] = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@3",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 10.0,
    "max_bin": 256,
    "tree_method": "hist",
    "random_state": SEED,
}


@dataclass(frozen=True)
class PreparedFold:
    train: pd.DataFrame
    fit: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_start: date
    embargo_dates: tuple[date, date]


def preprocess_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """PIT daily winsorisation and percentile ranks inside the event pool."""
    result = frame.copy()
    numeric = result[list(MODEL_FEATURES)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise StrategyError("XGBoost input features must all be finite")
    for column in CONTINUOUS_FEATURES:
        grouped = numeric.groupby(result["signal_date"], sort=False)[column]
        lower = grouped.transform(lambda values_: values_.quantile(0.01))
        upper = grouped.transform(lambda values_: values_.quantile(0.99))
        clipped = numeric[column].clip(lower=lower, upper=upper)
        result[column] = clipped.groupby(result["signal_date"], sort=False).rank(
            method="average", pct=True
        )
    for column in BINARY_FEATURES:
        result[column] = numeric[column].astype(float)
        if not result[column].isin([0.0, 1.0]).all():
            raise StrategyError(f"binary feature contains values outside 0/1: {column}")
    transformed = result[list(MODEL_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(transformed).all():
        raise StrategyError("preprocessing produced a non-finite feature")
    return result


def make_relevance(frame: pd.DataFrame) -> pd.Series:
    """Map same-day returns to stable 0-4 relevance grades; ties share grades."""
    labels = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")
    if not np.isfinite(labels.to_numpy(dtype=float)).all():
        raise StrategyError("ranking labels must all be finite")
    percentile = labels.groupby(frame["signal_date"], sort=False).rank(
        method="average", pct=True
    )
    relevance = np.minimum(np.ceil(percentile * 5.0).astype(int) - 1, 4)
    return pd.Series(relevance, index=frame.index, name="relevance", dtype=int)


def prepare_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    validation_sessions: int = 126,
    embargo_sessions: int = 2,
) -> PreparedFold:
    train = train.sort_values(
        ["signal_date", "instrument_id"], kind="mergesort", ignore_index=True
    ).copy()
    test = test.sort_values(
        ["signal_date", "instrument_id"], kind="mergesort", ignore_index=True
    ).copy()
    unique_dates = sorted(set(train["signal_date"]))
    if len(unique_dates) <= validation_sessions + embargo_sessions:
        raise StrategyError("training fold has too few dates for validation and embargo")
    validation_dates = unique_dates[-validation_sessions:]
    validation_start = validation_dates[0]
    before = [day for day in unique_dates if day < validation_start]
    embargo = tuple(before[-embargo_sessions:])
    fit = train[
        (train["signal_date"] < validation_start)
        & (train["label_end_date"] < validation_start)
        & ~train["signal_date"].isin(embargo)
    ].copy()
    validation = train[train["signal_date"].isin(validation_dates)].copy()
    if fit.empty or validation.empty:
        raise StrategyError("purged fold produced an empty fit or validation set")
    if len(embargo) != 2:
        raise StrategyError("fold did not produce the required two-session embargo")
    return PreparedFold(train, fit, validation, test, validation_start, embargo)


def _matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ordered = frame.sort_values(
        ["signal_date", "instrument_id"], kind="mergesort", ignore_index=True
    )
    x = ordered[list(MODEL_FEATURES)].astype(float)
    y = make_relevance(ordered).to_numpy(dtype=int)
    qid, _ = pd.factorize(ordered["signal_date"], sort=False)
    return x, y, qid.astype(np.uint32)


def _ranker(device: str, *, iterations: int, early_stopping: bool) -> Any:
    try:
        from xgboost import XGBRanker
    except ImportError as exc:
        raise StrategyError(
            "XGBoost training requires the 'ml-ranking' optional dependency"
        ) from exc
    parameters = {
        **XGB_PARAMETERS,
        "device": device,
        "n_estimators": iterations,
    }
    if early_stopping:
        parameters["early_stopping_rounds"] = 100
    return XGBRanker(**parameters)


def fit_early_stopping(fold: PreparedFold, device: str) -> tuple[Any, int, float]:
    x_fit, y_fit, qid_fit = _matrix(fold.fit)
    x_valid, y_valid, qid_valid = _matrix(fold.validation)
    model = _ranker(device, iterations=2000, early_stopping=True)
    model.fit(
        x_fit, y_fit, qid=qid_fit,
        eval_set=[(x_valid, y_valid)], eval_qid=[qid_valid],
        verbose=False,
    )
    best_iteration = int(getattr(model, "best_iteration", 1999)) + 1
    results = model.evals_result()
    scores = results.get("validation_0", {}).get("ndcg@3", [])
    best_score = float(scores[best_iteration - 1]) if scores else float("nan")
    return model, best_iteration, best_score


def refit_full(fold: PreparedFold, device: str, iterations: int) -> Any:
    x_train, y_train, qid_train = _matrix(fold.train)
    model = _ranker(device, iterations=iterations, early_stopping=False)
    model.fit(x_train, y_train, qid=qid_train, verbose=False)
    return model


def predict(model: Any, frame: pd.DataFrame) -> np.ndarray:
    ordered = frame.sort_values(
        ["signal_date", "instrument_id"], kind="mergesort", ignore_index=True
    )
    score = np.asarray(model.predict(ordered[list(MODEL_FEATURES)]), dtype=float)
    if score.shape != (len(ordered),) or not np.isfinite(score).all():
        raise StrategyError("XGBoost produced invalid prediction scores")
    return score


def determinism_check(fold: PreparedFold, device: str) -> dict[str, Any]:
    first, first_iterations, first_metric = fit_early_stopping(fold, device)
    second, second_iterations, second_metric = fit_early_stopping(fold, device)
    sample = fold.validation.sort_values(
        ["signal_date", "instrument_id"], kind="mergesort", ignore_index=True
    )
    first_score = predict(first, sample)
    second_score = predict(second, sample)
    maximum_error = float(np.max(np.abs(first_score - second_score)))
    return {
        "stable": maximum_error <= 1e-12 and first_iterations == second_iterations,
        "maximum_absolute_error": maximum_error,
        "first_iterations": first_iterations,
        "second_iterations": second_iterations,
        "first_ndcg_at_3": first_metric,
        "second_ndcg_at_3": second_metric,
    }


def ndcg_at_3(frame: pd.DataFrame) -> float:
    values = []
    for _, group in frame.groupby("signal_date", sort=False):
        ranked = group.sort_values(
            ["score", "instrument_id"], ascending=[False, True], kind="mergesort"
        ).head(3)
        ideal = group.sort_values(
            ["relevance", "instrument_id"], ascending=[False, True], kind="mergesort"
        ).head(3)
        discounts = 1.0 / np.log2(np.arange(2, 2 + len(ranked)))
        dcg = float(np.sum((2.0 ** ranked["relevance"].to_numpy() - 1.0) * discounts))
        ideal_discounts = 1.0 / np.log2(np.arange(2, 2 + len(ideal)))
        idcg = float(np.sum((2.0 ** ideal["relevance"].to_numpy() - 1.0) * ideal_discounts))
        values.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(values)) if values else float("nan")


def prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    ranked = frame.sort_values(
        ["signal_date", "score", "instrument_id"],
        ascending=[True, False, True], kind="mergesort",
    )
    top1 = ranked.groupby("signal_date", sort=False).head(1)
    top3 = ranked.groupby("signal_date", sort=False).head(3)
    by_grade = frame.groupby("relevance")[LABEL_COLUMN].agg(["count", "mean"]).reset_index()
    return {
        "rows": len(frame),
        "dates": int(frame["signal_date"].nunique()),
        "ndcg_at_3": ndcg_at_3(frame),
        "pool_mean_return": float(frame[LABEL_COLUMN].mean()),
        "top1_mean_return": float(top1[LABEL_COLUMN].mean()),
        "top3_mean_return": float(top3[LABEL_COLUMN].mean()),
        "relevance_returns": by_grade.to_dict("records"),
    }


def model_sha256(path: Path) -> str:
    return hash_file(path)


def atomic_publish_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise StrategyError(f"immutable model directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)


def prediction_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["signal_date", "instrument_id", "score", "model_id"]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()
