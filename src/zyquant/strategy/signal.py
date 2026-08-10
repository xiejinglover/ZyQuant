from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from zyquant.core.exceptions import StrategyError
from zyquant.core.hashing import hash_payload
from zyquant.factors import BaseFactor

from .types import SignalFrame, StrategyContext, UniverseSnapshot


@dataclass(frozen=True)
class ExternalSignalGenerator:
    signals: pd.DataFrame
    source_id: str = "external"
    source_version: str = "1"

    def generate(self, context: StrategyContext, universe: UniverseSnapshot) -> SignalFrame:
        frame = self.signals.copy()
        frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.date
        frame = frame[
            (frame["signal_date"] == context.signal_date)
            & frame["instrument_id"].astype(str).isin(universe.eligible)
        ].copy()
        frame["source_id"] = self.source_id
        frame["source_version"] = self.source_version
        return _validated(frame)


@dataclass(frozen=True)
class PredictionSignalGenerator:
    predictions: pd.DataFrame

    def generate(self, context: StrategyContext, universe: UniverseSnapshot) -> SignalFrame:
        frame = self.predictions.copy()
        required = {
            "signal_date", "instrument_id", "score", "model_id", "model_version",
            "feature_cutoff", "train_cutoff", "data_fingerprint", "feature_set_id",
        }
        missing = required - set(frame.columns)
        if missing:
            raise StrategyError(f"prediction signal missing columns: {sorted(missing)}")
        for column in ("signal_date", "feature_cutoff", "train_cutoff"):
            frame[column] = pd.to_datetime(frame[column]).dt.date
        frame = frame[
            (frame["signal_date"] == context.signal_date)
            & frame["instrument_id"].astype(str).isin(universe.eligible)
        ].copy()
        if not frame.empty:
            if set(frame["data_fingerprint"].astype(str)) != {
                context.data.metadata.fingerprint
            }:
                raise StrategyError("prediction data fingerprint mismatch")
            if (frame["feature_cutoff"] > context.cutoff).any():
                raise StrategyError("prediction feature cutoff exceeds strategy cutoff")
            if (frame["train_cutoff"] > context.cutoff).any():
                raise StrategyError("prediction training cutoff exceeds strategy cutoff")
        frame["source_id"] = frame["model_id"].astype(str)
        frame["source_version"] = frame["model_version"].astype(str)
        return _validated(frame)


@dataclass(frozen=True)
class FactorSignalGenerator:
    factors: Mapping[BaseFactor, float]
    processors: tuple[str, ...] = ("zscore",)

    def generate(self, context: StrategyContext, universe: UniverseSnapshot) -> SignalFrame:
        if context.factor_engine is None:
            raise StrategyError("factor signal requires a FactorEngine")
        merged = None
        cache_keys = []
        for factor, weight in self.factors.items():
            observation_date = min(context.signal_date, context.cutoff)
            result = context.factor_engine.compute(
                factor, context.data, observation_date, observation_date,
                universe.eligible, context.cutoff,
            )
            cache_keys.append(result.cache_key)
            part = result.frame.rename(columns={"value": factor.name})
            merged = part if merged is None else merged.merge(
                part, on=["trade_date", "instrument_id"], how="inner"
            )
        if merged is None:
            merged = pd.DataFrame(columns=["trade_date", "instrument_id"])
        columns = [factor.name for factor in self.factors]
        for processor in self.processors:
            if processor == "zscore":
                for column in columns:
                    std = merged[column].std(ddof=0)
                    merged[column] = 0.0 if not std or pd.isna(std) else (
                        merged[column] - merged[column].mean()
                    ) / std
            elif processor == "rank":
                for column in columns:
                    merged[column] = merged[column].rank(pct=True, method="first")
            elif processor == "winsorize":
                for column in columns:
                    low, high = merged[column].quantile([0.01, 0.99])
                    merged[column] = merged[column].clip(low, high)
            else:
                raise StrategyError(f"unknown signal processor: {processor}")
        merged["score"] = sum(merged[factor.name] * weight for factor, weight in self.factors.items())
        merged["signal_date"] = context.signal_date
        merged["source_id"] = "factor"
        merged["source_version"] = hash_payload(cache_keys)
        return _validated(merged[
            ["signal_date", "instrument_id", "score", "source_id", "source_version"]
        ])


def _validated(frame: pd.DataFrame) -> SignalFrame:
    required = {"signal_date", "instrument_id", "score", "source_id", "source_version"}
    if required - set(frame.columns):
        raise StrategyError(f"signal frame missing columns: {sorted(required - set(frame.columns))}")
    if frame.duplicated(["signal_date", "instrument_id"]).any():
        raise StrategyError("signal frame contains duplicate keys")
    scores = pd.to_numeric(frame["score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores).all():
        raise StrategyError("signal frame contains non-finite scores")
    result = frame.copy()
    result["score"] = scores.astype(float)
    result["instrument_id"] = result["instrument_id"].astype(str)
    result.sort_values(["score", "instrument_id"], ascending=[False, True], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return SignalFrame(result, hash_payload(result.to_dict("records")))
