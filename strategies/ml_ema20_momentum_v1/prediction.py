from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from zyquant.core.exceptions import StrategyError
from zyquant.core.hashing import hash_file, hash_payload


REQUIRED_COLUMNS = {
    "signal_date", "instrument_id", "score", "model_id", "model_version",
    "feature_cutoff", "train_cutoff", "dataset_id", "data_fingerprint",
    "feature_set_id",
}


@dataclass(frozen=True)
class PredictionBook:
    frame: pd.DataFrame
    fingerprint: str
    source_path: Path

    @classmethod
    def load(cls, path: str | Path, snapshot) -> "PredictionBook":
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise StrategyError(f"prediction file does not exist: {source}")
        frame = pd.read_parquet(source)
        missing = REQUIRED_COLUMNS - set(frame)
        if missing:
            raise StrategyError(
                f"prediction file missing columns: {sorted(missing)}"
            )
        result = frame.copy()
        for column in ("signal_date", "feature_cutoff", "train_cutoff"):
            result[column] = pd.to_datetime(
                result[column], errors="raise"
            ).dt.date
        result["instrument_id"] = result["instrument_id"].astype(str)
        text_columns = (
            "instrument_id", "model_id", "model_version", "dataset_id",
            "data_fingerprint", "feature_set_id",
        )
        for column in text_columns:
            if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
                raise StrategyError(f"prediction {column} must not be empty")
        result["score"] = pd.to_numeric(result["score"], errors="coerce")
        if result["score"].isna().any() or not np.isfinite(result["score"]).all():
            raise StrategyError("prediction scores must be finite")
        if result.duplicated(["signal_date", "instrument_id"]).any():
            raise StrategyError("prediction file contains duplicate keys")
        if set(result["dataset_id"].astype(str)) != {
            snapshot.metadata.dataset_id
        }:
            raise StrategyError("prediction dataset_id does not match snapshot")
        if set(result["data_fingerprint"].astype(str)) != {
            snapshot.metadata.fingerprint
        }:
            raise StrategyError(
                "prediction data_fingerprint does not match snapshot"
            )
        if (result["feature_cutoff"] > result["signal_date"]).any():
            raise StrategyError("prediction feature cutoff exceeds signal date")
        if (result["train_cutoff"] > result["signal_date"]).any():
            raise StrategyError("prediction train cutoff exceeds signal date")
        as_of = snapshot.metadata.as_of_date
        if (result["signal_date"] > as_of).any():
            raise StrategyError("prediction signal date exceeds snapshot as-of date")
        result.sort_values(
            ["signal_date", "score", "instrument_id"],
            ascending=[True, False, True], kind="mergesort", inplace=True,
            ignore_index=True,
        )
        fingerprint = hash_payload({
            "file_sha256": hash_file(source),
            "rows": len(result),
            "dataset": snapshot.metadata.fingerprint,
        })
        return cls(result, fingerprint, source)

    def on(self, day: date, eligible: tuple[str, ...]) -> pd.DataFrame:
        allowed = set(eligible)
        result = self.frame[
            (self.frame["signal_date"] == day)
            & self.frame["instrument_id"].isin(allowed)
        ].copy()
        result.sort_values(
            ["score", "instrument_id"], ascending=[False, True],
            kind="mergesort", inplace=True, ignore_index=True,
        )
        return result
