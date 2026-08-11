from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from zyquant.core.exceptions import StrategyError
from zyquant.core.hashing import hash_file, hash_payload
from zyquant.data import DataSnapshot


@dataclass(frozen=True)
class TrainingDataset:
    features: pd.DataFrame
    labels: pd.Series
    index: pd.DataFrame
    label_start_dates: pd.Series
    label_end_dates: pd.Series
    feature_columns: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    path: Path
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FeatureSetDefinition:
    feature_set_id: str
    version: str
    features: tuple[str, ...]
    preprocessing: tuple[str, ...] = ()


@dataclass(frozen=True)
class LabelDefinition:
    label_id: str
    version: str
    horizon: int
    price_field: str = "close_post"


@dataclass(frozen=True)
class PredictionFrame:
    frame: pd.DataFrame
    fingerprint: str


class ModelTrainer(Protocol):
    def fit(
        self,
        dataset: TrainingDataset,
        train_indexes: Sequence[int],
        validation_indexes: Sequence[int],
        artifact_path: Path,
    ) -> Any: ...


class Predictor(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class DatasetBuilder:
    """Build PIT feature/label panels with configurable executable prices.

    The legacy default remains close-to-future-close.  For a signal produced
    at T close, entered at T+1 open and exited at T+2 close, use
    ``entry_offset=1``, ``horizon=1``, ``entry_price_field="open_post"`` and
    ``exit_price_field="close_post"``.
    """

    def build(
        self,
        snapshot: DataSnapshot,
        feature_frames: Mapping[str, pd.DataFrame],
        start: date,
        end: date,
        horizon: int = 1,
        instruments: Sequence[str] | None = None,
        cutoff: date | None = None,
        entry_offset: int = 0,
        entry_price_field: str = "close_post",
        exit_price_field: str = "close_post",
    ) -> TrainingDataset:
        if horizon < 1:
            raise ValueError("label horizon must be positive")
        if entry_offset < 0:
            raise ValueError("label entry offset must not be negative")
        merged = None
        for name, source in feature_frames.items():
            part = source[["trade_date", "instrument_id", "value"]].rename(
                columns={"value": name}
            )
            merged = part if merged is None else merged.merge(
                part, on=["trade_date", "instrument_id"], how="inner"
            )
        if merged is None or merged.empty:
            raise StrategyError("feature frames produced an empty dataset")
        cutoff = cutoff or end
        price_fields = list(dict.fromkeys(
            [entry_price_field, exit_price_field]
        ))
        prices = snapshot.post_adjusted_bars(
            start, end, instruments, price_fields, cutoff=cutoff
        ).sort_values(["instrument_id", "trade_date"])
        grouped = prices.groupby("instrument_id", sort=False)
        exit_offset = entry_offset + horizon
        entry_price = grouped[entry_price_field].shift(-entry_offset)
        exit_price = grouped[exit_price_field].shift(-exit_offset)
        prices["label"] = exit_price / entry_price - 1
        prices["label_start_date"] = grouped["trade_date"].shift(-entry_offset)
        prices["label_end_date"] = grouped["trade_date"].shift(-exit_offset)
        dataset = merged.merge(
            prices[[
                "trade_date", "instrument_id", "label",
                "label_start_date", "label_end_date",
            ]],
            on=["trade_date", "instrument_id"], how="inner",
        ).dropna()
        dataset = dataset[dataset["label_end_date"] <= cutoff].copy()
        feature_columns = tuple(feature_frames)
        dataset.sort_values(["trade_date", "instrument_id"], inplace=True, ignore_index=True)
        fingerprint = hash_payload({
            "dataset": snapshot.metadata.fingerprint,
            "features": feature_columns,
            "start": start, "end": end, "horizon": horizon,
            "entry_offset": entry_offset,
            "entry_price_field": entry_price_field,
            "exit_price_field": exit_price_field,
            "rows": dataset[["trade_date", "instrument_id", "label"]].to_dict("records"),
        })
        return TrainingDataset(
            dataset[list(feature_columns)].reset_index(drop=True),
            dataset["label"].reset_index(drop=True),
            dataset[["trade_date", "instrument_id"]].reset_index(drop=True),
            dataset["label_start_date"].reset_index(drop=True),
            dataset["label_end_date"].reset_index(drop=True),
            feature_columns,
            fingerprint,
        )


@dataclass(frozen=True)
class PurgedTimeSeriesSplitter:
    folds: int = 5
    embargo_periods: int = 1

    def split(self, dataset: TrainingDataset) -> tuple[tuple[list[int], list[int]], ...]:
        dates = dataset.index["trade_date"]
        unique = sorted(dates.unique())
        if len(unique) < 2:
            return ()
        chunks = [list(chunk) for chunk in np.array_split(unique, min(self.folds, len(unique))) if len(chunk)]
        result = []
        for validation_dates in chunks[1:]:
            validation_start = min(validation_dates)
            validation = dates.isin(validation_dates)
            training = (dates < validation_start) & (dataset.label_end_dates < validation_start)
            if self.embargo_periods:
                before = [day for day in unique if day < validation_start]
                embargo = set(before[-self.embargo_periods:])
                training &= ~dates.isin(embargo)
            train_indexes = list(np.flatnonzero(training.to_numpy()))
            valid_indexes = list(np.flatnonzero(validation.to_numpy()))
            if train_indexes and valid_indexes:
                result.append((train_indexes, valid_indexes))
        return tuple(result)


def validate_predictions(
    frame: pd.DataFrame,
    snapshot: DataSnapshot,
    strategy_id: str,
) -> PredictionFrame:
    required = {
        "signal_date", "instrument_id", "score", "model_id", "feature_cutoff",
        "train_cutoff", "dataset_id", "feature_set_id",
        "data_fingerprint", "model_version",
    }
    missing = required - set(frame.columns)
    if missing:
        raise StrategyError(f"prediction frame missing columns: {sorted(missing)}")
    result = frame.copy()
    for column in ("signal_date", "feature_cutoff", "train_cutoff"):
        result[column] = pd.to_datetime(result[column], errors="raise").dt.date
    if set(result["dataset_id"].astype(str)) != {snapshot.metadata.dataset_id}:
        raise StrategyError("prediction dataset_id does not match data snapshot")
    if set(result["data_fingerprint"].astype(str)) != {snapshot.metadata.fingerprint}:
        raise StrategyError("prediction data_fingerprint does not match data snapshot")
    if (result["feature_cutoff"] > result["signal_date"]).any():
        raise StrategyError("prediction uses features after signal date")
    if (result["train_cutoff"] > result["signal_date"]).any():
        raise StrategyError("prediction training cutoff exceeds signal date")
    if result.duplicated(["signal_date", "instrument_id"]).any():
        raise StrategyError("prediction frame contains duplicate keys")
    scores = pd.to_numeric(result["score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores).all():
        raise StrategyError("prediction scores must be finite")
    result["strategy_id"] = strategy_id
    result.sort_values(["signal_date", "instrument_id"], inplace=True, ignore_index=True)
    return PredictionFrame(result, hash_payload(result.to_dict("records")))


@dataclass
class StandardPreprocessor:
    """Train-only finite-value imputation and standardization."""

    fill_values: dict[str, float] | None = None
    means: dict[str, float] | None = None
    scales: dict[str, float] | None = None

    def fit(self, frame: pd.DataFrame) -> "StandardPreprocessor":
        numeric = frame.astype(float)
        self.fill_values = {
            column: float(numeric[column].median())
            if numeric[column].notna().any() else 0.0
            for column in numeric
        }
        filled = numeric.fillna(self.fill_values)
        self.means = {column: float(filled[column].mean()) for column in filled}
        self.scales = {
            column: float(filled[column].std(ddof=0)) or 1.0 for column in filled
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.fill_values is None or self.means is None or self.scales is None:
            raise StrategyError("preprocessor must be fit on a training set")
        result = frame.astype(float).fillna(self.fill_values)
        for column in result:
            result[column] = (
                result[column] - self.means[column]
            ) / self.scales[column]
        return result


class ModelRegistry:
    """Immutable local model registry with atomic metadata and model commits."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        model_id: str,
        model: Any,
        metadata: Mapping[str, Any],
    ) -> ModelArtifact:
        final = self.root / model_id
        if final.exists():
            raise StrategyError(f"immutable model already exists: {model_id}")
        staging = Path(tempfile.mkdtemp(prefix=f".{model_id}.", dir=self.root))
        try:
            model_path = staging / "model.pkl"
            with model_path.open("wb") as stream:
                pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
            payload = {
                "schema_version": "1.0",
                "model_id": model_id,
                **dict(metadata),
                "model_sha256": hash_file(model_path),
            }
            (staging / "metadata.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(staging, final)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return ModelArtifact(model_id, final / "model.pkl", payload)

    def load(self, model_id: str) -> tuple[Any, ModelArtifact]:
        directory = self.root / model_id
        metadata_path = directory / "metadata.json"
        model_path = directory / "model.pkl"
        if not metadata_path.exists() or not model_path.exists():
            raise StrategyError(f"unknown model artifact: {model_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata["model_sha256"]
        actual = hash_file(model_path)
        if expected != actual:
            raise StrategyError(f"model artifact hash mismatch: {model_id}")
        with model_path.open("rb") as stream:
            model = pickle.load(stream)
        return model, ModelArtifact(model_id, model_path, metadata)


class SklearnTrainer:
    """Optional sklearn reference adapter; sklearn is imported only when used."""

    def __init__(self, estimator: Any, preprocessor: StandardPreprocessor | None = None):
        self.estimator = estimator
        self.preprocessor = preprocessor or StandardPreprocessor()

    def fit(
        self,
        dataset: TrainingDataset,
        train_indexes: Sequence[int],
        validation_indexes: Sequence[int],
        artifact_path: Path,
    ) -> Mapping[str, Any]:
        try:
            from sklearn.base import clone
        except ImportError as exc:
            raise StrategyError("SklearnTrainer requires the 'ml' optional dependency") from exc
        x_train = dataset.features.iloc[list(train_indexes)]
        x_valid = dataset.features.iloc[list(validation_indexes)]
        processor = StandardPreprocessor().fit(x_train)
        estimator = clone(self.estimator)
        estimator.fit(processor.transform(x_train), dataset.labels.iloc[list(train_indexes)])
        prediction = np.asarray(estimator.predict(processor.transform(x_valid)), dtype=float)
        target = dataset.labels.iloc[list(validation_indexes)].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        bundle = {"estimator": estimator, "preprocessor": processor}
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("wb") as stream:
            pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return {"model": bundle, "validation_rmse": rmse}


class RollingModelTrainer:
    """Train and register one immutable model per purged rolling split."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def train(
        self,
        dataset: TrainingDataset,
        splitter: PurgedTimeSeriesSplitter,
        trainer: ModelTrainer,
        model_prefix: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ModelArtifact, ...]:
        artifacts = []
        for fold, (train_indexes, validation_indexes) in enumerate(
            splitter.split(dataset), start=1
        ):
            train_cutoff = max(dataset.label_end_dates.iloc[train_indexes])
            model_id = f"{model_prefix}-fold-{fold:03d}-{train_cutoff}"
            with tempfile.TemporaryDirectory() as temporary:
                outcome = trainer.fit(
                    dataset, train_indexes, validation_indexes,
                    Path(temporary) / "model.pkl",
                )
            model = outcome.get("model", outcome) if isinstance(outcome, Mapping) else outcome
            metrics = (
                {key: value for key, value in outcome.items() if key != "model"}
                if isinstance(outcome, Mapping) else {}
            )
            artifacts.append(self.registry.register(model_id, model, {
                **dict(metadata or {}),
                "dataset_fingerprint": dataset.fingerprint,
                "train_cutoff": train_cutoff,
                "validation_start": min(
                    dataset.index.iloc[validation_indexes]["trade_date"]
                ),
                "metrics": metrics,
            }))
        return tuple(artifacts)


def make_prediction_frame(
    model: Any,
    features: pd.DataFrame,
    index: pd.DataFrame,
    snapshot: DataSnapshot,
    model_id: str,
    model_version: str,
    feature_set_id: str,
    feature_cutoff: date,
    train_cutoff: date,
    strategy_id: str,
) -> PredictionFrame:
    bundle = model
    estimator = bundle["estimator"] if isinstance(bundle, Mapping) else bundle
    processor = bundle.get("preprocessor") if isinstance(bundle, Mapping) else None
    transformed = processor.transform(features) if processor is not None else features
    score = np.asarray(estimator.predict(transformed), dtype=float)
    result = index.rename(columns={"trade_date": "signal_date"}).copy()
    result["score"] = score
    result["model_id"] = model_id
    result["model_version"] = model_version
    result["feature_cutoff"] = feature_cutoff
    result["train_cutoff"] = train_cutoff
    result["dataset_id"] = snapshot.metadata.dataset_id
    result["data_fingerprint"] = snapshot.metadata.fingerprint
    result["feature_set_id"] = feature_set_id
    return validate_predictions(result, snapshot, strategy_id)
