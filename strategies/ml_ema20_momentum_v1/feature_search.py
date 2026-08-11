from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from zyquant.core.exceptions import StrategyError
from zyquant.core.hashing import hash_file, hash_payload
from zyquant.experiment.store import ExperimentStore

from .dataset import LABEL_COLUMN, MODEL_FEATURES
from .xgb_training import (
    fit_early_stopping,
    make_relevance,
    predict,
    prepare_fold,
    preprocess_daily,
    refit_full,
)


SEED = 20260811
PRIMARY_TRIALS = 10_000
MIN_FEATURES = 5
MAX_FEATURES = 30
DEVELOPMENT_YEARS = tuple(range(2015, 2023))
SEARCH_VERSION = "xgb_feature_subset_10k_v1"
DEFAULT_DATASET_ROOT = Path(
    "/data/zzh/ZyQuant/runs/ml_ema20_momentum_v1/datasets/"
    "rolling_3y_1y_v2_clean"
)
DEFAULT_ROOT = Path(
    "/data/zzh/ZyQuant/runs/ml_ema20_momentum_v1/feature_search/"
    f"{SEARCH_VERSION}"
)
DEFAULT_DATABASE = Path("/data/zzh/ZyQuant/runs/experiments.sqlite")
TERMINAL_STATUSES = {"succeeded", "technical_failed"}


@dataclass(frozen=True)
class FeatureTrial:
    sequence: int
    trial_id: str
    trial_key: str
    features: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["features"] = list(self.features)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "FeatureTrial":
        return cls(
            sequence=int(payload["sequence"]),
            trial_id=str(payload["trial_id"]),
            trial_key=str(payload["trial_key"]),
            features=tuple(map(str, payload["features"])),
        )


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def search_code_fingerprint() -> str:
    root = Path(__file__).parent
    files = (
        root / "feature_search.py",
        root / "xgb_training.py",
        root / "dataset.py",
    )
    return hash_payload({path.name: hash_file(path) for path in files})


def generate_feature_trials(
    count: int = PRIMARY_TRIALS,
    *,
    seed: int = SEED,
    minimum: int = MIN_FEATURES,
    maximum: int = MAX_FEATURES,
) -> list[FeatureTrial]:
    if not 1 <= minimum <= maximum <= len(MODEL_FEATURES):
        raise ValueError("invalid feature subset bounds")
    rng = np.random.default_rng(seed)
    feature_order = {feature: index for index, feature in enumerate(MODEL_FEATURES)}
    seen: set[tuple[str, ...]] = set()
    result: list[FeatureTrial] = []
    while len(result) < count:
        size = int(rng.integers(minimum, maximum + 1))
        chosen = rng.choice(MODEL_FEATURES, size=size, replace=False).tolist()
        features = tuple(sorted(map(str, chosen), key=feature_order.__getitem__))
        if features in seen:
            continue
        seen.add(features)
        sequence = len(result) + 1
        trial_key = hash_payload({
            "version": SEARCH_VERSION,
            "seed": seed,
            "features": features,
        })
        result.append(FeatureTrial(
            sequence=sequence,
            trial_id=f"mfs-{sequence:05d}-{trial_key[:8]}",
            trial_key=trial_key,
            features=features,
        ))
    return result


def search_objective(annual: Sequence[Mapping[str, Any]]) -> float:
    excess = np.asarray(
        [float(item["top3_mean_return"]) - float(item["pool_mean_return"])
         for item in annual],
        dtype=float,
    )
    if len(excess) != len(DEVELOPMENT_YEARS) or not np.isfinite(excess).all():
        raise StrategyError("feature-search annual metrics are incomplete or non-finite")
    return float(excess.mean() - 0.5 * excess.std(ddof=0))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_trials(path: Path) -> list[FeatureTrial]:
    return [
        FeatureTrial.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prepare_search(root: Path, dataset_root: Path) -> dict[str, Any]:
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise StrategyError(f"immutable search root is not empty: {root}")
    dataset_root = dataset_root.resolve()
    dataset_manifest = json.loads(
        (dataset_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if dataset_manifest.get("dataset_version") != "rolling_3y_1y_v2_clean":
        raise StrategyError("feature search requires rolling_3y_1y_v2_clean")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    began = time.perf_counter()
    try:
        panel = pd.read_parquet(dataset_root / "labeled_panel.parquet")
        required = {
            "signal_date", "label_end_date", "instrument_id", LABEL_COLUMN,
            *MODEL_FEATURES,
        }
        missing = required - set(panel)
        if missing:
            raise StrategyError(f"clean panel is missing columns: {sorted(missing)}")
        panel = preprocess_daily(panel)
        panel["relevance"] = make_relevance(panel)
        values = panel[list(MODEL_FEATURES)].to_numpy(dtype=float)
        labels = pd.to_numeric(panel[LABEL_COLUMN], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or not np.isfinite(labels).all():
            raise StrategyError("prepared search panel contains NaN or infinity")
        panel.sort_values(
            ["signal_date", "instrument_id"], kind="mergesort", inplace=True,
            ignore_index=True,
        )
        cache_path = staging / "preprocessed_panel.parquet"
        panel.to_parquet(cache_path, index=False)
        trials = generate_feature_trials()
        _write_jsonl(staging / "trials.jsonl", (item.as_json() for item in trials))
        frequencies = Counter(
            feature for trial in trials for feature in trial.features
        )
        size_counts = Counter(len(item.features) for item in trials)
        frozen = {
            "search_version": SEARCH_VERSION,
            "seed": SEED,
            "trial_count": len(trials),
            "minimum_features": MIN_FEATURES,
            "maximum_features": MAX_FEATURES,
            "development_years": DEVELOPMENT_YEARS,
            "final_holdout_years": [2023, 2024, 2025, 2026],
            "dataset_root": str(dataset_root),
            "dataset_panel_fingerprint": dataset_manifest["panel_fingerprint"],
            "snapshot_fingerprint": dataset_manifest["snapshot_fingerprint"],
            "source_features": MODEL_FEATURES,
            "feature_frequency": dict(sorted(frequencies.items())),
            "subset_size_distribution": dict(sorted(size_counts.items())),
            "preprocessed_panel_sha256": hash_file(cache_path),
            "trials_sha256": hash_file(staging / "trials.jsonl"),
            "code_fingerprint": search_code_fingerprint(),
            "prepared_at": utcnow(),
            "prepare_seconds": round(time.perf_counter() - began, 3),
            "evaluation": {
                "protocol": "rolling_3y_train_1y_test_on_development_years",
                "validation_sessions": 126,
                "purge": "label_end_date < validation_start",
                "embargo_sessions": 2,
                "max_estimators": 500,
                "early_stopping_rounds": 100,
                "objective": "mean_annual_top3_excess_minus_half_std",
            },
        }
        search_id = (
            f"ml-ema20-feature-search-10k-v1-"
            f"{hash_payload(frozen)[:12]}"
        )
        frozen["search_id"] = search_id
        _write_json(staging / "manifest.json", frozen)
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return frozen


_WORKER_FOLDS: dict[int, Any] = {}
_WORKER_FEATURE_THREADS = 1
_WORKER_DEVICE = "cuda"
_WORKER_MAX_ESTIMATORS = 500


def _worker_initializer(
    panel_path: str,
    device: str,
    feature_threads: int,
    max_estimators: int,
) -> None:
    global _WORKER_FOLDS, _WORKER_FEATURE_THREADS
    global _WORKER_DEVICE, _WORKER_MAX_ESTIMATORS
    os.environ.setdefault("OMP_NUM_THREADS", str(feature_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(feature_threads))
    panel = pd.read_parquet(panel_path)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"]).dt.date
    panel["label_end_date"] = pd.to_datetime(panel["label_end_date"]).dt.date
    year = pd.Series([item.year for item in panel["signal_date"]], index=panel.index)
    folds = {}
    for test_year in DEVELOPMENT_YEARS:
        train = panel[year.between(test_year - 3, test_year - 1)].copy()
        test = panel[year == test_year].copy()
        folds[test_year] = prepare_fold(train, test)
    _WORKER_FOLDS = folds
    _WORKER_FEATURE_THREADS = int(feature_threads)
    _WORKER_DEVICE = device
    _WORKER_MAX_ESTIMATORS = int(max_estimators)


def _annual_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    working = frame[[
        "signal_date", "instrument_id", LABEL_COLUMN, "relevance",
    ]].copy()
    working["score"] = scores
    ranked = working.sort_values(
        ["signal_date", "score", "instrument_id"],
        ascending=[True, False, True], kind="mergesort",
    )
    top1 = ranked.groupby("signal_date", sort=False).head(1)
    top3 = ranked.groupby("signal_date", sort=False).head(3).copy()
    top3["rank"] = top3.groupby("signal_date", sort=False).cumcount()
    discounts = np.asarray([1.0, 1.0 / np.log2(3.0), 0.5])
    top3["dcg"] = (
        np.power(2.0, top3["relevance"].to_numpy(dtype=float)) - 1.0
    ) * discounts[top3["rank"].to_numpy(dtype=int)]
    ideal = working.sort_values(
        ["signal_date", "relevance", "instrument_id"],
        ascending=[True, False, True], kind="mergesort",
    ).groupby("signal_date", sort=False).head(3).copy()
    ideal["rank"] = ideal.groupby("signal_date", sort=False).cumcount()
    ideal["idcg"] = (
        np.power(2.0, ideal["relevance"].to_numpy(dtype=float)) - 1.0
    ) * discounts[ideal["rank"].to_numpy(dtype=int)]
    dcg = top3.groupby("signal_date", sort=False)["dcg"].sum()
    idcg = ideal.groupby("signal_date", sort=False)["idcg"].sum()
    ndcg = (dcg / idcg.where(idcg > 0)).fillna(0.0)
    return {
        "rows": len(working),
        "dates": int(working["signal_date"].nunique()),
        "ndcg_at_3": float(ndcg.mean()),
        "pool_mean_return": float(working[LABEL_COLUMN].mean()),
        "top1_mean_return": float(top1[LABEL_COLUMN].mean()),
        "top3_mean_return": float(top3[LABEL_COLUMN].mean()),
    }


def _worker_run(payload: Mapping[str, Any]) -> dict[str, Any]:
    trial = FeatureTrial.from_json(payload)
    began = time.perf_counter()
    annual = []
    capped = 0
    iterations = []
    for year in DEVELOPMENT_YEARS:
        fold = _WORKER_FOLDS[year]
        _, best, validation_ndcg = fit_early_stopping(
            fold, _WORKER_DEVICE, trial.features,
            max_estimators=_WORKER_MAX_ESTIMATORS,
            n_jobs=_WORKER_FEATURE_THREADS,
        )
        if best >= _WORKER_MAX_ESTIMATORS:
            capped += 1
        model = refit_full(
            fold, _WORKER_DEVICE, best, trial.features,
            n_jobs=_WORKER_FEATURE_THREADS,
        )
        scores = predict(model, fold.test, trial.features)
        metrics = _annual_metrics(fold.test, scores)
        metrics.update({
            "test_year": year,
            "best_iterations": best,
            "validation_ndcg_at_3": validation_ndcg,
        })
        annual.append(metrics)
        iterations.append(best)
    objective = search_objective(annual)
    excess = [
        item["top3_mean_return"] - item["pool_mean_return"]
        for item in annual
    ]
    metrics = {
        "objective": objective,
        "annual": annual,
        "mean_top3_return": float(np.mean([
            item["top3_mean_return"] for item in annual
        ])),
        "mean_pool_return": float(np.mean([
            item["pool_mean_return"] for item in annual
        ])),
        "mean_top3_excess": float(np.mean(excess)),
        "std_top3_excess": float(np.std(excess, ddof=0)),
        "minimum_top3_excess": float(np.min(excess)),
        "positive_excess_years": int(np.count_nonzero(np.asarray(excess) > 0)),
        "mean_ndcg_at_3": float(np.mean([
            item["ndcg_at_3"] for item in annual
        ])),
        "mean_best_iterations": float(np.mean(iterations)),
        "capped_folds": capped,
        "seconds": round(time.perf_counter() - began, 3),
    }
    return {"trial": trial.as_json(), "metrics": metrics}


class FeatureSearchController:
    def __init__(
        self,
        root: Path,
        database: Path,
        *,
        workers: int,
        device: str,
        feature_threads: int,
    ):
        self.root = root.resolve()
        self.database = database.resolve()
        self.workers = int(workers)
        self.device = device
        self.feature_threads = int(feature_threads)
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.search_id = str(self.manifest["search_id"])
        self.trials = _load_trials(self.root / "trials.jsonl")
        self.store = ExperimentStore(self.database)
        self.stop_requested = False
        self.state_path = self.root / "controller.json"

    def close(self) -> None:
        self.store.close()

    def _write_state(self, **updates: Any) -> None:
        state: dict[str, Any] = {}
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state.update({
            "search_id": self.search_id,
            "controller_pid": os.getpid(),
            "heartbeat": utcnow(),
            **updates,
        })
        _write_json(self.state_path, state)

    def _ensure_run(self) -> None:
        existing = self.store.get_run(self.search_id)
        config = {
            "manifest": str(self.root / "manifest.json"),
            "workers": self.workers,
            "device": self.device,
            "feature_threads": self.feature_threads,
        }
        if existing is None:
            self.store.start_run(
                self.search_id, "search", config,
                {"strategy": "ml_ema20_momentum_v1", "kind": SEARCH_VERSION},
            )
        elif existing["status"] != "running":
            self.store.connection.execute(
                "UPDATE runs SET status='running', completed_at=NULL, error=NULL, "
                "updated_at=? WHERE run_id=?", (utcnow(), self.search_id),
            )
            self.store.connection.commit()

    def _initialize_trials(self) -> None:
        existing = {
            row["trial_key"] for row in self.store.connection.execute(
                "SELECT trial_key FROM trials WHERE search_run_id=?",
                (self.search_id,),
            )
        }
        rows = [(
            item.trial_key, self.search_id, item.trial_id, "queued",
            json.dumps(item.as_json(), ensure_ascii=False, sort_keys=True),
            "{}", None, None, 0, utcnow(),
        ) for item in self.trials if item.trial_key not in existing]
        for offset in range(0, len(rows), 1000):
            self.store.connection.executemany(
                "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows[offset:offset + 1000],
            )
            self.store.connection.commit()

    def _pending(self) -> list[FeatureTrial]:
        statuses = {
            row["trial_key"]: str(row["status"])
            for row in self.store.connection.execute(
                "SELECT trial_key,status FROM trials WHERE search_run_id=?",
                (self.search_id,),
            )
        }
        return [
            item for item in self.trials
            if statuses.get(item.trial_key) not in TERMINAL_STATUSES
        ]

    def run(self) -> int:
        if self.manifest["code_fingerprint"] != search_code_fingerprint():
            raise StrategyError("search code changed after manifest was frozen")
        self._ensure_run()
        self._initialize_trials()
        self.store.connection.execute(
            "UPDATE trials SET status='interrupted',heartbeat_at=? "
            "WHERE search_run_id=? AND status='running'",
            (utcnow(), self.search_id),
        )
        self.store.connection.commit()
        queue = deque(self._pending())
        initial_pending = len(queue)
        completed = failed = 0
        began = time.perf_counter()
        panel_path = str(self.root / "preprocessed_panel.parquet")
        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_worker_initializer,
            initargs=(
                panel_path, self.device, self.feature_threads,
                int(self.manifest["evaluation"]["max_estimators"]),
            ),
        ) as pool:
            futures: dict[Any, FeatureTrial] = {}
            while queue or futures:
                if (self.root / "STOP").exists():
                    self.stop_requested = True
                while queue and len(futures) < self.workers and not self.stop_requested:
                    item = queue.popleft()
                    current = self.store.get_trial(item.trial_key)
                    attempts = int(current["attempts"]) + 1 if current else 1
                    self.store.upsert_trial(
                        item.trial_key, self.search_id, item.trial_id, "running",
                        item.as_json(), attempts=attempts,
                    )
                    futures[pool.submit(_worker_run, item.as_json())] = item
                if not futures:
                    break
                done, _ = wait(futures, timeout=5.0, return_when=FIRST_COMPLETED)
                self.store.heartbeat([item.trial_key for item in futures.values()])
                for future in done:
                    item = futures.pop(future)
                    current = self.store.get_trial(item.trial_key)
                    attempts = int(current["attempts"]) if current else 1
                    try:
                        result = future.result()
                        metrics = result["metrics"]
                        self.store.upsert_trial(
                            item.trial_key, self.search_id, item.trial_id,
                            "succeeded", item.as_json(),
                            objective=float(metrics["objective"]), metrics=metrics,
                            attempts=attempts,
                        )
                        completed += 1
                    except Exception as exc:
                        failed += 1
                        if attempts < 3 and not self.stop_requested:
                            self.store.upsert_trial(
                                item.trial_key, self.search_id, item.trial_id,
                                "interrupted", item.as_json(), error=repr(exc),
                                attempts=attempts,
                            )
                            queue.append(item)
                        else:
                            self.store.upsert_trial(
                                item.trial_key, self.search_id, item.trial_id,
                                "technical_failed", item.as_json(), error=repr(exc),
                                attempts=attempts,
                            )
                elapsed = max(time.perf_counter() - began, 1e-9)
                self._write_state(
                    status="stopping" if self.stop_requested else "running",
                    workers=self.workers,
                    active=len(futures),
                    queued=len(queue),
                    completed_this_process=completed,
                    failures_this_process=failed,
                    trials_per_hour=completed / elapsed * 3600.0,
                    initial_pending=initial_pending,
                )
            if self.stop_requested:
                for future in futures:
                    future.cancel()
        if self.stop_requested:
            self.store.interrupt_running_trials(self.search_id)
            self.store.fail_run(
                self.search_id, "graceful stop requested", "interrupted",
            )
            self._write_state(status="interrupted", workers=0, active=0)
            return 130
        counts = status_counts(self.store, self.search_id)
        if counts.get("technical_failed", 0):
            self.store.fail_run(
                self.search_id,
                f"persistent technical failures: {counts['technical_failed']}",
            )
            self._write_state(status="failed", workers=0, active=0, counts=counts)
            return 1
        self.store.finish_run(self.search_id, counts)
        generate_report(self.root, self.database)
        self._write_state(status="succeeded", workers=0, active=0, counts=counts)
        return 0


def status_counts(store: ExperimentStore, search_id: str) -> dict[str, int]:
    return {
        str(row["status"]): int(row["n"])
        for row in store.connection.execute(
            "SELECT status,COUNT(*) n FROM trials WHERE search_run_id=? "
            "GROUP BY status", (search_id,),
        )
    }


def search_status(root: Path, database: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    with ExperimentStore(database) as store:
        counts = status_counts(store, str(manifest["search_id"]))
    state = {}
    if (root / "controller.json").exists():
        state = json.loads((root / "controller.json").read_text(encoding="utf-8"))
    return {"search_id": manifest["search_id"], "counts": counts, **state}


def generate_report(root: Path, database: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    search_id = str(manifest["search_id"])
    records = []
    with ExperimentStore(database) as store:
        rows = store.connection.execute(
            "SELECT * FROM trials WHERE search_run_id=? AND status='succeeded'",
            (search_id,),
        ).fetchall()
        for row in rows:
            trial = FeatureTrial.from_json(json.loads(row["parameters_json"]))
            metrics = json.loads(row["metrics_json"])
            records.append({
                "trial_id": trial.trial_id,
                "trial_key": trial.trial_key,
                "sequence": trial.sequence,
                "feature_count": len(trial.features),
                "features_json": json.dumps(list(trial.features), ensure_ascii=False),
                "objective": float(row["objective"]),
                **{key: metrics[key] for key in (
                    "mean_top3_return", "mean_pool_return", "mean_top3_excess",
                    "std_top3_excess", "minimum_top3_excess",
                    "positive_excess_years", "mean_ndcg_at_3",
                    "mean_best_iterations", "capped_folds", "seconds",
                )},
            })
    frame = pd.DataFrame(records).sort_values(
        ["objective", "mean_ndcg_at_3", "trial_id"],
        ascending=[False, False, True], kind="mergesort", ignore_index=True,
    )
    temporary = root / "trials.tmp.parquet"
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, root / "trials.parquet")
    frame.head(100).to_csv(root / "top100.csv", index=False)
    top = frame.head(100)
    trial_by_id = {item.trial_id: item for item in _load_trials(root / "trials.jsonl")}
    top_frequency = Counter(
        feature
        for trial_id in top["trial_id"]
        for feature in trial_by_id[str(trial_id)].features
    )
    all_frequency = Counter(
        feature for item in trial_by_id.values() for feature in item.features
    )
    feature_rows = [{
        "feature": feature,
        "all_frequency": all_frequency[feature],
        "all_rate": all_frequency[feature] / max(1, len(frame)),
        "top100_frequency": top_frequency[feature],
        "top100_rate": top_frequency[feature] / max(1, len(top)),
        "top100_lift": (
            top_frequency[feature] / max(1, len(top))
            / (all_frequency[feature] / max(1, len(frame)))
        ),
    } for feature in MODEL_FEATURES]
    feature_frame = pd.DataFrame(feature_rows).sort_values(
        ["top100_lift", "top100_frequency", "feature"],
        ascending=[False, False, True], kind="mergesort",
    )
    feature_frame.to_csv(root / "feature_frequency.csv", index=False)
    size_stats = frame.groupby("feature_count").agg(
        trials=("trial_id", "count"),
        mean_objective=("objective", "mean"),
        median_objective=("objective", "median"),
        best_objective=("objective", "max"),
    ).reset_index()
    size_stats.to_csv(root / "subset_size_summary.csv", index=False)
    best = frame.iloc[0].to_dict() if len(frame) else {}
    summary = {
        "search_id": search_id,
        "completed_trials": len(frame),
        "development_years": DEVELOPMENT_YEARS,
        "final_holdout_untouched": [2023, 2024, 2025, 2026],
        "best_trial": best,
        "generated_at": utcnow(),
    }
    _write_json(root / "search_summary.json", summary)
    lines = [
        "# EMA20 XGBoost 随机因子组合搜索", "",
        f"- 搜索ID：`{search_id}`",
        f"- 完成组合：{len(frame):,}",
        "- 组合范围：5–30个因子",
        "- 选择区间：2015–2022年度样本外",
        "- 最终保留区间：2023–2026（未用于选择）",
    ]
    if best:
        lines.extend([
            f"- 最优trial：`{best['trial_id']}`",
            f"- 最优目标值：{best['objective']:.8f}",
            f"- 最优组合因子数：{int(best['feature_count'])}",
            "", "最终保留期结果必须在冻结入选规则后另行计算。",
        ])
    (root / "SEARCH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def benchmark(
    root: Path,
    workers: Sequence[int],
    *,
    trials_per_level: int,
    device: str,
    feature_threads: int,
) -> list[dict[str, Any]]:
    specs = _load_trials(root / "trials.jsonl")
    results = []
    panel_path = str(root / "preprocessed_panel.parquet")
    max_estimators = int(json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )["evaluation"]["max_estimators"])
    for count in workers:
        sample = specs[:trials_per_level]
        began = time.perf_counter()
        seconds = []
        with ProcessPoolExecutor(
            max_workers=count,
            initializer=_worker_initializer,
            initargs=(panel_path, device, feature_threads, max_estimators),
        ) as pool:
            for item in pool.map(_worker_run, (spec.as_json() for spec in sample)):
                seconds.append(float(item["metrics"]["seconds"]))
        elapsed = time.perf_counter() - began
        results.append({
            "workers": count,
            "trials": len(sample),
            "wall_seconds": elapsed,
            "trials_per_hour": len(sample) / elapsed * 3600.0,
            "mean_trial_seconds": float(np.mean(seconds)),
        })
    _write_json(root / "benchmark.json", results)
    return results


def _preflight() -> None:
    if os.uname().nodename != "E5":
        raise StrategyError("formal feature search must run on server E5")
    if Path.cwd().resolve() != Path("/data/zzh/ZyQuant"):
        raise StrategyError("formal feature search must run from /data/zzh/ZyQuant")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Random XGBoost feature subset search")
    parser.add_argument(
        "command", choices=("prepare", "benchmark", "run", "status", "stop", "report"),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--benchmark-workers", default="1,2,4,8")
    parser.add_argument("--benchmark-trials", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--feature-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    if not 1 <= args.feature_threads <= 16:
        parser.error("--feature-threads must be between 1 and 16")
    if args.command in {"prepare", "benchmark", "run"}:
        _preflight()
    if args.command == "prepare":
        print(json.dumps(
            prepare_search(args.root, args.dataset_root),
            ensure_ascii=False, indent=2, default=str,
        ))
        return 0
    if args.command == "benchmark":
        levels = tuple(map(int, args.benchmark_workers.split(",")))
        print(json.dumps(benchmark(
            args.root, levels, trials_per_level=args.benchmark_trials,
            device=args.device, feature_threads=args.feature_threads,
        ), ensure_ascii=False, indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(
            search_status(args.root, args.database), ensure_ascii=False, indent=2,
        ))
        return 0
    if args.command == "stop":
        (args.root / "STOP").write_text(utcnow() + "\n", encoding="utf-8")
        return 0
    if args.command == "report":
        print(json.dumps(
            generate_report(args.root, args.database),
            ensure_ascii=False, indent=2, default=str,
        ))
        return 0
    stop = args.root / "STOP"
    if stop.exists():
        stop.unlink()
    controller = FeatureSearchController(
        args.root, args.database, workers=args.workers,
        device=args.device, feature_threads=args.feature_threads,
    )
    signal.signal(signal.SIGTERM, lambda *_: setattr(controller, "stop_requested", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(controller, "stop_requested", True))
    try:
        return controller.run()
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
