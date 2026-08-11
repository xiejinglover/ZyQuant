from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

from zyquant.data import ParquetDataProvider
from zyquant.factors import FactorEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.ml_ema20_momentum_v1.dataset import (  # noqa: E402
    LABEL_COLUMN,
    LABEL_DEFINITION,
    MODEL_FEATURES,
    add_cross_sectional_features,
    attach_executable_labels,
    attach_factor,
    candidate_keys,
    frame_fingerprint,
    numeric_quality,
    rolling_year_folds,
)
from strategies.ml_ema20_momentum_v1.factors import (  # noqa: E402
    FEATURE_NAMES,
    momentum_factor_catalog,
)
from strategies.ml_ema20_momentum_v1.universe import (  # noqa: E402
    build_ema20_universe,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _distribution(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    quantiles = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        row: dict[str, Any] = {"column": column, "finite_rows": int(len(finite))}
        row.update({f"q{value:g}": float(finite.quantile(value)) for value in quantiles})
        row["mean"] = float(finite.mean()) if len(finite) else None
        row["std_ddof0"] = float(finite.std(ddof=0)) if len(finite) else None
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build EMA20 event datasets with rolling 3y train / 1y test folds."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--factor-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args(argv)

    snapshot = ParquetDataProvider(args.root).open_snapshot(args.dataset, False)
    calendar = sorted(set(snapshot.table("trade_calendar")["trade_date"]))
    start = args.start or calendar[0]
    end = args.end or calendar[-1]
    cutoff = calendar[-1]
    if not (calendar[0] <= start <= end <= cutoff):
        raise SystemExit("dataset range falls outside the snapshot calendar")
    factor_manifest = json.loads(args.factor_manifest.read_text(encoding="utf-8"))
    if factor_manifest.get("dataset_fingerprint") != snapshot.metadata.fingerprint:
        raise SystemExit("factor manifest belongs to another snapshot")
    if factor_manifest.get("range") != [str(start), str(end)]:
        raise SystemExit("factor manifest range does not match dataset range")

    began = time.perf_counter()
    universe = build_ema20_universe(snapshot, start, end)
    panel = candidate_keys(universe)
    factor_engine = FactorEngine(args.cache_root, cache_policy="require")
    cache_keys: dict[str, str] = {}
    catalog = momentum_factor_catalog()
    for feature_name in FEATURE_NAMES:
        factor = catalog[f"ml_ema20_{feature_name}"]
        result = factor_engine.compute(factor, snapshot, start, end, None, cutoff)
        expected = factor_manifest["factors"][factor.name]["cache_key"]
        if result.cache_key != expected or not result.from_cache:
            raise SystemExit(f"factor cache mismatch: {factor.name}")
        panel = attach_factor(panel, result.frame, feature_name)
        cache_keys[feature_name] = result.cache_key
        print(f"attached {feature_name:36s} rows={len(panel):>9,}", flush=True)
    panel = add_cross_sectional_features(panel)

    instruments = sorted(set(panel["instrument_id"].astype(str)))
    price_start = min(panel["signal_date"])
    prices = snapshot.post_adjusted_bars(
        price_start, cutoff, instruments, ["open_post", "close_post"], cutoff=cutoff,
    )
    panel = attach_executable_labels(panel, calendar, prices)
    panel.sort_values(
        ["signal_date", "instrument_id"], kind="mergesort", inplace=True,
        ignore_index=True,
    )
    folds = rolling_year_folds(
        panel, first_year=start.year, last_year=end.year, training_years=3,
    )
    if not folds:
        raise SystemExit("rolling 3y/1y construction produced no folds")

    quality = numeric_quality(panel, (*MODEL_FEATURES, LABEL_COLUMN))
    feature_inf = sum(
        item["positive_inf"] + item["negative_inf"]
        for name, item in quality.items() if name in MODEL_FEATURES
    )
    label_inf = quality[LABEL_COLUMN]["positive_inf"] + quality[LABEL_COLUMN][
        "negative_inf"
    ]
    fold_manifest = []
    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise SystemExit(f"immutable dataset directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset.", dir=destination.parent))
    try:
        panel.to_parquet(staging / "labeled_panel.parquet", index=False)
        universe.diagnostics.to_parquet(
            staging / "universe_diagnostics.parquet", index=False
        )
        _distribution(panel, (*MODEL_FEATURES, LABEL_COLUMN)).to_parquet(
            staging / "distribution.parquet", index=False
        )
        for fold in folds:
            fold_path = staging / "folds" / fold.fold_id
            fold_path.mkdir(parents=True)
            fold.train.to_parquet(fold_path / "train.parquet", index=False)
            fold.test.to_parquet(fold_path / "test.parquet", index=False)
            train_quality = numeric_quality(fold.train, (*MODEL_FEATURES, LABEL_COLUMN))
            test_quality = numeric_quality(fold.test, (*MODEL_FEATURES, LABEL_COLUMN))
            _write_json(fold_path / "quality.json", {
                "train": train_quality,
                "test": test_quality,
            })
            fold_manifest.append({
                "fold_id": fold.fold_id,
                "train_years": fold.train_years,
                "test_year": fold.test_year,
                "train_rows": len(fold.train),
                "test_rows": len(fold.test),
                "train_start": fold.train["signal_date"].min(),
                "train_end": fold.train["signal_date"].max(),
                "train_label_end": fold.train["label_end_date"].max(),
                "test_start": fold.test["signal_date"].min(),
                "test_end": fold.test["signal_date"].max(),
            })
        valid_panel = panel[panel["label_valid"]]
        label_values = valid_panel[LABEL_COLUMN]
        label_by_year = []
        for year, group in valid_panel.groupby(
            valid_panel["signal_date"].map(lambda value: value.year)
        ):
            values = group[LABEL_COLUMN]
            label_by_year.append({
                "year": int(year), "rows": len(values),
                "mean": float(values.mean()), "std_ddof0": float(values.std(ddof=0)),
                "positive_rate": float((values > 0).mean()),
                "q01": float(values.quantile(0.01)),
                "q50": float(values.quantile(0.5)),
                "q99": float(values.quantile(0.99)),
            })
        manifest = {
            "strategy": "ml_ema20_momentum_v1",
            "dataset_version": "rolling_3y_1y_v1",
            "snapshot_id": snapshot.metadata.dataset_id,
            "snapshot_fingerprint": snapshot.metadata.fingerprint,
            "factor_manifest": str(args.factor_manifest.resolve()),
            "factor_cache_keys": cache_keys,
            "range": [start, end],
            "cutoff": cutoff,
            "universe_fingerprint": universe.fingerprint,
            "features": MODEL_FEATURES,
            "cached_feature_count": len(FEATURE_NAMES),
            "cross_sectional_feature_count": len(MODEL_FEATURES) - len(FEATURE_NAMES),
            "label_column": LABEL_COLUMN,
            "label_definition": LABEL_DEFINITION,
            "candidate_rows": len(panel),
            "valid_label_rows": int(panel["label_valid"].sum()),
            "panel_fingerprint": frame_fingerprint(
                panel, ["signal_date", "instrument_id", *MODEL_FEATURES, LABEL_COLUMN]
            ),
            "folds": fold_manifest,
            "seconds": round(time.perf_counter() - began, 3),
        }
        quality_report = {
            "training_ready": feature_inf == 0 and label_inf == 0,
            "feature_inf_count": feature_inf,
            "label_inf_count": label_inf,
            "label_status": panel["label_status"].value_counts(dropna=False).to_dict(),
            "complete_feature_rows": int(panel[list(MODEL_FEATURES)].notna().all(axis=1).sum()),
            "any_feature_nan_rows": int(panel[list(MODEL_FEATURES)].isna().any(axis=1).sum()),
            "columns": quality,
            "label_distribution": {
                "rows": int(len(label_values)),
                "mean": float(label_values.mean()),
                "std_ddof0": float(label_values.std(ddof=0)),
                "positive_rate": float((label_values > 0).mean()),
                "minimum": float(label_values.min()),
                "maximum": float(label_values.max()),
                "quantiles": {
                    str(q): float(label_values.quantile(q))
                    for q in (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)
                },
            },
            "label_by_year": label_by_year,
        }
        _write_json(staging / "dataset_manifest.json", manifest)
        _write_json(staging / "quality_report.json", quality_report)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"published rolling dataset: {destination}")
    print(
        f"candidates={len(panel):,} valid_labels={int(panel['label_valid'].sum()):,} "
        f"feature_inf={feature_inf} label_inf={label_inf} folds={len(folds)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
