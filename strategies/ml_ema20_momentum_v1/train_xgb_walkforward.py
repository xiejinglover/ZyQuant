from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.ml_ema20_momentum_v1.dataset import (  # noqa: E402
    LABEL_COLUMN, MODEL_FEATURES,
)
from strategies.ml_ema20_momentum_v1.xgb_training import (  # noqa: E402
    BINARY_FEATURES, CONTINUOUS_FEATURES, FEATURE_SET_ID, MODEL_VERSION,
    SEED, XGB_PARAMETERS, determinism_check, fit_early_stopping,
    model_sha256, prediction_fingerprint, prediction_metrics, predict,
    prepare_fold, preprocess_daily, refit_full,
)


def _json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# XGBoost首轮滚动训练研究结果", "",
        f"- 设备：`{manifest['device']}`",
        f"- 年度模型：{len(manifest['models'])}",
        f"- 预测行数：{manifest['prediction_rows']:,}",
        f"- 汇总 NDCG@3：{manifest['overall_metrics']['ndcg_at_3']:.6f}",
        f"- Top3平均标签收益：{manifest['overall_metrics']['top3_mean_return']:.6%}",
        "", "| 测试年 | 训练行 | 测试行 | 最佳轮数 | 验证NDCG@3 | 测试NDCG@3 | Top3收益 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in manifest["models"]:
        metrics = item["test_metrics"]
        lines.append(
            f"| {item['test_year']} | {item['train_rows']:,} | {item['test_rows']:,} "
            f"| {item['best_iterations']} | {item['validation_ndcg_at_3']:.6f} "
            f"| {metrics['ndcg_at_3']:.6f} | {metrics['top3_mean_return']:.6%} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train annual XGBoost EMA20 rankers.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--first-test-year", type=int, default=2015)
    parser.add_argument("--last-test-year", type=int, default=2026)
    args = parser.parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    dataset_manifest = json.loads(
        (dataset_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if dataset_manifest.get("dataset_version") != "rolling_3y_1y_v2_clean":
        raise SystemExit("XGBoost training requires rolling_3y_1y_v2_clean")
    selected = [
        item for item in dataset_manifest["folds"]
        if args.first_test_year <= int(item["test_year"]) <= args.last_test_year
    ]
    if len(selected) != args.last_test_year - args.first_test_year + 1:
        raise SystemExit("dataset does not contain every requested annual fold")
    if args.prediction_output.exists():
        raise SystemExit(
            f"immutable prediction output already exists: {args.prediction_output}"
        )
    try:
        import xgboost
    except ImportError as exc:
        raise SystemExit("install the 'ml-ranking' optional dependency") from exc

    loaded: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for item in selected:
        fold_path = dataset_root / "folds" / item["fold_id"]
        train = preprocess_daily(pd.read_parquet(fold_path / "train.parquet"))
        test = preprocess_daily(pd.read_parquet(fold_path / "test.parquet"))
        loaded[int(item["test_year"])] = (train, test)

    first_year = int(selected[0]["test_year"])
    first_prepared = prepare_fold(*loaded[first_year])
    requested_device = "cuda" if args.device == "auto" else args.device
    stability = determinism_check(first_prepared, requested_device)
    device = requested_device
    if not stability["stable"]:
        if args.device == "cuda":
            raise SystemExit(f"CUDA determinism check failed: {stability}")
        device = "cpu"
        stability = {
            "cuda": stability,
            "cpu": determinism_check(first_prepared, "cpu"),
        }
        if not stability["cpu"]["stable"]:
            raise SystemExit(f"CPU determinism check failed: {stability}")

    destination = args.model_output.resolve()
    if destination.exists():
        raise SystemExit(f"immutable model output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".xgb-models.", dir=destination.parent))
    all_predictions = []
    model_entries = []
    began = time.perf_counter()
    try:
        for item in selected:
            test_year = int(item["test_year"])
            train, test = loaded[test_year]
            fold = prepare_fold(train, test)
            _, best_iterations, validation_metric = fit_early_stopping(fold, device)
            model = refit_full(fold, device, best_iterations)
            model_id = f"xgb-ranker-3y1y-{test_year}"
            model_dir = staging / model_id
            model_dir.mkdir()
            model_path = model_dir / "model.json"
            model.save_model(model_path)
            ordered_test = fold.test.sort_values(
                ["signal_date", "instrument_id"], kind="mergesort", ignore_index=True
            ).copy()
            ordered_test["score"] = predict(model, ordered_test)
            from strategies.ml_ema20_momentum_v1.xgb_training import make_relevance
            ordered_test["relevance"] = make_relevance(ordered_test)
            metrics = prediction_metrics(ordered_test)
            train_cutoff = max(fold.train["label_end_date"])
            predictions = ordered_test[["signal_date", "instrument_id", "score"]].copy()
            predictions["model_id"] = model_id
            predictions["model_version"] = MODEL_VERSION
            predictions["feature_cutoff"] = predictions["signal_date"]
            predictions["train_cutoff"] = train_cutoff
            predictions["dataset_id"] = dataset_manifest["snapshot_id"]
            predictions["data_fingerprint"] = dataset_manifest["snapshot_fingerprint"]
            predictions["feature_set_id"] = FEATURE_SET_ID
            all_predictions.append(predictions)
            gain = model.get_booster().get_score(importance_type="gain")
            metadata = {
                "model_id": model_id,
                "model_version": MODEL_VERSION,
                "test_year": test_year,
                "train_years": item["train_years"],
                "train_rows": len(fold.train),
                "fit_rows": len(fold.fit),
                "validation_rows": len(fold.validation),
                "test_rows": len(fold.test),
                "validation_start": fold.validation_start,
                "embargo_dates": fold.embargo_dates,
                "train_cutoff": train_cutoff,
                "best_iterations": best_iterations,
                "validation_ndcg_at_3": validation_metric,
                "test_metrics": metrics,
                "features": MODEL_FEATURES,
                "binary_features": BINARY_FEATURES,
                "continuous_features": CONTINUOUS_FEATURES,
                "feature_set_id": FEATURE_SET_ID,
                "preprocessing": "daily_1pct_99pct_winsor_then_percentile_rank",
                "label": "daily_0_to_4_relevance_from_T1_open_T2_close_return",
                "parameters": {**XGB_PARAMETERS, "device": device},
                "seed": SEED,
                "device": device,
                "xgboost_version": xgboost.__version__,
                "dataset_panel_fingerprint": dataset_manifest["panel_fingerprint"],
                "feature_importance_gain": gain,
                "model_sha256": model_sha256(model_path),
            }
            _json(model_dir / "metadata.json", metadata)
            model_entries.append(metadata)
            print(
                f"trained {test_year}: train={len(fold.train):,} test={len(fold.test):,} "
                f"iterations={best_iterations} ndcg@3={metrics['ndcg_at_3']:.6f}",
                flush=True,
            )

        prediction_frame = pd.concat(all_predictions, ignore_index=True)
        prediction_frame.sort_values(
            ["signal_date", "score", "instrument_id"],
            ascending=[True, False, True], kind="mergesort", inplace=True,
            ignore_index=True,
        )
        evaluation = prediction_frame[["signal_date", "instrument_id", "score"]].merge(
            pd.concat([value[1] for value in loaded.values()], ignore_index=True)[
                ["signal_date", "instrument_id", LABEL_COLUMN]
            ], on=["signal_date", "instrument_id"], how="inner", validate="one_to_one",
        )
        from strategies.ml_ema20_momentum_v1.xgb_training import make_relevance
        evaluation["relevance"] = make_relevance(evaluation)
        overall = prediction_metrics(evaluation)
        importance_rows = []
        for feature in MODEL_FEATURES:
            values = [
                float(item["feature_importance_gain"].get(feature, 0.0))
                for item in model_entries
            ]
            importance_rows.append({
                "feature": feature,
                "mean_gain": float(np.mean(values)),
                "std_gain": float(np.std(values, ddof=0)),
                "nonzero_years": int(np.count_nonzero(values)),
            })
        importance_stability = pd.DataFrame(importance_rows).sort_values(
            ["mean_gain", "feature"], ascending=[False, True], kind="mergesort"
        )
        importance_stability.to_parquet(
            staging / "feature_importance_stability.parquet", index=False
        )
        manifest = {
            "strategy": "ml_ema20_momentum_v1",
            "research_name": "XGBoost首轮研究结果",
            "model_family": "XGBRanker",
            "objective": "rank:ndcg",
            "metric": "ndcg@3",
            "device": device,
            "xgboost_version": xgboost.__version__,
            "determinism_check": stability,
            "dataset_root": str(dataset_root),
            "dataset_panel_fingerprint": dataset_manifest["panel_fingerprint"],
            "snapshot_fingerprint": dataset_manifest["snapshot_fingerprint"],
            "feature_set_id": FEATURE_SET_ID,
            "prediction_rows": len(prediction_frame),
            "prediction_fingerprint": prediction_fingerprint(prediction_frame),
            "overall_metrics": overall,
            "feature_importance_stability": importance_stability.to_dict("records"),
            "models": model_entries,
            "seconds": round(time.perf_counter() - began, 3),
        }
        _json(staging / "training_manifest.json", manifest)
        (staging / "RESEARCH_REPORT.md").write_text(
            _markdown(manifest), encoding="utf-8"
        )
        os.replace(staging, destination)
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_prediction = args.prediction_output.with_suffix(".tmp.parquet")
        prediction_frame.to_parquet(temporary_prediction, index=False)
        os.replace(temporary_prediction, args.prediction_output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"published models: {destination}")
    print(f"published predictions: {args.prediction_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
