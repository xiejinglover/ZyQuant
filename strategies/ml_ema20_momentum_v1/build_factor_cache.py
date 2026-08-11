from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from zyquant.data import ParquetDataProvider
from zyquant.factors import FactorEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.ml_ema20_momentum_v1.factors import (  # noqa: E402
    FEATURE_SPECS,
    momentum_factor_catalog,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prewarm EMA20 momentum factor caches.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--cache-policy", choices=("compute", "require"), default="compute",
    )
    args = parser.parse_args(argv)

    snapshot = ParquetDataProvider(args.root).open_snapshot(args.dataset, False)
    calendar = sorted(set(snapshot.table("trade_calendar")["trade_date"]))
    start = args.start or calendar[0]
    end = args.end or calendar[-1]
    cutoff = calendar[-1]
    if start < calendar[0] or end > cutoff or start > end:
        raise SystemExit("factor range falls outside the snapshot calendar")
    catalog = momentum_factor_catalog()
    selected = [
        factor for name, factor in catalog.items()
        if not args.only or name in args.only or factor.feature_name in args.only
    ]
    if not selected:
        raise SystemExit(f"no factor matched --only={args.only}")

    engine = FactorEngine(args.cache_root, cache_policy=args.cache_policy)
    entries: dict[str, Any] = {}
    log_lines = []
    for factor in selected:
        began = time.perf_counter()
        result = engine.compute(factor, snapshot, start, end, None, cutoff)
        elapsed = time.perf_counter() - began
        values = pd.to_numeric(result.frame["value"], errors="coerce")
        finite = values[values.notna()]
        spec = FEATURE_SPECS[factor.feature_name]
        entries[factor.name] = {
            "feature_name": factor.feature_name,
            "version": factor.version,
            "definition": dict(factor.definition()),
            "formula": spec["formula"],
            "bar_policy": spec["bar_policy"],
            "cache_key": result.cache_key,
            "from_cache": bool(result.from_cache),
            "rows": int(len(result.frame)),
            "valid_rows": int(values.notna().sum()),
            "missing_rate": float(values.isna().mean()) if len(values) else 1.0,
            "first_valid_date": (
                result.frame.loc[values.notna(), "trade_date"].min()
                if values.notna().any() else None
            ),
            "minimum": float(finite.min()) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None,
            "seconds": round(elapsed, 3),
        }
        line = (
            f"{factor.name:46s} rows={len(result.frame):>10,} "
            f"valid={int(values.notna().sum()):>10,} "
            f"{'cached' if result.from_cache else 'computed':>8s} {elapsed:8.1f}s"
        )
        print(line, flush=True)
        log_lines.append(line)

    manifest = {
        "strategy": "ml_ema20_momentum_v1",
        "dataset_id": snapshot.metadata.dataset_id,
        "dataset_fingerprint": snapshot.metadata.fingerprint,
        "snapshot_as_of_date": snapshot.metadata.as_of_date,
        "range": [start, end],
        "cutoff": cutoff,
        "instruments": None,
        "output_universe": "XSHG_XSHE_A_shares",
        "factor_count": len(entries),
        "cache_policy": args.cache_policy,
        "factors": entries,
    }
    quality = {
        "factor_count": len(entries),
        "all_finite_or_missing": all(
            item["minimum"] is None
            or (item["minimum"] != float("-inf") and item["maximum"] != float("inf"))
            for item in entries.values()
        ),
        "coverage": {
            name: 1.0 - item["missing_rate"] for name, item in entries.items()
        },
    }

    destination = args.output.expanduser().resolve()
    if args.cache_policy == "require" and destination.exists():
        existing = json.loads(
            (destination / "cache_manifest.json").read_text(encoding="utf-8")
        )
        if existing.get("dataset_fingerprint") != snapshot.metadata.fingerprint:
            raise SystemExit("existing factor manifest belongs to another snapshot")
        existing_factors = dict(existing.get("factors", {}))
        mismatched = sorted(
            name for name, item in entries.items()
            if name not in existing_factors
            or existing_factors[name].get("cache_key") != item["cache_key"]
        )
        if mismatched:
            raise SystemExit(
                f"required caches do not match the manifest: {mismatched[:10]}"
            )
        print(f"verified {len(entries)} required caches against {destination}")
        return 0
    if destination.exists():
        raise SystemExit(f"immutable factor audit directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".factors.", dir=destination.parent))
    try:
        _write_json(staging / "cache_manifest.json", manifest)
        _write_json(staging / "quality_report.json", quality)
        (staging / "build.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"published factor audit: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
