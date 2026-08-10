"""Build detailed attribution from an already completed immutable backtest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from zyquant.analysis import attribution_report
from zyquant.config import ResolvedRunConfig
from zyquant.data import ParquetDataProvider


def _read_frame(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"derived attribution already exists: {output_dir}")
    resolved_path = run_dir / "resolved_config.json"
    manifest_path = run_dir / "manifest.json"
    if not resolved_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"not a completed ZyQuant run directory: {run_dir}")

    config = ResolvedRunConfig.model_validate(
        json.loads(resolved_path.read_text(encoding="utf-8"))
    )
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = ParquetDataProvider(config.data.root).open_snapshot(
        config.data.dataset_id, config.data.verify_hashes
    )
    attribution = attribution_report(
        _read_frame(run_dir, "nav"),
        _read_frame(run_dir, "fill_allocations"),
        _read_frame(run_dir, "positions"),
        _read_frame(run_dir, "corporate_actions"),
        snapshot.table(
            "industry_membership", cutoff=config.data.end_date
        ),
        tolerance=config.analysis.attribution_tolerance,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ))
    try:
        attribution.to_parquet(staging / "attribution.parquet", index=False)
        derived_manifest = {
            "artifact_type": "posthoc_attribution",
            "source_run_id": source_manifest["run_id"],
            "source_run_fingerprint": source_manifest["run_fingerprint"],
            "data_fingerprint": source_manifest["data_fingerprint"],
            "rows": len(attribution),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(derived_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
