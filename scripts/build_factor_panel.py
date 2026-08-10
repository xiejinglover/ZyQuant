"""Build the reusable daily factor panel for a published snapshot.

Computes every `zyquant.factors.cn_equity` factor once over the snapshot's full
range and leaves the results in the `FactorEngine` cache, which is the artifact.
Strategies consume it through `FactorEngine.load_view`, so they only materialize
their observation dates and never recompute a prewarmed factor.

Two parameters are fixed deliberately and must not be varied per caller, because
both sit inside the cache identity and changing either forfeits every hit:

* `instruments=None` — the whole universe;
* `cutoff` — the snapshot's as-of date.

Using the snapshot's as-of date as the cutoff is not look-ahead. The cutoff only
bounds which rows a factor may read at all; each emitted row is still computed
from its own trailing window alone. `tests/test_cn_equity_factors.py` pins that
property by recomputing a narrower window and comparing the shared rows.

A wide panel and a manifest are also written, because the engine's own cache
metadata records neither the factor name nor the universe, and there is no CLI
to inspect it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from zyquant.data import ParquetDataProvider
from zyquant.factors import FactorEngine, cn_equity_factors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--only", action="append", default=[],
        help="factor name; repeatable, defaults to all",
    )
    args = parser.parse_args(argv)

    snapshot = ParquetDataProvider(args.root).open_snapshot(
        args.dataset, False
    )
    calendar = sorted(set(snapshot.table("trade_calendar")["trade_date"]))
    end = args.end or calendar[-1]
    start = args.start or calendar[0]
    # The cutoff must be identical for every factor and every later consumer,
    # otherwise the cache fragments; the snapshot's own end is the natural pick.
    cutoff = calendar[-1]
    if end > cutoff:
        raise SystemExit(f"end {end} exceeds snapshot cutoff {cutoff}")

    engine = FactorEngine(args.cache_root)
    factors = [
        factor for factor in cn_equity_factors(workers=args.workers)
        if not args.only or factor.name in args.only
    ]
    if not factors:
        raise SystemExit(f"no factor matched {args.only}")

    report: dict[str, object] = {
        "dataset": args.dataset,
        "dataset_fingerprint": snapshot.metadata.fingerprint,
        "as_of_date": str(snapshot.metadata.as_of_date),
        "range": [str(start), str(end)],
        "cutoff": str(cutoff),
        "instruments": None,
        "workers": args.workers,
        "factors": {},
    }
    panels: dict[str, pd.DataFrame] = {}
    for factor in factors:
        began = time.perf_counter()
        result = engine.compute(factor, snapshot, start, end, None, cutoff)
        elapsed = time.perf_counter() - began
        frame = result.frame
        report["factors"][factor.name] = {
            "version": factor.version,
            "definition": dict(factor.definition()),
            "cache_key": result.cache_key,
            "from_cache": bool(result.from_cache),
            "seconds": round(elapsed, 1),
            "diagnostics": dict(result.diagnostics),
        }
        print(
            f"{factor.name:26s} rows={len(frame):>10,} "
            f"{'cached' if result.from_cache else 'computed':>8s} "
            f"{elapsed:7.1f}s",
            flush=True,
        )
        if args.output is not None:
            panels[factor.name] = frame.set_index(
                ["trade_date", "instrument_id"]
            )["value"].rename(factor.name)

    if args.output is not None and panels:
        wide = pd.concat(panels.values(), axis=1).reset_index()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(args.output, index=False)
        report["panel"] = {
            "path": str(args.output),
            "rows": int(len(wide)),
            "columns": list(wide.columns),
            "coverage": {
                name: round(float(wide[name].notna().mean()), 4)
                for name in panels
            },
        }
        manifest = args.output.with_name(
            args.output.stem + "_manifest.json"
        )
        manifest.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\npanel   {args.output}  rows={len(wide):,}")
        print(f"manifest {manifest}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
