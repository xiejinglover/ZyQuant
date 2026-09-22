from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from zyquant.config import ResolvedRunConfig, resolve_project_path
from zyquant.core.hashing import hash_payload
from zyquant.data import ParquetDataProvider

from .base import BaseFactor
from .engine import FactorEngine


def collect_factor_requirements(
    strategies: Iterable[Any],
) -> dict[str, BaseFactor]:
    """Collect the pure declarations supplied by new strategies.

    Legacy strategies deliberately remain supported by the backtest path, but
    the unified factor CLI refuses to guess their needs from ``prepare_run``.
    """
    requirements: dict[str, BaseFactor] = {}
    unsupported: list[str] = []
    for strategy in strategies:
        declare = getattr(strategy, "factor_requirements", None)
        if not callable(declare):
            unsupported.append(str(getattr(strategy, "strategy_id", type(strategy).__name__)))
            continue
        declared = declare()
        if not isinstance(declared, Mapping):
            raise TypeError("factor_requirements() must return a mapping")
        for alias, factor in declared.items():
            key = str(alias)
            if not key:
                raise ValueError("factor requirement aliases must be non-empty")
            if not isinstance(factor, BaseFactor):
                raise TypeError(f"factor requirement {key!r} is not a BaseFactor")
            if key in requirements and requirements[key].definition() != factor.definition():
                raise ValueError(f"conflicting factor requirement alias: {key}")
            requirements[key] = factor
    if unsupported:
        raise ValueError(
            "strategies do not declare factor_requirements; use their legacy "
            f"prewarm scripts: {sorted(unsupported)}"
        )
    return dict(sorted(requirements.items()))


def run_factor_cache_workflow(
    config: ResolvedRunConfig,
    strategies: Iterable[Any],
    project_root: str | Path,
    *,
    mode: str,
) -> dict[str, Any]:
    """Prepare or read-only verify canonical full-universe factor caches."""
    if mode not in {"prepare", "verify"}:
        raise ValueError("factor cache mode must be 'prepare' or 'verify'")
    requirements = collect_factor_requirements(strategies)
    if not requirements:
        return {"status": "no_factors", "mode": mode, "factors": {}}
    if config.data.cutoff is None:
        raise ValueError(
            "factor workflows require data.cutoff; formal ZyQuant configs use "
            "2026-07-24"
        )

    root = Path(project_root).expanduser().resolve()
    data_root = resolve_project_path(config.data.root, root)
    cache_root = resolve_project_path(config.factor.cache_root, root)
    output_root = resolve_project_path(config.output_root, root)
    snapshot = ParquetDataProvider(data_root).open_snapshot(
        config.data.dataset_id, config.data.verify_hashes
    )
    calendar = sorted(set(snapshot.table("trade_calendar")["trade_date"]))
    cutoff = config.data.cutoff
    eligible = [day for day in calendar if day <= cutoff]
    if not eligible:
        raise ValueError(f"snapshot has no trading sessions on or before cutoff {cutoff}")
    if cutoff > snapshot.metadata.as_of_date:
        raise ValueError(
            f"factor cutoff {cutoff} exceeds snapshot as-of date "
            f"{snapshot.metadata.as_of_date}"
        )
    start, end = calendar[0], eligible[-1]
    engine = FactorEngine(
        cache_root,
        config.factor.lock_timeout_seconds,
        "compute" if mode == "prepare" else "require",
    )

    entries: dict[str, Any] = {}
    for alias, factor in requirements.items():
        result = engine.compute(
            factor, snapshot, start, end, instruments=None, cutoff=cutoff
        )
        metadata_path = (
            cache_root / snapshot.metadata.fingerprint / factor.name
            / f"{result.cache_key}.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        entries[alias] = {
            "factor_name": factor.name,
            "factor_version": factor.version,
            "definition": dict(factor.definition()),
            "cache_key": result.cache_key,
            "from_cache": bool(result.from_cache),
            "parquet_sha256": metadata["parquet_sha256"],
            "diagnostics": dict(result.diagnostics),
        }

    identity = {
        "data_fingerprint": snapshot.metadata.fingerprint,
        "cutoff": cutoff,
        "instruments": None,
        "requirements": {
            alias: {
                "definition": item["definition"],
                "cache_key": item["cache_key"],
            }
            for alias, item in entries.items()
        },
    }
    manifest = {
        "schema_version": "1.0",
        "mode": mode,
        "config_fingerprint": config.fingerprint,
        "dataset_id": snapshot.metadata.dataset_id,
        "data_fingerprint": snapshot.metadata.fingerprint,
        "cache_root": str(cache_root),
        "range": [start, end],
        "cutoff": cutoff,
        "instruments": None,
        "factors": entries,
    }
    manifest_id = hash_payload(identity)[:20]
    manifest["manifest_id"] = manifest_id

    if mode == "prepare":
        parent = output_root / "factors"
        destination = parent / manifest_id
        parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            if (
                existing.get("manifest_id") != manifest_id
                or {
                    alias: item.get("cache_key")
                    for alias, item in existing.get("factors", {}).items()
                } != {
                    alias: item["cache_key"] for alias, item in entries.items()
                }
            ):
                raise ValueError(
                    f"immutable factor manifest has conflicting content: {destination}"
                )
        else:
            staging = Path(tempfile.mkdtemp(prefix=f".{manifest_id}.", dir=parent))
            try:
                (staging / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                os.replace(staging, destination)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        manifest["manifest_path"] = str(destination / "manifest.json")
    manifest["status"] = "prepared" if mode == "prepare" else "verified"
    return manifest
