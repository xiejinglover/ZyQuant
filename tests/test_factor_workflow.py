from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from zyquant.config import (
    AccountConfig, DataConfig, FactorConfig, ResolvedRunConfig,
)
from zyquant.core.exceptions import FactorCacheMiss
from zyquant.data import SnapshotPublisher
from zyquant.factors import BaseFactor
from zyquant.factors.workflow import (
    collect_factor_requirements, run_factor_cache_workflow,
)

from tests.support import canonical_tables


class _DeclaredFactor(BaseFactor):
    name = "declared_workflow_factor"

    def compute(self, context, dependencies):
        days = sorted({
            day for day in context.snapshot.table("trade_calendar")["trade_date"]
            if context.start <= day <= context.end
        })
        return pd.DataFrame({
            "trade_date": days,
            "instrument_id": "600000.XSHG",
            "value": 1.0,
        })


class _DeclaredStrategy:
    strategy_id = "declared"

    @staticmethod
    def factor_requirements():
        return {"signal": _DeclaredFactor()}


class _LegacyStrategy:
    strategy_id = "legacy"


def _config(project: Path, days) -> ResolvedRunConfig:
    return ResolvedRunConfig(
        data=DataConfig(
            root=Path("data"), dataset_id="factor-workflow-v1",
            start_date=days[0], end_date=days[-2], cutoff=days[-1],
        ),
        factor=FactorConfig(
            cache_root=Path(".zyquant/cache/factors"),
            cache_policy="require",
        ),
        account=AccountConfig(initial_cash=1_000_000),
        output_root=Path("runs"),
    )


def test_prepare_uses_project_relative_cache_and_verify_is_read_only(tmp_path: Path):
    tables, days = canonical_tables()
    SnapshotPublisher(tmp_path / "data").publish("factor-workflow-v1", tables)
    config = _config(tmp_path, days)

    prepared = run_factor_cache_workflow(
        config, [_DeclaredStrategy()], tmp_path, mode="prepare"
    )
    assert prepared["status"] == "prepared"
    assert prepared["cutoff"] == days[-1]
    assert prepared["cache_root"] == str(
        (tmp_path / ".zyquant/cache/factors").resolve()
    )
    assert prepared["factors"]["signal"]["from_cache"] is False
    assert Path(prepared["manifest_path"]).exists()

    tracked = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*") if path.is_file()
    }
    verified = run_factor_cache_workflow(
        config, [_DeclaredStrategy()], tmp_path, mode="verify"
    )
    assert verified["status"] == "verified"
    assert verified["factors"]["signal"]["from_cache"] is True
    assert tracked == {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*") if path.is_file()
    }


def test_verify_miss_does_not_create_cache_or_output(tmp_path: Path):
    tables, days = canonical_tables()
    SnapshotPublisher(tmp_path / "data").publish("factor-workflow-v1", tables)
    config = _config(tmp_path, days)

    with pytest.raises(FactorCacheMiss, match="required factor cache is missing"):
        run_factor_cache_workflow(
            config, [_DeclaredStrategy()], tmp_path, mode="verify"
        )
    assert not (tmp_path / ".zyquant").exists()
    assert not (tmp_path / "runs").exists()


def test_legacy_strategy_gets_an_actionable_compatibility_error():
    with pytest.raises(ValueError, match="legacy prewarm scripts.*legacy"):
        collect_factor_requirements([_LegacyStrategy()])
