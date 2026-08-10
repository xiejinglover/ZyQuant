"""因子缓存的单版本策略：代码改动淘汰旧缓存，合法变体共存。

约定（factors/engine.py 模块 docstring）：因子只有「当前」一个版本。
写入新缓存时，同一因子目录下「定义、cutoff、universe 相同但 identity 不同」
的条目会被自动删除——identity 差异此时只可能来自因子源码或依赖，即旧代码
的产物。参数化变体（definition 不同）与不同 cutoff 的缓存必须共存，
否则参数搜索会互相误删。

「改源码」用 monkeypatch `inspect.getsource` 模拟：真实场景是同一个类
（definition 中的限定类名不变）类体文本变化，不能用两个不同类名的类模拟
——那会连 definition 一起变，落入「参数变体共存」的分支。
"""
from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import zyquant.factors.engine as engine_module
from zyquant.core.exceptions import FactorCacheMiss
from zyquant.data import SnapshotPublisher
from zyquant.factors import FactorEngine
from zyquant.factors.base import BaseFactor

from tests.test_cn_equity_factors import _tables, _trading_days


def _calendar_rows(context, value: float) -> pd.DataFrame:
    days = sorted({
        day for day in context.snapshot.table("trade_calendar")["trade_date"]
        if context.start <= day <= context.end
    })
    return pd.DataFrame({
        "trade_date": days,
        "instrument_id": "600000.XSHG",
        "value": value,
    })


class _Probe(BaseFactor):
    name = "policy_probe"
    version = "1"

    def compute(self, context, dependencies):
        return _calendar_rows(context, 1.0)


class _Parameterised(BaseFactor):
    name = "policy_param"
    version = "1"

    def __init__(self, k: int):
        self.k = k

    def definition(self):
        return {**super().definition(), "k": self.k}

    def compute(self, context, dependencies):
        return _calendar_rows(context, float(self.k))


@pytest.fixture(scope="module")
def world():
    days = _trading_days(date(2024, 1, 2), 40)
    with tempfile.TemporaryDirectory() as directory:
        snapshot = SnapshotPublisher(directory).publish(
            "cache-policy-v1", _tables(days), schema_version="1.1",
            lineage={"capabilities": {"financials": {
                "schema_version": "1.1", "pit_validated": True,
            }}},
        )
        yield snapshot, days


def _entries(cache: str, snapshot, name: str) -> list[Path]:
    directory = Path(cache) / snapshot.metadata.fingerprint / name
    return sorted(directory.glob("*.json"))


def _pretend_source(monkeypatch, text: str) -> None:
    monkeypatch.setattr(
        engine_module.inspect, "getsource", lambda cls: text
    )


def test_a_source_change_replaces_the_old_cache_entry(world, monkeypatch):
    snapshot, days = world
    start, end = days[5], days[-1]
    with tempfile.TemporaryDirectory() as cache:
        engine = FactorEngine(cache)
        _pretend_source(monkeypatch, "class _Probe: pass  # v1")
        engine.compute(_Probe(), snapshot, start, end, None, end)
        assert len(_entries(cache, snapshot, "policy_probe")) == 1

        _pretend_source(monkeypatch, "class _Probe: pass  # v2")
        fresh = engine.compute(_Probe(), snapshot, start, end, None, end)
        assert not fresh.from_cache, "a source change must recompute"
        entries = _entries(cache, snapshot, "policy_probe")
        assert len(entries) == 1, "the pre-change entry must be gone"
        surviving = json.loads(entries[0].read_text(encoding="utf-8"))
        assert surviving["cache_key"] == fresh.cache_key
        # 新版之后的读取是正常命中。
        assert engine.compute(
            _Probe(), snapshot, start, end, None, end
        ).from_cache


def test_parameter_variants_are_not_purged(world):
    snapshot, days = world
    start, end = days[5], days[-1]
    with tempfile.TemporaryDirectory() as cache:
        engine = FactorEngine(cache)
        engine.compute(_Parameterised(1), snapshot, start, end, None, end)
        engine.compute(_Parameterised(2), snapshot, start, end, None, end)
        assert len(_entries(cache, snapshot, "policy_param")) == 2
        # 两个参数变体都仍可命中。
        assert engine.compute(
            _Parameterised(1), snapshot, start, end, None, end
        ).from_cache


def test_different_cutoffs_are_not_purged(world):
    snapshot, days = world
    start, end = days[5], days[-5]
    with tempfile.TemporaryDirectory() as cache:
        engine = FactorEngine(cache)
        engine.compute(_Probe(), snapshot, start, end, None, end)
        engine.compute(_Probe(), snapshot, start, end, None, days[-1])
        assert len(_entries(cache, snapshot, "policy_probe")) == 2


def test_a_hit_upgrades_legacy_metadata_and_makes_it_purgeable(
    world, monkeypatch
):
    snapshot, days = world
    start, end = days[5], days[-1]
    with tempfile.TemporaryDirectory() as cache:
        engine = FactorEngine(cache)
        _pretend_source(monkeypatch, "class _Probe: pass  # v1")
        engine.compute(_Probe(), snapshot, start, end, None, end)
        [entry] = _entries(cache, snapshot, "policy_probe")
        # 人工降级回 1.0：抹掉判别字段，模拟旧引擎写下的存量条目。
        metadata = json.loads(entry.read_text(encoding="utf-8"))
        for key in ("definition_key", "source_key", "instruments",
                    "dataset_id", "factor_version", "created_at"):
            metadata.pop(key, None)
        metadata["schema_version"] = "1.0"
        entry.write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

        # 源码变体写入时，无判别字段的存量条目必须被保守跳过。
        _pretend_source(monkeypatch, "class _Probe: pass  # v2")
        engine.compute(_Probe(), snapshot, start, end, None, end)
        assert len(_entries(cache, snapshot, "policy_probe")) == 2

        # 命中一次旧条目 → 元数据原地升级到 1.1。
        _pretend_source(monkeypatch, "class _Probe: pass  # v1")
        hit = engine.compute(_Probe(), snapshot, start, end, None, end)
        assert hit.from_cache
        upgraded = json.loads(entry.read_text(encoding="utf-8"))
        assert upgraded["schema_version"] == "1.1"
        assert upgraded["definition_key"]

        # 升级后，下一次代码变体写入把 v1、v2 两个旧版一起清掉。
        _pretend_source(monkeypatch, "class _Probe: pass  # v3")
        final = engine.compute(_Probe(), snapshot, start, end, None, end)
        entries = _entries(cache, snapshot, "policy_probe")
        assert len(entries) == 1
        assert json.loads(entries[0].read_text(encoding="utf-8"))[
            "cache_key"
        ] == final.cache_key


def test_compute_policy_materializes_then_loads_a_sparse_view(world):
    snapshot, days = world
    start, end = days[5], days[-1]
    requested = (days[7], days[11], days[19])
    with tempfile.TemporaryDirectory() as cache:
        engine = FactorEngine(cache, cache_policy="compute")
        first = engine.load_view(
            _Probe(), snapshot, start, end, dates=requested, cutoff=end,
        )
        assert not first.source_from_cache
        assert tuple(first.frame["trade_date"].unique()) == requested
        assert first.requested_dates == requested

        second = FactorEngine(
            cache, cache_policy="require"
        ).load_view(
            _Probe(), snapshot, days[7], days[19],
            dates=requested, cutoff=end,
        )
        assert second.source_from_cache
        assert second.cache_key == first.cache_key
        assert second.cache_start == start
        assert second.cache_end == end
        pd.testing.assert_frame_equal(first.frame, second.frame)


def test_require_policy_miss_is_read_only(world):
    snapshot, days = world
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary) / "does-not-exist"
        engine = FactorEngine(cache, cache_policy="require")
        with pytest.raises(FactorCacheMiss, match="policy_probe"):
            engine.load_view(
                _Probe(), snapshot, days[5], days[-1],
                dates=[days[7]], cutoff=days[-1],
            )
        assert not cache.exists()


def test_require_policy_applies_to_compute_too(world):
    snapshot, days = world
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary) / "read-only"
        with pytest.raises(FactorCacheMiss, match="prewarm"):
            FactorEngine(cache, cache_policy="require").compute(
                _Probe(), snapshot, days[5], days[-1], None, days[-1],
            )
        assert not cache.exists()


def test_sparse_view_filters_instruments_without_changing_cache_identity(world):
    snapshot, days = world
    start, end = days[5], days[-1]
    with tempfile.TemporaryDirectory() as cache:
        engine = FactorEngine(cache)
        broad = engine.load_view(_Probe(), snapshot, start, end, cutoff=end)
        empty = engine.load_view(
            _Probe(), snapshot, start, end,
            dates=[days[8]], instruments=["000001.XSHE"], cutoff=end,
        )
        assert empty.frame.empty
        assert empty.cache_key == broad.cache_key
        assert empty.source_from_cache


def test_sparse_view_rejects_dates_outside_the_interval(world):
    snapshot, days = world
    with tempfile.TemporaryDirectory() as cache:
        with pytest.raises(ValueError, match="inside"):
            FactorEngine(cache).load_view(
                _Probe(), snapshot, days[5], days[10],
                dates=[days[4]], cutoff=days[-1],
            )
