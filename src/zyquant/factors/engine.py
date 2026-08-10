"""因子计算引擎：内容寻址缓存 + 依赖解析 + 产出校验。

引擎干四件事，因子本身一件都不用管：

1. **递归算依赖**，并检测循环依赖与「同名不同版本」冲突；
2. **算缓存键**（内容寻址），命中就直接返回，还能从更宽区间的缓存里切片；
3. **加锁计算**，先写临时文件再 `os.replace` 原子落盘，附带 sha256 校验；
4. **校验产出**并生成诊断（行数、缺失率、极值）。

缓存布局：

    <cache_root>/<数据集 fingerprint>/<因子名>/<cache_key>.parquet
    <cache_root>/<数据集 fingerprint>/<因子名>/<cache_key>.json

建议在项目中使用 `.zyquant/cache/factors` 作为 `cache_root`。

版本策略：**因子只有「当前」一个版本**。写入新缓存时，同一因子目录下
「定义、cutoff、universe 都相同、但 identity 不同」的条目（⇒ 只可能是因子
源码或其依赖变了）会被自动删除；参数化变体（definition 不同）与不同
cutoff/universe 的缓存合法共存、互不干扰。所以想核对某批因子值，认
`cache_key` 即可——同一份数据、同一份代码下它是唯一的。
"""
from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from zyquant.core.exceptions import FactorCacheMiss, FactorError
from zyquant.core.hashing import hash_file, hash_payload
from zyquant.data import DataSnapshot

from .base import BaseFactor, FactorContext, FactorResult, FactorView


@dataclass(frozen=True)
class _CacheEntry:
    path: Path
    metadata_path: Path
    cache_key: str
    identity_key: str
    start: date
    end: date
    diagnostics: dict


class FactorEngine:
    def __init__(
        self,
        cache_root: str | Path,
        lock_timeout: float = 120.0,
        cache_policy: str = "compute",
    ):
        if cache_policy not in {"compute", "require"}:
            raise ValueError(
                "factor cache_policy must be 'compute' or 'require'"
            )
        self.cache_root = Path(cache_root).expanduser().resolve()
        if cache_policy == "compute":
            self.cache_root.mkdir(parents=True, exist_ok=True)
        self.lock_timeout = lock_timeout
        self.cache_policy = cache_policy
        self._verified_files: set[tuple[str, int, int, str]] = set()

    def compute(
        self,
        factor: BaseFactor,
        snapshot: DataSnapshot,
        start: date,
        end: date,
        instruments: Sequence[str] | None = None,
        cutoff: date | None = None,
    ) -> FactorResult:
        """算（或取）一个因子在 `[start, end]` 上的值。

        `cutoff` 默认等于 `end`，即「只用到 end 当天为止的数据」。构建可复用的
        全历史面板时应显式传数据集末日，并且**所有调用方传同一个值**，
        否则缓存按 cutoff 分裂。

        `instruments` 会被排序去重后放进缓存键，所以传 `["600000","000001"]` 和
        `["000001","600000"]` 命中同一份缓存；但传 `None`（全市场）和传一个
        子集是**两份**不同的缓存，不会互相命中。日常消费一律传 `None`。
        """
        cutoff = cutoff or end
        if end > cutoff:
            raise FactorError("factor end date exceeds cutoff")
        return self._compute(
            factor, snapshot, start, end,
            tuple(sorted(map(str, instruments))) if instruments is not None else None,
            cutoff, [], {},
        )

    def load_view(
        self,
        factor: BaseFactor,
        snapshot: DataSnapshot,
        start: date,
        end: date,
        dates: Sequence[date] | None = None,
        instruments: Sequence[str] | None = None,
        cutoff: date | None = None,
    ) -> FactorView:
        """Read a sparse view from the canonical full-universe factor cache.

        ``dates`` and ``instruments`` are view filters only: neither changes
        the source cache identity or creates another cache entry. On a miss,
        ``compute`` policy materializes the continuous full-universe
        ``[start, end]`` cache first; ``require`` policy fails without writing.
        """
        cutoff = cutoff or end
        if end > cutoff:
            raise FactorError("factor end date exceeds cutoff")
        requested_dates = (
            tuple(sorted(set(dates))) if dates is not None else None
        )
        if requested_dates is not None and any(
            day < start or day > end for day in requested_dates
        ):
            raise ValueError(
                "factor view dates must fall inside the requested interval"
            )
        entry, source_from_cache = self._resolve_entry(
            factor, snapshot, start, end, cutoff, [], {},
        )
        self._verify_cache_file(entry.path, entry.metadata_path)
        columns = ["trade_date", "instrument_id", "value"]
        if requested_dates == ():
            frame = pd.DataFrame(columns=columns)
        else:
            predicate = (
                (pads.field("trade_date") >= start)
                & (pads.field("trade_date") <= end)
            )
            if requested_dates is not None:
                predicate = predicate & pads.field("trade_date").isin(
                    list(requested_dates)
                )
            if instruments is not None:
                wanted = sorted(set(map(str, instruments)))
                predicate = predicate & pads.field("instrument_id").isin(
                    wanted
                )
            frame = pads.dataset(
                entry.path, format="parquet"
            ).to_table(filter=predicate, columns=columns).to_pandas()
            frame["trade_date"] = pd.to_datetime(
                frame["trade_date"]
            ).dt.date
            frame["instrument_id"] = frame["instrument_id"].astype(str)
            frame.sort_values(
                ["trade_date", "instrument_id"],
                inplace=True, ignore_index=True,
            )
        return FactorView(
            name=factor.name,
            frame=frame,
            cache_key=entry.cache_key,
            cache_start=entry.start,
            cache_end=entry.end,
            requested_dates=requested_dates,
            source_from_cache=source_from_cache,
            diagnostics=entry.diagnostics,
        )

    @staticmethod
    def _identity_parts(
        factor: BaseFactor,
        snapshot: DataSnapshot,
        cutoff: date,
        instruments: tuple[str, ...] | None,
        dependency_keys: Sequence[str],
    ) -> tuple[str, str, str]:
        try:
            source = inspect.getsource(type(factor))
        except (OSError, TypeError):
            source = repr(type(factor))
        definition_key = hash_payload(factor.definition())
        source_key = hash_payload(source)
        identity_key = hash_payload({
            "dataset": snapshot.metadata.fingerprint,
            "factor": factor.definition(),
            "factor_source": source_key,
            "dependencies": list(dependency_keys),
            "cutoff": cutoff,
            "instruments": instruments,
            "engine": "1.0",
        })
        return identity_key, definition_key, source_key

    def _resolve_entry(
        self, factor, snapshot, start, end, cutoff,
        stack: list[tuple[str, str]], definitions: dict[str, str],
    ) -> tuple[_CacheEntry, bool]:
        node = (factor.name, factor.version)
        if node in stack:
            chain = " -> ".join(name for name, _ in stack + [node])
            raise FactorError(f"factor dependency cycle: {chain}")
        existing = definitions.get(factor.name)
        if existing is not None and existing != factor.version:
            raise FactorError(
                f"factor version conflict for {factor.name}: "
                f"{existing} vs {factor.version}"
            )
        definitions[factor.name] = factor.version
        stack.append(node)
        dependency_start = self._history_start(
            snapshot, start, int(getattr(factor, "lookback", 0))
        )
        dependency_entries = [
            self._resolve_entry(
                item, snapshot, dependency_start, end, cutoff,
                stack, definitions,
            )[0]
            for item in factor.dependencies
        ]
        stack.pop()
        identity_key, definition_key, source_key = self._identity_parts(
            factor, snapshot, cutoff, None,
            [item.cache_key for item in dependency_entries],
        )
        cache_key = hash_payload({
            "identity": identity_key, "start": start, "end": end,
        })
        directory = (
            self.cache_root / snapshot.metadata.fingerprint / factor.name
        )
        entry = self._cache_entry(
            directory, cache_key, identity_key
        ) or self._broader_entry(directory, identity_key, start, end)
        provenance = {
            "definition_key": definition_key,
            "source_key": source_key,
            "instruments": None,
            "dataset_id": snapshot.metadata.dataset_id,
            "factor_version": factor.version,
        }
        if entry is not None:
            self._upgrade_metadata(entry.metadata_path, provenance)
            return entry, True
        if self.cache_policy == "require":
            raise FactorCacheMiss(
                "required factor cache is missing: "
                f"factor={factor.name} start={start} end={end} "
                f"cutoff={cutoff} identity={identity_key}; "
                "prewarm it with scripts/build_factor_panel.py"
            )
        result = self._compute(
            factor, snapshot, start, end, None, cutoff, [], {},
        )
        entry = self._cache_entry(
            directory, result.cache_key, identity_key
        )
        if entry is None:
            raise FactorError(
                f"computed factor cache metadata is missing: {factor.name}"
            )
        return entry, bool(result.from_cache)

    @staticmethod
    def _cache_entry(
        directory: Path, cache_key: str, identity_key: str,
    ) -> _CacheEntry | None:
        metadata_path = directory / f"{cache_key}.json"
        path = directory / f"{cache_key}.parquet"
        if not path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata["cache_key"] != cache_key
                or metadata["identity_key"] != identity_key
            ):
                return None
            return _CacheEntry(
                path=path,
                metadata_path=metadata_path,
                cache_key=cache_key,
                identity_key=identity_key,
                start=date.fromisoformat(metadata["start"]),
                end=date.fromisoformat(metadata["end"]),
                diagnostics=dict(metadata["diagnostics"]),
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def _broader_entry(
        self, directory: Path, identity_key: str, start: date, end: date,
    ) -> _CacheEntry | None:
        if not directory.exists():
            return None
        for metadata_path in sorted(directory.glob("*.json")):
            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                cache_start = date.fromisoformat(metadata["start"])
                cache_end = date.fromisoformat(metadata["end"])
                if (
                    metadata.get("identity_key") == identity_key
                    and cache_start <= start and cache_end >= end
                ):
                    key = str(metadata["cache_key"])
                    path = directory / f"{key}.parquet"
                    if path.exists():
                        return _CacheEntry(
                            path, metadata_path, key, identity_key,
                            cache_start, cache_end,
                            dict(metadata["diagnostics"]),
                        )
            except (
                OSError, ValueError, KeyError, json.JSONDecodeError,
            ):
                continue
        return None

    def _compute(
        self, factor, snapshot, start, end, instruments, cutoff,
        stack: list[tuple[str, str]], definitions: dict[str, str],
    ):
        # --- 依赖图的两道守卫 -------------------------------------------------
        # `stack` 是当前递归路径，用来发现环；`definitions` 是全图已见过的
        # (因子名 -> 版本)，用来发现「同一个名字挂了两个版本」——那会让缓存
        # 语义不唯一，必须直接拒绝而不是随便选一个。
        node = (factor.name, factor.version)
        if node in stack:
            chain = " -> ".join(name for name, _ in stack + [node])
            raise FactorError(f"factor dependency cycle: {chain}")
        existing = definitions.get(factor.name)
        if existing is not None and existing != factor.version:
            raise FactorError(
                f"factor version conflict for {factor.name}: {existing} vs {factor.version}"
            )
        definitions[factor.name] = factor.version
        stack.append(node)
        # 依赖必须比本因子多算 `lookback` 个交易日的历史，否则本因子在 start
        # 当天就拿不到足够的依赖值。
        dependency_start = self._history_start(
            snapshot, start, int(getattr(factor, "lookback", 0))
        )
        dependency_results = [
            self._compute(
                item, snapshot, dependency_start, end, instruments, cutoff,
                stack, definitions,
            )
            for item in factor.dependencies
        ]
        stack.pop()
        # identity_key 不含区间：它回答「这是哪个因子的哪个算法在哪份数据上」。
        identity_key, definition_key, source_key = self._identity_parts(
            factor, snapshot, cutoff, instruments,
            [item.cache_key for item in dependency_results],
        )
        cache_key = hash_payload({
            "identity": identity_key, "start": start, "end": end,
        })
        directory = self.cache_root / snapshot.metadata.fingerprint / factor.name
        path = directory / f"{cache_key}.parquet"
        metadata_path = directory / f"{cache_key}.json"
        provenance = {
            "definition_key": definition_key,
            "source_key": source_key,
            "instruments": list(instruments) if instruments is not None else None,
            "dataset_id": snapshot.metadata.dataset_id,
            "factor_version": factor.version,
        }
        # 一级命中：区间完全相同。
        cached = self._read_cache(path, metadata_path, start, end)
        if cached is not None:
            self._upgrade_metadata(metadata_path, provenance)
            return FactorResult(
                factor.name, cached[0], cache_key, True, cached[1]
            )
        # 二级命中：存在一个**更宽**区间的同 identity 缓存，直接切片。
        # 这正是「一次算全历史、之后任意窄区间都免费」的机制。
        broader = self._find_broader(directory, identity_key, start, end)
        if broader is not None:
            frame, diagnostics, broader_key = broader
            self._upgrade_metadata(
                directory / f"{broader_key}.json", provenance
            )
            return FactorResult(factor.name, frame, broader_key, True, diagnostics)
        if self.cache_policy == "require":
            raise FactorCacheMiss(
                "required factor cache is missing: "
                f"factor={factor.name} start={start} end={end} "
                f"cutoff={cutoff} identity={identity_key}; "
                "prewarm it with scripts/build_factor_panel.py"
            )
        directory.mkdir(parents=True, exist_ok=True)
        # --- 真算 -------------------------------------------------------------
        # 文件锁让并发的多个进程只算一遍：抢不到锁的一方等成品出现后直接读。
        lock = path.with_suffix(".lock")
        acquired = self._acquire(lock, metadata_path)
        if not acquired:
            cached = self._read_cache(path, metadata_path, start, end)
            if cached is None:
                raise FactorError(f"factor cache completed without valid output: {path}")
            return FactorResult(factor.name, cached[0], cache_key, True, cached[1])
        temporary = path.with_suffix(".tmp.parquet")
        temporary_metadata = metadata_path.with_suffix(".tmp.json")
        try:
            context = FactorContext(snapshot, start, end, cutoff, instruments)
            dependencies = {item.name: item.frame for item in dependency_results}
            frame = factor.compute(context, dependencies).copy()
            diagnostics = self._validate(factor.name, frame, start, end, instruments)
            frame.to_parquet(temporary, index=False)
            payload = {
                "schema_version": "1.1",
                "cache_key": cache_key,
                "identity_key": identity_key,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "cutoff": cutoff.isoformat(),
                "parquet_sha256": hash_file(temporary),
                "diagnostics": diagnostics,
                "upstream": [item.cache_key for item in dependency_results],
                # 1.1 起的溯源字段：让每个条目自我描述（谁的定义、哪份数据、
                # 何时算的），并支撑下面的单版本清理判别。不进 identity。
                "definition_key": definition_key,
                "source_key": source_key,
                "instruments": (
                    list(instruments) if instruments is not None else None
                ),
                "dataset_id": snapshot.metadata.dataset_id,
                "factor_version": factor.version,
                "created_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
            temporary_metadata.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # 先落数据再落元数据都用 os.replace：任何时刻旁观者要么看到旧的一对，
            # 要么看到新的一对，不会看到写了一半的 parquet。
            os.replace(temporary, path)
            os.replace(temporary_metadata, metadata_path)
            # 新版落盘成功后才清旧版（仍持有锁）：宁可短暂多一份，不可少一份。
            self._purge_stale(directory, identity_key, definition_key, cutoff,
                              instruments)
            return FactorResult(factor.name, frame, cache_key, False, diagnostics)
        except Exception:
            # 算失败绝不留下半成品，否则下次会当成有效缓存读进去。
            temporary.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
            raise
        finally:
            lock.unlink(missing_ok=True)

    def _purge_stale(self, directory, identity_key, definition_key, cutoff,
                     instruments):
        """单版本策略的落地处：删除被代码改动淘汰的旧版缓存。

        判据必须同时满足「定义相同、cutoff 相同、universe 相同、identity 不同」
        ——identity 的差异此时只可能来自因子源码或其依赖，即旧代码的产物。
        参数化变体（definition 不同）与不同 cutoff/universe 的条目不受影响，
        参数搜索之类的合法共存不会互相误删。

        旧 schema（1.0）的元数据没有判别字段，跳过不动；它们会在被命中时由
        `_upgrade_metadata` 原地升级，此后自然纳入清理范围。

        删除与并发读之间存在窄竞态（另一进程恰在读将被删的旧版）：只会发生在
        「改了因子代码的同时还有旧代码进程在跑」的过渡瞬间，读方会得到明确的
        FactorError 而非错误数据，可接受。
        """
        wanted = list(instruments) if instruments is not None else None
        for metadata_path in directory.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("identity_key") == identity_key:
                continue
            if metadata.get("definition_key") != definition_key:
                continue
            if metadata.get("cutoff") != cutoff.isoformat():
                continue
            if metadata.get("instruments") != wanted:
                continue
            stale_key = metadata.get("cache_key", "")
            (directory / f"{stale_key}.parquet").unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)

    @staticmethod
    def _upgrade_metadata(metadata_path, provenance):
        """把命中的旧 schema（1.0）元数据原地升级到 1.1。

        命中即证明该条目的 identity 与当前代码一致，此刻补写的溯源字段
        （definition_key/source_key/instruments/dataset_id）就是它的真实出身。
        这让存量缓存逐次获得单版本清理所需的判别字段，无需一次性迁移。
        """
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if metadata.get("schema_version") != "1.0":
            return
        metadata.update(provenance)
        metadata["schema_version"] = "1.1"
        temporary = metadata_path.with_suffix(".upgrade.json")
        try:
            temporary.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, metadata_path)
        except OSError:
            temporary.unlink(missing_ok=True)

    def _read_cache(self, path, metadata_path, start, end):
        """读缓存并按请求区间切片；哈希不符即报错，不静默使用。

        元数据里的 `parquet_sha256` 是防篡改/防写坏，不是防过期——过期由
        身份键负责。缓存文件可能覆盖比请求更宽的区间，所以读完要裁剪。
        """
        if not path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._verify_cache_file(
                path, metadata_path, metadata=metadata
            )
            frame = pd.read_parquet(path)
            frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
            frame = frame[
                (frame["trade_date"] >= start) & (frame["trade_date"] <= end)
            ].reset_index(drop=True)
            return frame, metadata["diagnostics"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise FactorError(f"invalid factor cache metadata: {metadata_path}") from exc

    def _verify_cache_file(
        self, path: Path, metadata_path: Path, metadata: dict | None = None,
    ) -> None:
        try:
            payload = metadata or json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            expected = str(payload["parquet_sha256"])
            stat = path.stat()
            verification = (
                str(path), int(stat.st_size), int(stat.st_mtime_ns), expected,
            )
            if verification in self._verified_files:
                return
            if hash_file(path) != expected:
                raise FactorError(f"factor cache hash mismatch: {path}")
            self._verified_files.add(verification)
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise FactorError(
                f"invalid factor cache metadata: {metadata_path}"
            ) from exc

    def _find_broader(self, directory, identity_key, start, end):
        """在同一因子目录下找一个覆盖 `[start, end]` 的同 identity 缓存。

        注意返回的 `diagnostics` 是**宽区间**的诊断（行数、缺失率都是宽窗口的），
        不是切片后的。别拿它判断切片数据的质量。
        """
        if not directory.exists():
            return None
        for metadata_path in sorted(directory.glob("*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("identity_key") == identity_key
                    and date.fromisoformat(metadata["start"]) <= start
                    and date.fromisoformat(metadata["end"]) >= end
                ):
                    key = metadata["cache_key"]
                    cached = self._read_cache(
                        directory / f"{key}.parquet", metadata_path, start, end
                    )
                    if cached is not None:
                        return cached[0], cached[1], key
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return None

    @staticmethod
    def _history_start(snapshot, start, bars):
        """按交易日历往前推 `bars` 个交易日。"""
        if bars <= 0:
            return start
        calendar = sorted(set(snapshot.table("trade_calendar")["trade_date"]))
        prior = [day for day in calendar if day <= start]
        if not prior:
            return start
        index = calendar.index(prior[-1])
        return calendar[max(0, index - bars)]

    def _acquire(self, lock: Path, completed: Path) -> bool:
        """`O_CREAT|O_EXCL` 独占创建锁文件；返回 False 表示别人已经算完了。

        三种退出：抢到锁（True）；发现成品已存在（False，去读缓存）；
        锁文件比 `lock_timeout` 还旧则视为持锁进程已死，抢占重试。
        """
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(descriptor)
                return True
            except FileExistsError:
                if completed.exists():
                    return False
                try:
                    if time.time() - lock.stat().st_mtime > self.lock_timeout:
                        lock.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise FactorError(f"timed out waiting for factor cache lock: {lock}")
                time.sleep(0.05)

    @staticmethod
    def _validate(name, frame, start, end, instruments):
        """产出契约的强制校验——这是「一个因子只能是每格一个标量」的落地处。

        逐条：三列必备；(日期, 标的) 不得重复；不得越出请求区间；数值必须有限
        （**NaN 允许、inf 不允许**——NaN 表示「这格算不出来」是合法信息，
        inf 一定是除零之类的错误）；不得返回没被请求的标的。

        返回的诊断会写进缓存元数据，是事后核对的主要依据：`missing_rate` 突变
        往往就是上游数据出了问题。
        """
        required = {"trade_date", "instrument_id", "value"}
        if required - set(frame.columns):
            raise FactorError(f"factor {name} must output {sorted(required)}")
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        if frame.duplicated(["trade_date", "instrument_id"]).any():
            raise FactorError(f"factor {name} contains duplicate keys")
        if ((frame["trade_date"] < start) | (frame["trade_date"] > end)).any():
            raise FactorError(f"factor {name} returned rows outside requested range")
        values = pd.to_numeric(frame["value"], errors="coerce")
        if not np.isfinite(values.dropna()).all():
            raise FactorError(f"factor {name} contains non-finite values")
        if instruments is not None and set(frame["instrument_id"]) - set(instruments):
            raise FactorError(f"factor {name} returned unrequested instruments")
        frame.sort_values(["trade_date", "instrument_id"], inplace=True, ignore_index=True)
        total = len(frame)
        return {
            "rows": total,
            "instruments": int(frame["instrument_id"].nunique()) if total else 0,
            "dates": int(frame["trade_date"].nunique()) if total else 0,
            "missing": int(values.isna().sum()),
            "missing_rate": float(values.isna().mean()) if total else 0.0,
            "minimum": float(values.min()) if values.notna().any() else None,
            "maximum": float(values.max()) if values.notna().any() else None,
        }
