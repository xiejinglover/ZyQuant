"""A 股通用因子：自算 Beta、回归动量、质量三项、股息率。

这些都是**通用**的 A 股因子，不属于任何单一策略，所以放在与 `builtin.py`
并列的位置，用 `from zyquant.factors import ...` 即可导入。

共同性质：每个因子对请求区间内的每个 `(trade_date, instrument_id)` 产出一个值，
且**每个值只由该行日期及之前的数据算出**。正是这条逐行 PIT 性质，让我们可以
用一个很宽的 `cutoff` 一次算完全历史、之后任意切片——`cutoff` 限制的是
「总共能读到哪些行」，每行的回溯窗口限制的是「实际进入这个值的是哪些数据」。
`tests/test_cn_equity_factors.py` 用「换窄区间重算、共同日期必须逐位相同」
把这条性质钉死。

全 A 历史（约 2600 个交易日 × 5800 个标的）的实测成本：Beta 单进程约 41 分钟、
动量约 10 分钟。两者都按日期天然可并行，传 `workers` 即可。
**线程无效**——内层循环是 Python 绑定（GIL），不是 numpy 绑定，实测 4/8 线程
反而是 0.83× / 0.77×，必须用进程。
"""
from __future__ import annotations

import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .base import BaseFactor, FactorContext

# 每个子进程都必须把 BLAS 线程数钉成 1，否则 N 个进程各开一个线程池互相抢核，
# 越并行越慢。与 Hermes 财务归一化用的是同一份清单。
_THREAD_LIMITS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """长表 → `(交易日 × 标的)` 宽表。滚动窗口计算只能在宽表上做。"""
    return frame.pivot_table(
        index="trade_date", columns="instrument_id", values=column,
        aggfunc="last",
    ).sort_index()


def _trading_days(context: FactorContext) -> list[date]:
    """`cutoff`（经 `context.end`）以内的全部交易日，升序去重。

    回溯窗口一律按这个列表数格子，不按自然日。
    """
    calendar = context.snapshot.table("trade_calendar")
    return sorted({
        day for day in calendar["trade_date"] if day <= context.end
    })


def _restrict_to_listed(
    frame: pd.DataFrame, context: FactorContext,
) -> pd.DataFrame:
    """剔掉标的在该日期尚未上市或已经退市的行。

    **价格类因子天然不需要这一步**（退市了就没有 bar，值自然就没了），
    **但财务类因子必须要**：财务值是个阶梯函数，放任不管就会把最后一期报表
    一路前向填充下去——2015 年退市的公司到 2026 年还在报净利润。
    财务指标里还混着从来不在 `instruments` 表里的发行主体
    （厂商内部代码、上市前记录）。

    这两种都不是有意义的因子值，所以在这里统一剔除，而不是留给每个消费方
    自己记得处理。判据与股票池选择器用的一致，因子和股票池才不会打架。

    历史教训：这个函数是补上去的。此前财务因子有 754706 行（4.6%）落在上市
    窗口之外、977 个孤儿标的，表现为「财务因子行数比行情因子还多」。
    """
    if frame.empty:
        return frame
    instruments = context.snapshot.table("instruments")
    listed = pd.to_datetime(instruments["list_date"], errors="coerce")
    delisted = pd.to_datetime(instruments["delist_date"], errors="coerce")
    window = pd.DataFrame({
        "instrument_id": instruments["instrument_id"].astype(str),
        "_listed": listed,
        "_delisted": delisted,
    })
    # inner join 同时干掉了「不在 instruments 表里」的孤儿标的。
    merged = frame.merge(window, on="instrument_id", how="inner")
    stamp = pd.to_datetime(merged["trade_date"])
    # 左闭右开：上市日算在内，退市日不算——退市日是最后一个交易日之后的那天，
    # 已用真实数据核对（面板末日与退市日的缺口中位数与最大值都是 0 个交易日）。
    alive = (stamp >= merged["_listed"]) & (
        merged["_delisted"].isna() | (stamp < merged["_delisted"])
    )
    return merged.loc[alive, ["trade_date", "instrument_id", "value"]]


def _emit(
    values: Mapping[date, Mapping[str, float]],
    context: FactorContext,
) -> pd.DataFrame:
    """`{日期: {标的: 值}}` → 契约要求的三列长表，并做上市窗口裁剪。"""
    rows = [
        {"trade_date": day, "instrument_id": code, "value": value}
        for day, per_day in values.items()
        for code, value in per_day.items()
    ]
    if not rows:
        return pd.DataFrame(
            columns=["trade_date", "instrument_id", "value"]
        )
    return _restrict_to_listed(pd.DataFrame(rows), context)


_WORKER = None


def _run_worker(day: date) -> Any:
    """模块级蹦床函数，给进程池一个可 pickle 的调用目标。

    真正的 worker 是闭包（闭在已加载的面板上），闭包不能 pickle。
    子进程通过 fork 继承全局 `_WORKER`，顺带让面板以写时复制方式共享，
    而不是每个进程复制一份几 GB 的面板。
    """
    if _WORKER is None:  # pragma: no cover - only if forked incorrectly
        raise RuntimeError("factor worker was not installed before forking")
    return _WORKER(day)


def _map_dates(
    dates: Sequence[date], worker, workers: int,
) -> list[Any]:
    """把 `worker` 铺到各个日期上，可选多进程。

    日期之间彼此独立——每个因子值只依赖自己的回溯窗口——所以这就是一次纯粹的
    fan-out，没有跨日期的归约。实测 4 进程并行效率 92%。
    线程没有用：内层循环持有 GIL。
    """
    global _WORKER
    if workers <= 1 or len(dates) <= 1:
        return [worker(day) for day in dates]
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - platform without fork
        return [worker(day) for day in dates]
    for name in _THREAD_LIMITS:
        os.environ[name] = "1"
    _WORKER = worker
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as pool:
            return list(pool.map(_run_worker, dates))
    finally:
        _WORKER = None


# ----------------------------------------------------------------- market data

# 面板按 (数据集 fingerprint, 起, 止) 记忆化：全历史读一次要几分钟，
# 而 Beta 与动量共用同一份收盘价面板。
_PANEL_CACHE: dict[tuple[str, date, date], "MarketPanels"] = {}


class MarketPanels:
    """全市场的后复权收盘价、停牌标记与流通市值。

    注意这里**无条件读全市场**，与 `context.instruments` 无关：Beta 需要
    全 A 市值加权的市场收益，只读子集算出来的「市场」是错的。因子随后只对
    请求的标的产出行。
    """

    def __init__(self, snapshot: Any, start: date, end: date):
        adjusted = snapshot.post_adjusted_bars(
            start, end, None, ["close_post"], end,
        )
        raw = snapshot.table(
            "daily_raw", start, end, cutoff=end, fields=["paused"],
        )
        valuation = snapshot.table(
            "daily_valuation", start, end, cutoff=end,
            fields=["circulating_market_cap"],
        )
        self.close = _pivot(adjusted, "close_post")
        # 停牌与市值都对齐到收盘价面板的行列上：缺失的停牌标记按「未停牌」处理，
        # 因为没有 bar 的格子后面会被 NaN 收益率过滤掉，不需要这里再判一次。
        self.paused = _pivot(raw, "paused").reindex(
            index=self.close.index, columns=self.close.columns
        ).fillna(False).astype(bool)
        self.cap = _pivot(valuation, "circulating_market_cap").reindex(
            index=self.close.index, columns=self.close.columns
        )
        self._returns: tuple[pd.DataFrame, pd.Series] | None = None

    def returns(self) -> tuple[pd.DataFrame, pd.Series]:
        """个股日收益率矩阵，以及市值加权的市场收益率序列。

        三个关键处理：

        1. **用前一日的流通市值做权重**（`self.cap.shift(1)`）。用当日市值等于
           把当日涨跌反馈进自己的权重里，涨得多的自动加权更多，市场收益被高估。
        2. **停牌日收益记 0**，不是 NaN——停牌不是缺数据，是真的没有收益。
        3. 权重只在「个股与市场当日都有效」的子集上取（`valid`），
           所以**每只股票的有效日集合不同**。这一点在向量化改写时是最大的坑：
           分母 `Σw` 是个股相关的（实测一只股票缺 4 天，分母相对差 5.26e-03），
           而全有效的测试用例完全发现不了掩码写错。

        每一天的值只依赖当天与前一天，与「用多宽的窗口看它」无关，
        所以整个面板算一次、按日期切片使用。
        """
        if self._returns is not None:
            return self._returns
        returns = self.close.pct_change(fill_method=None)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.mask(self.paused, 0.0)
        previous = self.cap.shift(1)
        valid = returns.notna() & previous.notna() & (previous > 0)
        weighted = previous.where(valid)
        numerator = (returns.where(valid) * weighted).sum(axis=1)
        # 全市场当日无有效市值时分母为 0 → 置 NaN，市场收益缺失，
        # 下游 Beta 会因为 mask 里没有有效观测而产出 NaN，不会产出 0。
        denominator = weighted.sum(axis=1).replace(0.0, np.nan)
        market = (numerator / denominator).replace(
            [np.inf, -np.inf], np.nan
        )
        self._returns = (returns, market)
        return self._returns


def market_panels(context: FactorContext, lookback: int) -> MarketPanels:
    """取（或建）记忆化的市场面板。

    键里含 `history_start(lookback)`，所以不同 lookback 的因子会各建一份。
    长任务里要主动 `clear_panel_cache()` 释放内存。
    """
    key = (
        context.snapshot.metadata.fingerprint,
        context.history_start(lookback),
        context.end,
    )
    if key not in _PANEL_CACHE:
        _PANEL_CACHE[key] = MarketPanels(
            context.snapshot, key[1], key[2]
        )
    return _PANEL_CACHE[key]


def clear_panel_cache() -> None:
    """丢弃记忆化的市场面板；主要给测试和长驻任务用。"""
    _PANEL_CACHE.clear()


# ------------------------------------------------------------------- dividends


class DividendYieldFactor(BaseFactor):
    """近 12 个月现金股息率，直接取快照里的厂商序列。

    厂商序列本身已经是滚动 12 月口径，所以不需要回溯窗口（`lookback = 0`）。
    存的是**比率**不是百分数（归一化时已经 /100）。

    两条来自实战的注意事项：

    * 源字段是 `DIV_RATE_L12M` 而**不是** `DIV_RATE_TTM`。后者在每年 1-4 月
      财报季会塌陷 95%，曾导致 2016-01-29 的候选池只剩 5 只（正确值 106 只）。
    * **不要自己用 `corporate_actions` 手搓股息率**。已做全市场对拍：厂商值
      与「近 12 月现金分红 ÷ 不复权收盘」的比值中位数是 1.0000，偏差全部来自
      厂商会**按后续送转重述每股分红**，朴素求和不会（送 1.0 → 差 2 倍，
      送 3.0 → 差 4 倍）。
    """

    name = "dividend_yield_l12m"
    version = "1"
    lookback = 0
    inputs = ("daily_valuation.dividend_yield",)

    def compute(self, context, dependencies):
        frame = context.valuation(["dividend_yield"])
        frame = frame[
            frame["trade_date"].between(context.start, context.end)
        ]
        result = frame.rename(columns={"dividend_yield": "value"})
        return _restrict_to_listed(
            result[["trade_date", "instrument_id", "value"]], context
        )


class DividendCredibilityFactor(BaseFactor):
    """One-year log decomposition of dividend-yield changes.

    ``cash_proxy = dividend_yield_l12m * market_cap`` is invariant to stock
    splits and represents trailing cash distribution at company level.  The
    three published metrics obey the exact identity

    ``dlog(yield) = dlog(cash_proxy) - dlog(market_cap)``.

    Market-value change is intentionally not labelled pure price return:
    issuance and cancellation may also contribute and belong in a separate
    anti-dilution experiment.
    """

    version = "1"
    inputs = (
        "daily_valuation.dividend_yield",
        "daily_valuation.market_cap",
    )

    def __init__(self, metric: str, window: int = 252):
        names = {
            "cash_log_growth": "dividend_cash_log_growth_1y",
            "market_value_log_growth": "market_value_log_growth_1y",
            "yield_log_change": "dividend_yield_log_change_1y",
        }
        if metric not in names:
            raise ValueError(f"unsupported dividend credibility metric: {metric}")
        self.metric = metric
        self.window = int(window)
        self.lookback = self.window
        self.name = names[metric]

    def definition(self) -> Mapping[str, Any]:
        return {
            **super().definition(),
            "metric": self.metric,
            "window": self.window,
            "cash_proxy": "dividend_yield_l12m_times_market_cap",
        }

    def compute(self, context, dependencies):
        raw = context.valuation(
            ["dividend_yield", "market_cap"], lookback=self.lookback
        )
        dividend_yield = _pivot(raw, "dividend_yield")
        market_cap = _pivot(raw, "market_cap")
        columns = dividend_yield.columns.union(market_cap.columns)
        dividend_yield = dividend_yield.reindex(columns=columns)
        market_cap = market_cap.reindex(columns=columns)
        valid_yield = dividend_yield.where(dividend_yield > 0.0)
        valid_market_cap = market_cap.where(market_cap > 0.0)
        cash_proxy = valid_yield * valid_market_cap
        if self.metric == "cash_log_growth":
            level = cash_proxy
        elif self.metric == "market_value_log_growth":
            level = valid_market_cap
        else:
            level = valid_yield
        change = np.log(level) - np.log(level.shift(self.window))
        targets = pd.Index(
            [
                day for day in _trading_days(context)
                if context.start <= day <= context.end
            ],
            name="trade_date",
        )
        change = change.reindex(targets).replace([np.inf, -np.inf], np.nan)
        long = change.stack(future_stack=True).rename("value").reset_index()
        long.columns = ["trade_date", "instrument_id", "value"]
        return _restrict_to_listed(long.dropna(subset=["value"]), context)


class ValuationMultipleFactor(BaseFactor):
    """PIT valuation input derived from the signal-date valuation row.

    Invalid and non-positive multiples are deliberately absent.  Cross-
    sectional winsorisation belongs to the consuming strategy because its
    mother pool is part of the signal definition, not the factor definition.
    """

    version = "1"
    lookback = 0

    def __init__(self, metric: str):
        if metric not in {"earnings_yield", "pb"}:
            raise ValueError(f"unsupported valuation metric: {metric}")
        self.metric = metric
        self.name = "earnings_yield_ttm" if metric == "earnings_yield" else "pb_ratio"
        self.source_field = "pe_ttm" if metric == "earnings_yield" else "pb"
        self.inputs = (f"daily_valuation.{self.source_field}",)

    def definition(self) -> Mapping[str, Any]:
        return {
            **super().definition(), "metric": self.metric,
            "source_field": self.source_field,
            "invalid_policy": "drop_non_positive",
        }

    def compute(self, context, dependencies):
        frame = context.valuation([self.source_field])
        frame = frame[frame["trade_date"].between(context.start, context.end)]
        raw = pd.to_numeric(frame[self.source_field], errors="coerce")
        raw = raw.where(raw > 0.0)
        value = 1.0 / raw if self.metric == "earnings_yield" else raw
        result = frame[["trade_date", "instrument_id"]].copy()
        result["value"] = value.replace([np.inf, -np.inf], np.nan)
        return _restrict_to_listed(result.dropna(subset=["value"]), context)


class DividendContinuityFactor(BaseFactor):
    """Number of trailing 365-day buckets containing a realised cash dividend."""

    name = "dividend_continuity_3y"
    version = "1"
    inputs = (
        "corporate_actions.event_type", "corporate_actions.status",
        "corporate_actions.cash_per_share", "corporate_actions.announced_at",
        "corporate_actions.ex_date",
    )

    @staticmethod
    def _counts(targets: Sequence[date], events: Sequence[date]) -> np.ndarray:
        event_ordinals = np.asarray(
            sorted({item.toordinal() for item in events}), dtype=np.int64
        )
        target_ordinals = np.asarray(
            [item.toordinal() for item in targets], dtype=np.int64
        )
        result = np.zeros(len(target_ordinals), dtype=np.int8)
        if not len(event_ordinals):
            return result
        for bucket in range(3):
            upper = target_ordinals - 365 * bucket
            lower = target_ordinals - 365 * (bucket + 1)
            count = (
                np.searchsorted(event_ordinals, upper, side="right")
                - np.searchsorted(event_ordinals, lower, side="right")
            )
            result += (count > 0).astype(np.int8)
        return result

    def compute(self, context, dependencies):
        targets = [
            day for day in _trading_days(context)
            if context.start <= day <= context.end
        ]
        actions = context.corporate_actions(lookback_calendar_days=1095)
        visible = actions[
            (actions["event_type"].astype(str) == "cash_dividend")
            & (actions["status"].astype(str) == "active")
            & (pd.to_numeric(actions["cash_per_share"], errors="coerce") > 0)
        ].copy()
        visible["announced_at"] = pd.to_datetime(
            visible["announced_at"], errors="coerce"
        ).dt.date
        visible["ex_date"] = pd.to_datetime(
            visible["ex_date"], errors="coerce"
        ).dt.date
        visible = visible[
            visible["announced_at"].notna()
            & visible["ex_date"].notna()
            & (visible["announced_at"] <= visible["ex_date"])
        ]
        # The event date is the strict per-row PIT boundary.  announced_at is
        # also checked so a malformed future announcement cannot leak in.
        grouped = {
            str(code): tuple(group["ex_date"].dropna())
            for code, group in visible.groupby("instrument_id", sort=False)
        }
        instruments = context.snapshot.table("instruments")
        codes = (
            list(context.instruments) if context.instruments is not None
            else instruments["instrument_id"].astype(str).tolist()
        )
        frames = []
        for code in codes:
            dates = grouped.get(str(code), ())
            counts = self._counts(targets, dates)
            frames.append(pd.DataFrame({
                "trade_date": targets,
                "instrument_id": str(code),
                "value": counts.astype(float),
            }))
        if not frames:
            return pd.DataFrame(
                columns=["trade_date", "instrument_id", "value"]
            )
        # announced_at <= each target is structurally true for realised events
        # in the canonical data (announced_at <= ex_date <= target).
        return _restrict_to_listed(
            pd.concat(frames, ignore_index=True), context
        )


class DividendYieldHistoryFactor(BaseFactor):
    """Three-year month-end median yield or current/median anomaly ratio."""

    version = "1"
    inputs = ("daily_valuation.dividend_yield",)

    def __init__(self, metric: str, months: int = 36, minimum_months: int = 24):
        if metric not in {"median", "anomaly"}:
            raise ValueError(f"unsupported dividend history metric: {metric}")
        self.metric = metric
        self.months = int(months)
        self.minimum_months = int(minimum_months)
        self.lookback = 800
        self.name = (
            "dividend_yield_median_3y"
            if metric == "median" else "dividend_yield_anomaly_3y"
        )

    def definition(self) -> Mapping[str, Any]:
        return {
            **super().definition(), "metric": self.metric,
            "months": self.months, "minimum_months": self.minimum_months,
        }

    def compute(self, context, dependencies):
        raw = context.valuation(["dividend_yield"], lookback=self.lookback)
        matrix = _pivot(raw, "dividend_yield")
        if matrix.empty:
            return pd.DataFrame(
                columns=["trade_date", "instrument_id", "value"]
            )
        stamps = pd.DatetimeIndex(pd.to_datetime(matrix.index))
        monthly = matrix.assign(_month=stamps.to_period("M")).groupby(
            "_month", sort=True
        ).tail(1)
        monthly = monthly.drop(columns="_month")
        median = monthly.rolling(
            self.months, min_periods=self.minimum_months
        ).median()
        targets = [
            day for day in _trading_days(context)
            if context.start <= day <= context.end
        ]
        target_index = pd.DatetimeIndex(pd.to_datetime(pd.Series(targets)))
        monthly_index = pd.DatetimeIndex(pd.to_datetime(monthly.index))
        median.index = monthly_index
        expanded = median.reindex(median.index.union(target_index)).ffill()
        expanded = expanded.reindex(target_index)
        expanded.index = pd.Index(targets, name="trade_date")
        if self.metric == "anomaly":
            current = matrix.copy()
            current.index = pd.DatetimeIndex(pd.to_datetime(current.index))
            current = current.reindex(target_index)
            current.index = expanded.index
            expanded = current / expanded.replace(0.0, np.nan)
        expanded = expanded.replace([np.inf, -np.inf], np.nan)
        long = expanded.stack(future_stack=True).rename("value").reset_index()
        long.columns = ["trade_date", "instrument_id", "value"]
        return _restrict_to_listed(long.dropna(subset=["value"]), context)


class DividendFundingFactor(BaseFactor):
    """Payout or cash-flow coverage using yield times current market cap."""

    version = "1"
    inputs = (
        "daily_valuation.dividend_yield", "daily_valuation.market_cap",
        "fundamental_metrics",
    )

    _METRICS = {
        "payout": ("dividend_payout_ratio_ttm", "net_profit", True),
        "ocf_coverage": (
            "dividend_ocf_coverage_ttm", "operating_cash_flow", False,
        ),
        "fcf_coverage": (
            "dividend_fcf_coverage_ttm", "free_cash_flow", False,
        ),
    }

    def __init__(self, metric: str):
        if metric not in self._METRICS:
            raise ValueError(f"unsupported dividend funding metric: {metric}")
        self.metric = metric
        self.name, self.fundamental_code, self.cash_over_metric = (
            self._METRICS[metric]
        )

    def definition(self) -> Mapping[str, Any]:
        return {
            **super().definition(), "metric": self.metric,
            "fundamental_code": self.fundamental_code,
            "cash_proxy": "dividend_yield_l12m_times_market_cap",
        }

    @staticmethod
    def _ratio(
        dividend_yield: pd.DataFrame,
        market_cap: pd.DataFrame,
        fundamental: pd.DataFrame,
        cash_over_metric: bool,
    ) -> pd.DataFrame:
        columns = dividend_yield.columns.union(
            market_cap.columns
        ).union(fundamental.columns)
        cash = (
            dividend_yield.reindex(columns=columns)
            * market_cap.reindex(columns=columns)
        )
        metric = fundamental.reindex(columns=columns)
        if cash_over_metric:
            metric = metric.where(metric > 0.0)
            result = cash / metric
        else:
            cash = cash.where(cash > 0.0)
            result = metric / cash
        return result.replace([np.inf, -np.inf], np.nan)

    def compute(self, context, dependencies):
        targets = [
            day for day in _trading_days(context)
            if context.start <= day <= context.end
        ]
        valuation = context.valuation(["dividend_yield", "market_cap"])
        dividend_yield = _pivot(valuation, "dividend_yield").reindex(targets)
        market_cap = _pivot(valuation, "market_cap").reindex(targets)
        fundamental = context.fundamental_matrix(
            self.fundamental_code, "ttm", targets
        )
        result = self._ratio(
            dividend_yield, market_cap, fundamental, self.cash_over_metric
        )
        result.index = pd.Index(targets, name="trade_date")
        long = result.stack(future_stack=True).rename("value").reset_index()
        long.columns = ["trade_date", "instrument_id", "value"]
        return _restrict_to_listed(long.dropna(subset=["value"]), context)


# ------------------------------------------------------------------------ beta


class SelfBetaFactor(BaseFactor):
    """对市值加权全 A 市场做指数加权 CAPM 回归得到的斜率。

    252 个日收益率、63 个交易日半衰期、带截距（通过对加权均值中心化实现）。
    收益率取自后复权收盘价——这是安全的：前后复权价相差一个**每股常数**，
    相邻价格之比会把它精确抵消。

    窗口内配对有效观测少于 `minimum_observations` 的标的产出 NaN，
    而不是用太少的数据硬拟一个 Beta——不稳定的 Beta 会污染它参与的横截面分位。

    **已知尾部特征**：`|beta| > 5` 的 674 行（0.005%）里 98.5% 是上市不满两个月
    的新股（上市到出现极端值中位 34 天）。因为 `minimum_observations=21` 配在
    252 日窗口上，21 个交易日就够出一个 Beta。这是从参照策略继承的门槛，
    对低波策略本身无害（排在最差端会被剔掉），但它**参与分位计算**。
    """

    name = "beta_252_hl63"
    version = "1"
    inputs = (
        "daily_post_adjusted.close_post",
        "daily_raw.paused",
        "daily_valuation.circulating_market_cap",
    )

    def __init__(
        self,
        window: int = 252,
        half_life: float = 63.0,
        minimum_observations: int = 21,
        workers: int = 1,
    ):
        self.window = int(window)
        self.half_life = float(half_life)
        self.minimum_observations = int(minimum_observations)
        self.workers = int(workers)
        # 多要一天：252 个收益率需要 253 个收盘价。
        self.lookback = self.window + 1

    def definition(self) -> Mapping[str, Any]:
        # 故意不含 `workers`：它只改速度不改结果，放进来会让并行度一变
        # 缓存就全部作废。
        return {
            **super().definition(),
            "window": self.window,
            "half_life": self.half_life,
            "minimum_observations": self.minimum_observations,
        }

    def compute(self, context, dependencies):
        panels = market_panels(context, self.lookback)
        returns, market = panels.returns()
        calendar = _trading_days(context)
        targets = [
            day for day in calendar
            if context.start <= day <= context.end
        ]
        codes = (
            list(context.instruments) if context.instruments is not None
            else list(returns.columns)
        )
        # 指数衰减权重：age=0 是窗口最新一天，权重 1；每过 half_life 天减半。
        # 这条权重向量与日期无关，所以在循环外算一次。
        age = np.arange(self.window - 1, -1, -1, dtype=float)
        decay = np.power(0.5, age / self.half_life)

        def one(day: date) -> tuple[date, dict[str, float]]:
            usable = [item for item in calendar if item <= day]
            # 历史不足则整天不产出（新上市初期、数据起点附近）。
            if len(usable) < self.window + 1:
                return day, {}
            # 取最后 253 个交易日再丢掉第一个：收益率的第一天没有前收，
            # 于是剩下正好 252 个可用收益率日。
            window = usable[-(self.window + 1):][1:]
            block = returns.reindex(window)
            index = market.reindex(window).to_numpy(dtype=float)
            per_day: dict[str, float] = {}
            for code in codes:
                if code not in block.columns:
                    continue
                stock = block[code].to_numpy(dtype=float)
                # 只用个股与市场**同时**有效的日子——每只股票的 mask 都不同。
                mask = np.isfinite(stock) & np.isfinite(index)
                if int(mask.sum()) < self.minimum_observations:
                    continue
                weight = decay[mask]
                x = index[mask]
                y = stock[mask]
                total = float(weight.sum())
                if not math.isfinite(total) or total <= 0:
                    continue
                # 加权一阶矩与二阶矩。
                sx = float(np.sum(weight * x))
                sy = float(np.sum(weight * y))
                sxx = float(np.sum(weight * x * x))
                sxy = float(np.sum(weight * x * y))
                # 对加权均值中心化，在代数上等价于「拟合一个截距」。
                # 展开式：beta = (Sxy − Sx·Sy/Sw) / (Sxx − Sx²/Sw)
                # （已与直接解正规方程对拍，相对误差 1e-16）。
                denominator = sxx - sx * sx / total
                # 分母 <= 0 意味着市场收益在有效子集上几乎无方差，
                # 此时斜率没有意义，产出缺失而不是一个巨大的数。
                if not math.isfinite(denominator) or denominator <= 0:
                    continue
                value = (sxy - sx * sy / total) / denominator
                if math.isfinite(value):
                    per_day[code] = value
            return day, per_day

        results = _map_dates(targets, one, self.workers)
        return _emit(dict(results), context)


# --------------------------------------------------------------- risk / vol


class RollingRiskFactor(BaseFactor):
    """Annualised PIT risk statistics from trailing daily stock returns."""

    version = "1"
    inputs = (
        "daily_post_adjusted.close_post",
        "daily_raw.paused",
    )

    def __init__(
        self,
        name: str,
        metric: str,
        window: int,
        minimum_observations: int,
        tail_days: int = 5,
        workers: int = 1,
    ):
        if metric not in {"total_volatility", "downside_volatility", "tail_loss"}:
            raise ValueError(f"unsupported rolling risk metric: {metric}")
        self.name = str(name)
        self.metric = str(metric)
        self.window = int(window)
        self.minimum_observations = int(minimum_observations)
        self.tail_days = int(tail_days)
        self.workers = int(workers)
        self.lookback = self.window + 1

    def definition(self) -> Mapping[str, Any]:
        return {
            **super().definition(),
            "metric": self.metric,
            "window": self.window,
            "minimum_observations": self.minimum_observations,
            "tail_days": self.tail_days,
            "annualisation": 252.0,
        }

    def compute(self, context, dependencies):
        returns, _ = market_panels(context, self.lookback).returns()
        calendar = _trading_days(context)
        targets = [
            day for day in calendar if context.start <= day <= context.end
        ]
        codes = (
            list(context.instruments) if context.instruments is not None
            else list(returns.columns)
        )

        def one(day: date) -> tuple[date, dict[str, float]]:
            usable = [item for item in calendar if item <= day]
            if len(usable) < self.window + 1:
                return day, {}
            window = usable[-(self.window + 1):][1:]
            block = returns.reindex(window)
            per_day: dict[str, float] = {}
            for code in codes:
                if code not in block.columns:
                    continue
                values = block[code].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values) < self.minimum_observations:
                    continue
                if self.metric == "total_volatility":
                    if len(values) < 2:
                        continue
                    value = float(np.std(values, ddof=1) * np.sqrt(252.0))
                elif self.metric == "downside_volatility":
                    downside = np.minimum(values, 0.0)
                    value = float(np.sqrt(np.mean(downside * downside) * 252.0))
                else:
                    if len(values) < self.tail_days:
                        continue
                    worst = np.partition(values, self.tail_days - 1)[:self.tail_days]
                    value = float(-np.mean(worst))
                if math.isfinite(value):
                    per_day[code] = value
            return day, per_day

        return _emit(dict(_map_dates(targets, one, self.workers)), context)


class VolatilityOfVolatilityFactor(BaseFactor):
    """Stability of short-horizon realised volatility.

    Compute a 20-day rolling standard deviation of daily returns, then take
    the coefficient of variation of that series over 252 trading days.  A low
    value means the stock's realised-risk regime is persistent; a high value
    identifies names whose apparently low average volatility is punctuated by
    abrupt volatility bursts.
    """

    name = "volatility_of_volatility_20_252"
    version = "1"
    inputs = RollingRiskFactor.inputs

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 252,
        short_minimum: int = 16,
        long_minimum: int = 202,
    ):
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.short_minimum = int(short_minimum)
        self.long_minimum = int(long_minimum)
        self.lookback = self.short_window + self.long_window

    def definition(self) -> Mapping[str, Any]:
        return {
            **super().definition(),
            "short_window": self.short_window,
            "long_window": self.long_window,
            "short_minimum": self.short_minimum,
            "long_minimum": self.long_minimum,
            "normalisation": "rolling_std_over_rolling_mean",
        }

    @staticmethod
    def coefficient(
        returns: pd.DataFrame,
        short_window: int,
        long_window: int,
        short_minimum: int,
        long_minimum: int,
    ) -> pd.DataFrame:
        short_vol = returns.rolling(
            short_window, min_periods=short_minimum,
        ).std(ddof=1)
        mean = short_vol.rolling(
            long_window, min_periods=long_minimum,
        ).mean()
        variation = short_vol.rolling(
            long_window, min_periods=long_minimum,
        ).std(ddof=1)
        result = variation / mean.where(mean > 0.0)
        return result.replace([np.inf, -np.inf], np.nan)

    def compute(self, context, dependencies):
        returns, _ = market_panels(context, self.lookback).returns()
        values = self.coefficient(
            returns,
            self.short_window,
            self.long_window,
            self.short_minimum,
            self.long_minimum,
        )
        targets = [
            day for day in _trading_days(context)
            if context.start <= day <= context.end
        ]
        values = values.reindex(targets)
        long = values.stack(future_stack=True).rename("value").reset_index()
        long.columns = ["trade_date", "instrument_id", "value"]
        return _restrict_to_listed(long.dropna(subset=["value"]), context)


class ResidualVolatilityFactor(BaseFactor):
    """EW-CAPM residual RMS using the exact market model of SelfBetaFactor."""

    name = "residual_volatility_252_hl63"
    version = "1"
    inputs = SelfBetaFactor.inputs

    def __init__(
        self,
        window: int = 252,
        half_life: float = 63.0,
        minimum_observations: int = 202,
        workers: int = 1,
    ):
        self.window = int(window)
        self.half_life = float(half_life)
        self.minimum_observations = int(minimum_observations)
        self.workers = int(workers)
        self.lookback = self.window + 1

    def definition(self) -> Mapping[str, Any]:
        return {
            **super().definition(),
            "window": self.window,
            "half_life": self.half_life,
            "minimum_observations": self.minimum_observations,
            "annualisation": 252.0,
        }

    @staticmethod
    def _annualised_residual_rms(
        market: np.ndarray,
        stock: np.ndarray,
        weight: np.ndarray,
    ) -> float | None:
        total = float(weight.sum())
        if not math.isfinite(total) or total <= 0:
            return None
        sx = float(np.sum(weight * market))
        sy = float(np.sum(weight * stock))
        sxx = float(np.sum(weight * market * market))
        sxy = float(np.sum(weight * market * stock))
        denominator = sxx - sx * sx / total
        if not math.isfinite(denominator) or denominator <= 0:
            return None
        beta = (sxy - sx * sy / total) / denominator
        alpha = (sy - beta * sx) / total
        residual = stock - alpha - beta * market
        value = float(np.sqrt(
            np.sum(weight * residual * residual) / total * 252.0
        ))
        return value if math.isfinite(value) else None

    def compute(self, context, dependencies):
        returns, market = market_panels(context, self.lookback).returns()
        calendar = _trading_days(context)
        targets = [
            day for day in calendar if context.start <= day <= context.end
        ]
        codes = (
            list(context.instruments) if context.instruments is not None
            else list(returns.columns)
        )
        age = np.arange(self.window - 1, -1, -1, dtype=float)
        decay = np.power(0.5, age / self.half_life)

        def one(day: date) -> tuple[date, dict[str, float]]:
            usable = [item for item in calendar if item <= day]
            if len(usable) < self.window + 1:
                return day, {}
            window = usable[-(self.window + 1):][1:]
            block = returns.reindex(window)
            index = market.reindex(window).to_numpy(dtype=float)
            per_day: dict[str, float] = {}
            for code in codes:
                if code not in block.columns:
                    continue
                stock = block[code].to_numpy(dtype=float)
                mask = np.isfinite(stock) & np.isfinite(index)
                if int(mask.sum()) < self.minimum_observations:
                    continue
                weight = decay[mask]
                x, y = index[mask], stock[mask]
                value = self._annualised_residual_rms(x, y, weight)
                if value is not None:
                    per_day[code] = value
            return day, per_day

        return _emit(dict(_map_dates(targets, one, self.workers)), context)


# -------------------------------------------------------------------- momentum


class RegressionMomentumFactor(BaseFactor):
    """跳过最近若干日之后，对之前一段对数价做加权回归得到的趋势（6-1 动量）。

    先剔掉最近 `skip` 个交易日以压制短期反转，再对之前 `window` 个交易日的
    对数收盘价做加权线性回归，把斜率年化成 `exp(slope * year) - 1`，
    最后乘上加权 R² —— 陡但毛刺多的路径分数低于平稳上行的路径。

    **两处权重细节必须与参照策略逐字一致**，改任何一处都会平移全部动量分位：
    `numpy.polyfit(x, y, 1, w=W)` 内部会把权重**平方**（等价于权重 W² 的
    加权 OLS，已对拍误差 1e-15），而下面的 R² 用的是**线性** W，
    并且以**未加权均值**中心化。这个不对称是原文如此，不是笔误。

    窗口内缺任何一个交易日的标的产出 NaN —— 填补缺口会造出一条更平滑的趋势、
    把 R² 抬高。

    **已知尾部特征**：动量 > 20 的有 21617 行（0.16%），最大 3272.84。
    溯源发现是**长期停牌**：某股 120 天窗口里前约 80 天全程停牌、收盘价冻结，
    复牌后跳空，于是「120 天全有效」被冻结价满足了。本因子**不屏蔽停牌日**
    （Beta 屏蔽），但参照策略取价用的正是 `skip_paused=False, fill_paused=True`,
    行为一致，属忠实复现而非偏差。使用方要知道：**长期停牌复牌的股票会拿到
    最高的动量分位**。
    """

    name = "momentum_6_1"
    version = "1"
    inputs = ("daily_post_adjusted.close_post",)

    def __init__(
        self,
        window: int = 120,
        skip: int = 20,
        annualisation: float = 250.0,
        workers: int = 1,
    ):
        self.window = int(window)
        self.skip = int(skip)
        self.annualisation = float(annualisation)
        self.workers = int(workers)
        self.lookback = self.window + self.skip

    def definition(self) -> Mapping[str, Any]:
        return {
            **super().definition(),
            "window": self.window,
            "skip": self.skip,
            "annualisation": self.annualisation,
        }

    def compute(self, context, dependencies):
        panels = market_panels(context, self.lookback)
        prices = panels.close
        calendar = _trading_days(context)
        targets = [
            day for day in calendar
            if context.start <= day <= context.end
        ]
        codes = (
            list(context.instruments) if context.instruments is not None
            else list(prices.columns)
        )
        span = self.window
        # 自变量是 0..119 的序号，权重从 1 线性升到 2（越近权重越大）。
        # 两者与日期、与标的都无关，循环外算一次。
        positions = np.arange(span, dtype=float)
        weights = np.linspace(1.0, 2.0, span)

        def one(day: date) -> tuple[date, dict[str, float]]:
            usable = [item for item in calendar if item <= day]
            if len(usable) < self.lookback:
                return day, {}
            # 取最后 140 个交易日，再取其中**最老的 120 个**，
            # 即丢掉最近的 20 个（skip）。
            window = usable[-self.lookback:][:span]
            block = prices.reindex(window)
            per_day: dict[str, float] = {}
            for code in codes:
                if code not in block.columns:
                    continue
                series = block[code].to_numpy(dtype=float)
                # 要求 120 天全部有值且为正——缺一天就整格产出 NaN。
                if not np.all(np.isfinite(series)) or np.any(series <= 0):
                    continue
                y = np.log(series)
                try:
                    # 注意：polyfit 会把 w 平方；这与下面 R² 用线性 w 是
                    # 有意的不对称，见类文档。
                    slope, intercept = np.polyfit(
                        positions, y, 1, w=weights
                    )
                except (TypeError, ValueError, np.linalg.LinAlgError):
                    continue
                # 对数斜率 → 年化收益。250 是参照策略用的年交易日数。
                annualised = math.exp(slope * self.annualisation) - 1.0
                fitted = slope * positions + intercept
                residual = float(np.sum(weights * (y - fitted) ** 2))
                # 中心化用的是 np.mean(y)，**未加权**均值——原文如此。
                total = float(np.sum(weights * (y - np.mean(y)) ** 2))
                r_squared = 1.0 - residual / total if total else 0.0
                # 年化趋势 × 拟合优度：既要涨得多，也要涨得稳。
                value = annualised * r_squared
                if math.isfinite(value):
                    per_day[code] = value
            return day, per_day

        results = _map_dates(targets, one, self.workers)
        return _emit(dict(results), context)


# --------------------------------------------------------------------- quality


class MetricFactor(BaseFactor):
    """把 `fundamental_metrics` 里的**一个**指标按日期重建成 PIT 序列。

    每一格是「在该日期能看到的值」：已公布的最新报告期。注意排序键是
    `(fiscal_period_end, available_at, metric_id)`，**报告期优先于可见日**——
    一份晚公告的旧期重述，会输给一份已经在册的更新报告期。

    这是个**参数化**因子：`name` 由调用方给，所以同一个类可以产出多个因子
    （见下面的 `net_profit_factor()` / `operating_cash_flow_factor()`）。
    `metric_code` 与 `basis` 都进了 `definition()`，缓存不会串。

    `basis` 的含义：`ytd` 是年初至今的累计流量（利润、现金流），
    `instant` 是时点存量（净资产、总资产）。取错口径不会报错，只会算错。
    """

    version = "1"
    lookback = 0

    def __init__(self, name: str, metric_code: str, basis: str):
        self.name = name
        self.metric_code = metric_code
        self.basis = basis
        self.inputs = (f"fundamental_metrics.{metric_code}.{basis}",)

    def definition(self) -> Mapping[str, Any]:
        return {
            **super().definition(),
            "metric_code": self.metric_code,
            "basis": self.basis,
        }

    def compute(self, context, dependencies):
        calendar = [
            day for day in _trading_days(context)
            if context.start <= day <= context.end
        ]
        # 走宽表入口：长表在全历史日频下是上亿行。
        wide = context.fundamental_matrix(
            self.metric_code, self.basis, calendar
        )
        if wide.empty:
            return pd.DataFrame(
                columns=["trade_date", "instrument_id", "value"]
            )
        long = wide.stack(future_stack=True).rename("value").reset_index()
        long.columns = ["trade_date", "instrument_id", "value"]
        # 财务值是阶梯函数，必须裁剪上市窗口，否则退市后一路填充下去。
        return _restrict_to_listed(long.dropna(subset=["value"]), context)


class RoeFactor(BaseFactor):
    """指定周期的净资产收益率：归母净利润(period) / 归母净资产(instant)。

    快照里的 `roe` 只有 TTM 口径，而按报告期的质量筛需要报告期口径，
    所以在这里自己除。净资产为 0 视为缺失，不做除零。

    **必须知道的符号陷阱**：所有者权益为负时，负÷负 = 正，资不抵债的公司会拿到
    极高的正 ROE（宝鹰股份净利 −7.44 亿 → ROE 121.14；雏鹰退 −11.38 亿 →
    105.75）。全历史有 204369 行（1.41%）、955 只标的属于「净利为负但 ROE 为正」。
    数学没错，但**任何按 ROE 分位排序的质量筛都会把资不抵债的公司排在最前**。
    消费方必须自己加净利润硬门（参照策略正是这么做的）。
    """

    version = "1"
    lookback = 0

    def __init__(self, period: str = "ytd"):
        if period not in QUALITY_PERIODS:
            raise ValueError(
                f"unsupported quality period: {period!r}; "
                f"expected one of {QUALITY_PERIODS}"
            )
        self.period = period
        self.name = f"roe_{period}"
        self.inputs = (
            f"fundamental_metrics.net_profit_parent.{period}",
            "fundamental_metrics.equity_parent.instant",
        )

    def definition(self) -> Mapping[str, Any]:
        return {**super().definition(), "period": self.period}

    def compute(self, context, dependencies):
        calendar = [
            day for day in _trading_days(context)
            if context.start <= day <= context.end
        ]
        profit = context.fundamental_matrix(
            "net_profit_parent", self.period, calendar
        )
        equity = context.fundamental_matrix(
            "equity_parent", "instant", calendar
        )
        if profit.empty or equity.empty:
            return pd.DataFrame(
                columns=["trade_date", "instrument_id", "value"]
            )
        # 两个矩阵的标的集合可能不同，先对齐到并集再相除，
        # 缺一边的格子自然成为 NaN。
        columns = profit.columns.union(equity.columns)
        profit = profit.reindex(columns=columns)
        equity = equity.reindex(columns=columns).replace(0.0, np.nan)
        ratio = (profit / equity).replace([np.inf, -np.inf], np.nan)
        long = ratio.stack(future_stack=True).rename("value").reset_index()
        long.columns = ["trade_date", "instrument_id", "value"]
        return _restrict_to_listed(long.dropna(subset=["value"]), context)


class AssetGrowthFactor(BaseFactor):
    """PIT one-year growth in total assets.

    For each observation date, compare the latest balance-sheet total assets
    visible on that date with the latest value visible on the same calendar
    date one year earlier.  Using the earlier *information set* rather than a
    shifted daily panel keeps every output causal and avoids assuming that two
    firms published the same fiscal quarter on the same day.

    The raw factor is growth (larger means faster expansion).  Strategy users
    that seek conservative investment should reverse-rank it, i.e. smaller
    raw growth receives the higher score.
    """

    name = "asset_growth_1y"
    version = "1"
    lookback = 0
    inputs = ("fundamental_metrics.total_assets.instant",)

    @staticmethod
    def _prior_year(day: date) -> date:
        try:
            return day.replace(year=day.year - 1)
        except ValueError:  # February 29 -> February 28.
            return day.replace(year=day.year - 1, day=28)

    def compute(self, context, dependencies):
        targets = [
            day for day in _trading_days(context)
            if context.start <= day <= context.end
        ]
        if not targets:
            return pd.DataFrame(
                columns=["trade_date", "instrument_id", "value"]
            )
        priors = [self._prior_year(day) for day in targets]
        requested = sorted(set(targets + priors))
        assets = context.fundamental_matrix(
            "total_assets", "instant", requested
        )
        if assets.empty:
            return pd.DataFrame(
                columns=["trade_date", "instrument_id", "value"]
            )
        current = assets.reindex(targets)
        prior = assets.reindex(priors).copy()
        prior.index = pd.Index(targets, name="trade_date")
        columns = current.columns.union(prior.columns)
        current = current.reindex(columns=columns)
        prior = prior.reindex(columns=columns)
        valid = (current > 0.0) & (prior > 0.0)
        growth = (current / prior - 1.0).where(valid)
        growth = growth.replace([np.inf, -np.inf], np.nan)
        long = growth.stack(future_stack=True).rename("value").reset_index()
        long.columns = ["trade_date", "instrument_id", "value"]
        return _restrict_to_listed(long.dropna(subset=["value"]), context)


QUALITY_PERIODS: tuple[str, ...] = ("ytd", "single_quarter", "ttm")


def total_volatility_factor(
    window: int = 120, workers: int = 1,
) -> RollingRiskFactor:
    if window not in (120, 252):
        raise ValueError("total volatility window must be 120 or 252")
    minimum = 96 if window == 120 else 202
    return RollingRiskFactor(
        f"total_volatility_{window}", "total_volatility",
        window, minimum, workers=workers,
    )


def downside_volatility_factor(workers: int = 1) -> RollingRiskFactor:
    return RollingRiskFactor(
        "downside_volatility_252", "downside_volatility",
        252, 202, workers=workers,
    )


def worst5_loss_factor(workers: int = 1) -> RollingRiskFactor:
    return RollingRiskFactor(
        "worst5_loss_252", "tail_loss", 252, 202,
        tail_days=5, workers=workers,
    )


def dividend_yield_median_factor() -> DividendYieldHistoryFactor:
    return DividendYieldHistoryFactor("median")


def dividend_cash_growth_factor() -> DividendCredibilityFactor:
    return DividendCredibilityFactor("cash_log_growth")


def market_value_growth_factor() -> DividendCredibilityFactor:
    return DividendCredibilityFactor("market_value_log_growth")


def dividend_yield_change_factor() -> DividendCredibilityFactor:
    return DividendCredibilityFactor("yield_log_change")


def earnings_yield_factor() -> ValuationMultipleFactor:
    return ValuationMultipleFactor("earnings_yield")


def pb_ratio_factor() -> ValuationMultipleFactor:
    return ValuationMultipleFactor("pb")


def dividend_yield_anomaly_factor() -> DividendYieldHistoryFactor:
    return DividendYieldHistoryFactor("anomaly")


def dividend_payout_factor() -> DividendFundingFactor:
    return DividendFundingFactor("payout")


def dividend_ocf_coverage_factor() -> DividendFundingFactor:
    return DividendFundingFactor("ocf_coverage")


def dividend_fcf_coverage_factor() -> DividendFundingFactor:
    return DividendFundingFactor("fcf_coverage")


def net_profit_ttm_yoy_factor() -> MetricFactor:
    # Parent-company profit is the stable YoY series produced by the financial
    # normaliser; expose the strategy-facing name requested by the contract.
    return MetricFactor(
        "net_profit_ttm_yoy", "net_profit_parent_yoy", "ttm"
    )


def operating_cash_flow_ttm_yoy_factor() -> MetricFactor:
    return MetricFactor(
        "operating_cash_flow_ttm_yoy", "operating_cash_flow_yoy", "ttm"
    )


def revenue_ttm_factor() -> MetricFactor:
    return MetricFactor("revenue_ttm", "revenue", "ttm")


def revenue_ttm_yoy_factor() -> MetricFactor:
    return MetricFactor("revenue_ttm_yoy", "revenue_yoy", "ttm")


def roa_ttm_factor() -> MetricFactor:
    return MetricFactor("roa_ttm", "roa", "ttm")


def net_profit_factor(period: str = "ytd") -> MetricFactor:
    """指定周期的净利润；每个周期有独立因子名与缓存目录。"""
    if period not in QUALITY_PERIODS:
        raise ValueError(f"unsupported quality period: {period!r}")
    return MetricFactor(f"net_profit_{period}", "net_profit", period)


def operating_cash_flow_factor(period: str = "ytd") -> MetricFactor:
    """指定周期的经营现金流；每个周期有独立因子名与缓存目录。"""
    if period not in QUALITY_PERIODS:
        raise ValueError(f"unsupported quality period: {period!r}")
    return MetricFactor(
        f"operating_cash_flow_{period}", "operating_cash_flow", period
    )


def cn_equity_factors(workers: int = 1) -> tuple[BaseFactor, ...]:
    """本模块发布的通用因子，按构建顺序返回。

    构建面板与消费面板**都应该调这个函数**，这样两边的因子实例参数一致、
    缓存键一致。手工 new 一个 `SelfBetaFactor(window=250)` 就会另开一份缓存。
    """
    return (
        DividendYieldFactor(),
        dividend_cash_growth_factor(),
        market_value_growth_factor(),
        dividend_yield_change_factor(),
        earnings_yield_factor(),
        pb_ratio_factor(),
        SelfBetaFactor(workers=workers),
        total_volatility_factor(120, workers),
        total_volatility_factor(252, workers),
        ResidualVolatilityFactor(workers=workers),
        downside_volatility_factor(workers),
        worst5_loss_factor(workers),
        VolatilityOfVolatilityFactor(),
        DividendContinuityFactor(),
        dividend_yield_median_factor(),
        dividend_yield_anomaly_factor(),
        dividend_payout_factor(),
        dividend_ocf_coverage_factor(),
        dividend_fcf_coverage_factor(),
        net_profit_ttm_yoy_factor(),
        operating_cash_flow_ttm_yoy_factor(),
        revenue_ttm_factor(),
        revenue_ttm_yoy_factor(),
        roa_ttm_factor(),
        AssetGrowthFactor(),
        RegressionMomentumFactor(workers=workers),
        *(net_profit_factor(period) for period in QUALITY_PERIODS),
        *(operating_cash_flow_factor(period) for period in QUALITY_PERIODS),
        *(RoeFactor(period) for period in QUALITY_PERIODS),
    )


def cn_equity_factor_catalog(workers: int = 1) -> Mapping[str, BaseFactor]:
    """按完整因子名返回只读目录；不同周期绝不共享名称。"""
    from types import MappingProxyType

    factors = cn_equity_factors(workers)
    catalog = {factor.name: factor for factor in factors}
    if len(catalog) != len(factors):
        raise ValueError("cn equity factor catalog contains duplicate names")
    return MappingProxyType(catalog)
