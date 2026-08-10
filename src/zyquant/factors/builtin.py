"""内置示例因子：最小可用的几个，主要作为写新因子的模板。

这四个都很简单，真实的 A 股因子在 `cn_equity.py`。看它们主要是为了记住
`compute()` 的标准骨架：

    取数（多取 lookback 天） → 逐标的分组算 → 裁回 [start, end] → 返回三列

其中「多取再裁回」这一步是必须的：滚动窗口在 `start` 当天就需要之前的历史，
但产出**不允许**越出 `[start, end]`（`FactorEngine._validate` 会拦）。
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from .base import BaseFactor, FactorContext


class ReturnFactor(BaseFactor):
    """`window` 个交易日的简单收益率。

    `name` 是在 `__init__` 里按参数拼出来的，所以 `ReturnFactor(5)` 与
    `ReturnFactor(20)` 是两个不同名的因子，缓存目录也分开。
    """

    def __init__(self, window: int = 1):
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self.lookback = window
        self.name = f"return_{window}d"

    def definition(self):
        # 参数进定义 → 进缓存键。漏了这一步，改窗口不会换缓存。
        return {**super().definition(), "window": self.window}

    def compute(self, context: FactorContext, dependencies: Mapping[str, pd.DataFrame]):
        # 多取 window 天历史，否则 start 当天算不出来。
        frame = context.post_adjusted(["close_post"], self.window)
        # 必须按标的分组，否则 pct_change 会跨股票错位相减。
        frame["value"] = frame.groupby("instrument_id")["close_post"].pct_change(self.window)
        return frame.loc[
            frame["trade_date"].between(context.start, context.end),
            ["trade_date", "instrument_id", "value"],
        ].reset_index(drop=True)


class MomentumFactor(BaseFactor):
    """简单动量：当日收盘 / `window` 日前收盘 − 1。

    与 `cn_equity.RegressionMomentumFactor` 完全是两回事：这里是两点式，
    那里是跳期 + 加权回归 + R² 缩放。
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.lookback = window
        self.name = f"momentum_{window}d"

    def definition(self):
        return {**super().definition(), "window": self.window}

    def compute(self, context, dependencies):
        frame = context.post_adjusted(["close_post"], self.window)
        first = frame.groupby("instrument_id")["close_post"].shift(self.window)
        frame["value"] = frame["close_post"] / first - 1.0
        return frame.loc[
            frame["trade_date"].between(context.start, context.end),
            ["trade_date", "instrument_id", "value"],
        ].reset_index(drop=True)


class RollingAmountFactor(BaseFactor):
    """`window` 日成交额中位数。典型用途是流动性过滤。

    取的是 `context.raw`（不复权）——成交额是金额，不该复权。
    `min_periods=window` 意味着历史不足时产出 NaN 而不是用少量样本凑一个值。
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.lookback = window
        self.name = f"median_amount_{window}d"

    def definition(self):
        return {**super().definition(), "window": self.window}

    def compute(self, context, dependencies):
        frame = context.raw(["amount"], self.window)
        frame["value"] = frame.groupby("instrument_id")["amount"].transform(
            lambda values: values.rolling(self.window, min_periods=self.window).median()
        )
        return frame.loc[
            frame["trade_date"].between(context.start, context.end),
            ["trade_date", "instrument_id", "value"],
        ].reset_index(drop=True)


class CompositeFactor(BaseFactor):
    """多个因子的**原始值**加权求和。这是「依赖其它因子」的写法示例。

    `dependencies` 一填，引擎就会先把它们算完（或取缓存），再把结果按因子名
    塞进 `compute()` 的 `dependencies` 参数里。`lookback` 取依赖的最大值。

    **注意它的局限**：只做原始值加权求和，**不做标准化、不做排名**。
    量纲不同的因子（股息率 0.03 与净利润 1e8）直接相加是没有意义的。
    红利低波策略需要的是多级横截面分位 + 硬门 + 剔尾，装不进这个类，
    所以那部分逻辑留在策略层，没有用 `CompositeFactor`。
    """

    def __init__(self, name: str, factors: Mapping[BaseFactor, float]):
        self.name = name
        self.factors = tuple(factors.items())
        self.dependencies = tuple(factors)
        self.lookback = max((item.lookback for item in factors), default=0)

    def definition(self):
        return {
            **super().definition(),
            "weights": [(factor.name, weight) for factor, weight in self.factors],
        }

    def compute(self, context, dependencies):
        merged = None
        for factor, weight in self.factors:
            part = dependencies[factor.name].rename(columns={"value": factor.name})
            # inner join：任一因子缺值的格子整个丢掉，不做部分加权。
            merged = part if merged is None else merged.merge(
                part, on=["trade_date", "instrument_id"], how="inner"
            )
        if merged is None:
            return pd.DataFrame(columns=["trade_date", "instrument_id", "value"])
        merged["value"] = sum(merged[factor.name] * weight for factor, weight in self.factors)
        return merged[["trade_date", "instrument_id", "value"]]
