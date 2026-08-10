"""因子层的地基：契约、上下文与基类。

本文件定义「什么算一个因子」，不包含任何具体因子的计算逻辑：

* `FactorDefinition` —— 因子的元数据形状（当前未被引擎直接使用，
  引擎读的是 `BaseFactor.definition()` 返回的字典）；
* `FactorResult` —— 一次计算的返回物：数据 + 缓存键 + 是否命中缓存 + 诊断；
* `FactorContext` —— **因子唯一被允许的取数入口**；
* `BaseFactor` —— 所有因子的抽象基类，子类只需实现 `compute()`。

设计上最重要的一条：因子**不直接碰快照**，只通过 `FactorContext` 取数。
因为 `FactorContext` 在每个取数方法里都把 `cutoff` 和 `instruments` 塞了进去，
这是「因子不可能看到未来数据」的结构性保证——不依赖因子作者的自觉。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd

from zyquant.data import DataSnapshot


@dataclass(frozen=True)
class FactorDefinition:
    """因子元数据的规范形状。

    `missing_policy="preserve"` 表示缺失值原样保留、不填充——因子层允许 NaN，
    到了信号层才致命（`signal.py` 遇 NaN 直接失败），所以消费方必须自己剔除。
    """

    name: str
    version: str
    inputs: tuple[str, ...] = ()
    lookback: int = 0
    output_frequency: str = "daily"
    missing_policy: str = "preserve"
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorResult:
    """一次 `FactorEngine.compute()` 的产出。

    `cache_key` 是可复现的凭证：同样的 (数据集, 因子定义, 因子源码, 依赖,
    cutoff, instruments, 区间) 一定得到同一个键。记录它就等于记录了
    「这批因子值是怎么来的」。

    `from_cache=True` 表示这次没有真算，是读的缓存（可能是精确命中，
    也可能是从一个更宽区间的缓存里切出来的）。
    """

    name: str
    frame: pd.DataFrame
    cache_key: str
    from_cache: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorView:
    """A sparse read-only view backed by one canonical factor cache entry."""

    name: str
    frame: pd.DataFrame
    cache_key: str
    cache_start: date
    cache_end: date
    requested_dates: tuple[date, ...] | None
    source_from_cache: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class FactorService(Protocol):
    """Stable factor access surface exposed to code-first strategies."""

    def compute(
        self,
        factor: "BaseFactor",
        snapshot: DataSnapshot,
        start: date,
        end: date,
        instruments: Sequence[str] | None = None,
        cutoff: date | None = None,
    ) -> FactorResult: ...

    def load_view(
        self,
        factor: "BaseFactor",
        snapshot: DataSnapshot,
        start: date,
        end: date,
        dates: Sequence[date] | None = None,
        instruments: Sequence[str] | None = None,
        cutoff: date | None = None,
    ) -> FactorView: ...


@dataclass(frozen=True)
class FactorContext:
    """因子的取数上下文——因子能看到的世界的全部边界。

    四个字段的分工必须分清，混淆是这一层最容易出的错：

    * `start` / `end`：**要产出哪些日期的因子值**（输出区间）；
    * `cutoff`：**总共允许读到哪一天的数据**（可见性上限，PIT 闸门）；
    * `instruments`：只算这些标的，`None` 表示全市场。

    `cutoff` 不等于 `end`。构建全历史面板时我们故意把 `cutoff` 设成数据集末日，
    这**不是前视**：`cutoff` 只决定「哪些行可以被读到」，而每一行的值仍然只由
    该行日期及其之前的回溯窗口决定。这个性质由具体因子的实现保证，并被
    `tests/test_cn_equity_factors.py` 的窗口独立性测试锁死（换窗口重算，
    共同日期的值必须逐位相同）。

    把 `cutoff` 固定成末日的好处是缓存不碎：`cutoff` 在缓存身份键里，
    每个调用方各传一个 `cutoff` 就等于每个调用方各算一遍。
    """

    snapshot: DataSnapshot
    start: date
    end: date
    cutoff: date
    instruments: tuple[str, ...] | None

    def history_start(self, bars: int) -> date:
        """回溯 `bars` 个交易日，给出读数据的起点。

        注意是按**交易日历**回溯，不是自然日——252 个交易日约等于一年，
        但按自然日退 365 天会少拿几十根 bar，滚动窗口就会缺观测。
        """
        calendar = self.snapshot.table("trade_calendar")
        days = sorted(set(calendar["trade_date"]))
        eligible = [day for day in days if day <= self.start]
        if not eligible:
            return self.start
        index = days.index(eligible[-1])
        return days[max(0, index - bars)]

    def post_adjusted(self, fields: Sequence[str], lookback: int = 0) -> pd.DataFrame:
        """后复权行情。**只能用于研究/因子计算，不得用于成交与估值。**

        这是项目的价格防火墙：后复权价里含有未来的复权因子信息，用它算收益率
        和趋势没问题（差一个每股常数，比值/对数差会抵消），但用它下单就等于
        用未来的价格成交。成交与账户会计必须走 `raw()`。
        """
        return self.snapshot.post_adjusted_bars(
            self.history_start(lookback), self.end, self.instruments, fields, self.cutoff
        )

    def raw(self, fields: Sequence[str], lookback: int = 0) -> pd.DataFrame:
        """不复权行情：停牌标记、涨跌停价、成交量等状态类字段从这里取。"""
        return self.snapshot.raw_bars(
            self.history_start(lookback), self.end, self.instruments, fields, self.cutoff
        )

    def fundamentals(
        self,
        metric_codes: Sequence[str],
        bases: Sequence[str] | None = None,
        dates: Sequence[date] | None = None,
    ) -> pd.DataFrame:
        """财务指标的**长表** as-of 面板：一行一个 (日期, 标的, 指标, 口径)。

        `dates` 默认只取区间两端，是历史遗留——重写 `metric_panel` 之前
        它按日期逐个全表扫描，日频传 4000 个日期要跑五个小时。现在已经不慢了，
        但日频因子仍应改用 `fundamental_matrix()`：长表在全历史日频下是上亿行。
        """
        return self.snapshot.financial(self.cutoff).metric_panel(
            dates or [self.start, self.end],
            self.instruments,
            metric_codes,
            bases,
        )

    def fundamental_matrix(
        self,
        metric_code: str,
        basis: str,
        dates: Sequence[date],
    ) -> pd.DataFrame:
        """单个财务指标的**宽表** as-of 面板：`(交易日 × 标的)`。

        这是日频财务因子应该用的入口，形状就是因子实际消费的形状。

        底层 `metric_matrix` 的 as-of 语义有个必须记住的坑：「最新」的排序键是
        `(fiscal_period_end, available_at, metric_id)`，**报告期优先于可见日**。
        所以一份晚公告的旧期重述，虽然 `available_at` 更晚，也不会盖掉一份
        已经在册的更新报告期。照直觉对 `available_at` 做 `merge_asof` 是错的，
        而且在正常数据上完全测不出来。
        """
        return self.snapshot.financial(self.cutoff).metric_matrix(
            dates, metric_code, basis, self.instruments,
        )

    def valuation(
        self,
        fields: Sequence[str],
        lookback: int = 0,
    ) -> pd.DataFrame:
        """日频估值面板：股息率、流通市值、PE/PB 等厂商已算好的派生量。"""
        return self.snapshot.financial(self.cutoff).valuation(
            self.history_start(lookback),
            self.end,
            self.instruments,
            fields,
        )

    def corporate_actions(self, lookback_calendar_days: int = 0) -> pd.DataFrame:
        """PIT-visible corporate actions needed by event-history factors.

        Corporate actions use natural-day event windows, unlike price factors'
        trading-day lookbacks.  The snapshot applies ``announced_at <= cutoff``;
        the factor must still enforce its own per-output-date visibility.
        """
        start = self.start - timedelta(days=int(lookback_calendar_days))
        return self.snapshot.table(
            "corporate_actions", start, self.end, self.instruments,
            cutoff=self.cutoff,
        )


class BaseFactor(ABC):
    """所有因子的基类。写一个新因子只需要：给这些类属性赋值 + 实现 `compute()`。

    五个类属性的作用：

    * `name`：因子名，同时是缓存目录名，全局唯一；
    * `version`：**语义版本**。改了算法含义就必须手动 +1；它进缓存身份键，
      所以改它会强制重算。注意：即使不改 `version`，改动类体里的**任何字符
      （含注释与 docstring）**也会让缓存失效——引擎把类源码文本一起哈希了
      （`engine.py:72`）。两者的分工是：`version` 表达「这是另一个因子了」，
      源码哈希兜底防止「代码变了但忘记 +1」；
    * `lookback`：需要多少个**交易日**的历史才能算出 `start` 当天的值。
      引擎据此把依赖因子的起点往前推；因子自己在 `compute()` 里也要用它
      向 `context.*(lookback=...)` 多取数据；
    * `dependencies`：依赖的其它因子，引擎会先算它们、把结果按名字传进来，
      并检测循环依赖与同名不同版本冲突；
    * `inputs`：声明用到的原始字段（如 `"daily_raw.paused"`）。目前是文档性质，
      引擎不校验，但它进 `definition()`，因此也进缓存键。

    `compute()` 的硬约束（`FactorEngine._validate` 会逐条检查，违反即报错）：
    返回 `trade_date` / `instrument_id` / `value` 三列，**每个
    (日期, 标的) 至多一行**，不得越出 `[start, end]`，不得返回没被请求的标的，
    数值必须有限（NaN 允许，inf 不允许）。
    """

    name: str
    version: str = "1"
    lookback: int = 0
    dependencies: tuple["BaseFactor", ...] = ()
    inputs: tuple[str, ...] = ()

    def definition(self) -> Mapping[str, Any]:
        """进缓存身份键的因子定义。

        子类若有参数（窗口长度、半衰期等），**必须**在这里补上，否则改参数
        不会换缓存键，会读到用旧参数算出来的值。反过来，纯性能参数
        （如 `workers`）**不要**放进来，否则并行度一变缓存就全部作废。
        """
        return {
            "class": f"{type(self).__module__}:{type(self).__qualname__}",
            "name": self.name,
            "version": self.version,
            "lookback": self.lookback,
            "inputs": self.inputs,
            "dependencies": [
                {"name": item.name, "version": item.version}
                for item in self.dependencies
            ],
        }

    @abstractmethod
    def compute(
        self,
        context: FactorContext,
        dependencies: Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Return trade_date, instrument_id and value columns."""
