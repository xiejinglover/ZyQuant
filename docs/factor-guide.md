# 因子层指南

因子层负责一件事：**把「每只股票每天一个数」这类可复用的计算，算一次、缓存住、
让所有策略共享**。它不做选股、不做组合构建、不做横截面比较。

本文讲：分层与职责 → 一个因子怎么被算出来 → 数据和产物在哪 → 怎么消费 →
怎么加新因子 → 已知陷阱。代码本身已加详细中文注释，本文只讲串起来的脉络。

| 文件 | 职责 |
|---|---|
| [base.py](../src/zyquant/factors/base.py) | 契约：`BaseFactor`、`FactorContext`、`FactorResult` |
| [engine.py](../src/zyquant/factors/engine.py) | 引擎：依赖解析、内容寻址缓存、产出校验 |
| [cn_equity.py](../src/zyquant/factors/cn_equity.py) | 六个真实 A 股因子 |
| [builtin.py](../src/zyquant/factors/builtin.py) | 四个极简示例因子，写新因子的模板 |
| [scripts/build_factor_panel.py](../scripts/build_factor_panel.py) | 全历史面板构建脚本 |
| [tests/test_cn_equity_factors.py](../tests/test_cn_equity_factors.py) | 17 个正确性测试 |

---

## 一、边界：什么能做成因子，什么不能

引擎强制每个因子返回**每个 `(trade_date, instrument_id)` 一个标量**
（`FactorEngine._validate`）。这条约束不是形式主义，它决定了什么能下沉：

**能做成因子** —— 只依赖单只股票自身历史（或全市场公共信息）的量：
收益率、波动、Beta、动量、财务指标、估值、流动性。

**不能做成因子** —— 依赖「当期股票池」的量：横截面分位、排名、
Z-score、行业中性化、剔尾。因为因子只知道 `context.instruments`，
**不知道策略当期筛出来的母池**。分位是相对量，母池换了含义就变了。

红利低波 V1.7 就是个典型：股息率 > 3% ∩ 有 Beta 的交集构成母池，
每个调仓日都不同。所以我们把**原始值**（股息率、Beta、动量、三个财务项）
下沉成六个因子，把**分位、硬门、剔尾、综合加权**留在策略层。

> 顺带说明为什么没用 `CompositeFactor`：它只做原始值加权求和，
> 无标准化无排名。股息率 0.03 和净利润 1e8 直接相加没有意义。

---

## 二、一个因子是怎么被算出来的

调用方永远只调 `FactorEngine.compute()`，因子自己不知道缓存的存在。

```python
from zyquant.data import ParquetDataProvider
from zyquant.factors import FactorEngine, cn_equity_factors

snapshot = ParquetDataProvider("data").open_snapshot("hermes-cn-a-2010-20260724-v3", False)
engine = FactorEngine(".zyquant/cache/factors")

result = engine.compute(
    factor=cn_equity_factors()[1],   # SelfBetaFactor
    snapshot=snapshot,
    start=date(2016, 1, 1),
    end=date(2016, 6, 30),
    instruments=None,                 # 全市场
    cutoff=date(2026, 7, 24),         # 数据集末日
)
result.frame        # trade_date / instrument_id / value
result.from_cache   # True 表示没真算
result.cache_key    # 可复现凭证
```

引擎内部按顺序做六件事：

**1. 依赖解析。** 递归先算 `factor.dependencies`，检测循环依赖和「同名不同版本」
冲突。依赖的起点会自动往前推 `lookback` 个交易日。

**2. 算缓存身份。** 这是整层的核心。`identity_key` 由七样东西决定：

| 成分 | 变了会怎样 |
|---|---|
| 数据集 fingerprint | 换数据集 → 全部重算（正确，数据不同了） |
| `factor.definition()` | 改窗口/半衰期等参数 → 重算 |
| **因子类的源码文本哈希** | **改类体里任何一个字符（含注释）→ 重算** |
| 各依赖的 cache_key | 上游变 → 下游重算 |
| `cutoff` | 每个调用方各传一个 → 缓存按 cutoff 碎成 N 份 |
| `instruments` | 传 `None` 与传子集是两份缓存，不互相命中 |
| 引擎版本 | 引擎行为变更时统一失效 |

区间 `[start, end]` **不在** `identity_key` 里，而是单独进 `cache_key`。
这个拆分正是「一次算全历史、之后任意窄区间免费」的机制。

**3. 两级命中。** 先按 `cache_key` 精确命中；不中则 `_find_broader` 扫同目录，
找一个 `identity_key` 相同且区间**覆盖**请求的缓存，直接切片返回。

**4. 加锁计算。** `O_CREAT|O_EXCL` 文件锁，多进程并发只算一遍；抢不到锁的一方
等成品出现后读缓存。锁文件超过 `lock_timeout` 视为持锁进程已死，抢占重试。

**5. 原子落盘。** 先写 `.tmp`，算 sha256，再 `os.replace` 换上去。
任何时刻旁观者要么看到旧的一对文件，要么看到新的一对，不会看到写了一半的
parquet。失败路径删临时文件——绝不留半成品，否则下次会被当成有效缓存读进去。

**6. 校验 + 诊断。** 三列必备、`(日期,标的)` 不重复、不越界、不返回未请求的标的、
数值有限（**NaN 允许、inf 不允许**）。诊断（行数、缺失率、极值）写进元数据。

### `cutoff` 为什么不是前视

`cutoff` 只限制**总共能读到哪些行**；每一行的值仍然只由该行日期及之前的
回溯窗口决定。所以用数据集末日做 `cutoff` 一次算完全历史是安全的。

这条性质由 `tests/test_cn_equity_factors.py` 的**窗口独立性测试**锁死：
在 `[start, end]` 上算完取日期 d 那一行，再单独在 `[start, d]` 上算取末行，
两者必须逐位相同。因子偷看未来这个断言必然失败。

---

## 三、数据从哪来，产物在哪

### 输入：只能走 `FactorContext`

因子**不直接碰快照**。`FactorContext` 的每个取数方法都已经把 `cutoff` 和
`instruments` 塞进去了——这是结构性的 PIT 保证，不依赖因子作者的自觉。

| 方法 | 取什么 | 注意 |
|---|---|---|
| `post_adjusted(fields, lookback)` | 后复权行情 | **只能用于研究**，不得用于成交与估值 |
| `raw(fields, lookback)` | 不复权行情 | 停牌、涨跌停、成交量等状态字段 |
| `valuation(fields, lookback)` | 日频估值 | 股息率、流通市值、PE/PB |
| `fundamental_matrix(code, basis, dates)` | 财务指标**宽表** | 日频因子用这个 |
| `fundamentals(codes, bases, dates)` | 财务指标长表 | 全历史日频下是上亿行，慎用 |

`lookback` 一律按**交易日历**回溯，不按自然日——252 个交易日约一年，
按自然日退 365 天会少几十根 bar，滚动窗口就缺观测。

### 产物一：因子缓存（真正的快照）

```
.zyquant/cache/factors/<数据集 fingerprint>/<因子名>/<cache_key>.parquet
                                                                      /<cache_key>.json
```

当前约 880 MB。**这个目录是累积的**：不同参数、不同代码版本的结果都堆在同一个
因子目录下。所以核对某批因子值要认 `cache_key`，或者直接用下面的导出宽表——
**绝不能按文件大小或修改时间猜**（这个坑踩过一次，取到了修复前的旧文件，
结论完全反了）。

### 产物二：导出宽表 + manifest（给人看的）

缓存元数据不记因子名/版本/instruments，也没有 CLI 可查，所以额外导出：

```
runs/factors/panel.parquet          # trade_date × instrument_id × 6 列
runs/factors/panel_manifest.json    # 因子名、版本、definition、cache_key、区间、覆盖率
runs/factors/build.log
```

构建命令（在项目根目录执行）：

```bash
python scripts/build_factor_panel.py --root data --dataset <dataset-id> --cache-root .zyquant/cache/factors --workers 8 --output runs/factors/panel.parquet
```

脚本里**三个参数是故意固定的，不能按调用方变**，否则缓存作废：
`instruments=None`、`cutoff=<数据集末日>`、`start/end=数据集完整区间`。

### 当前六个因子的规模（v3，2010-01-04 ~ 2026-07-24）

| 因子 | 非空行数 | 覆盖率 | 首个有值日 | 说明 |
|---|---|---|---|---|
| `dividend_yield_l12m` | 14,517,984 | 99.97% | 2010-01-04 | lookback 0 |
| `beta_252_hl63` | 13,969,706 | 96.20% | 2011-01-18 | 253 交易日预热 |
| `momentum_6_1` | 13,721,392 | 94.49% | 2010-07-30 | 140 交易日预热 |
| `net_profit_ytd` | 14,312,603 | 98.56% | 2010-01-04 | lookback 0 |
| `operating_cash_flow_ytd` | 14,312,617 | 98.56% | 2010-01-04 | lookback 0 |
| `roe_ytd` | 14,312,218 | 98.56% | 2010-01-04 | lookback 0 |

宽表 14,522,000 行。一次性构建成本约 15 分钟（32 进程）。

---

## 四、后面怎么使用

### 用法 A：计算或预热一个因子

```python
frame = engine.compute(
    SelfBetaFactor(), snapshot, start, end,
    instruments=None, cutoff=calendar[-1],   # 必须与构建时一致
).frame
```

开发环境默认 `cache_policy="compute"`：缓存不存在时会计算并发布。正式实验
使用 `cache_policy="require"`，缺失时直接报错，不允许边回测边补算。

策略消费已有缓存时使用稀疏视图：

```python
view = engine.load_view(
    SelfBetaFactor(), snapshot, start, end,
    dates=decision_dates,
    instruments=None,
    cutoff=calendar[-1],
)
```

`dates` 和 `instruments` 只过滤返回数据；权威缓存仍是全市场连续区间，
视图不会创建新 cache key。

### 用法 B：策略消费（现行做法）

`strategies/dividend_lowbeta/base/` 的做法值得照抄，取数与选股分成两个文件：

```python
from strategies.dividend_lowbeta.base.panels import load_factor_panels
from strategies.dividend_lowbeta.base.selection import FactorSelectionEngine

panels = load_factor_panels(snapshot, engine, start, end, cutoff, workers)
engine_ = FactorSelectionEngine(panels, params)
```

- `panels.py` 的 `load_factor_panels()` 只读取策略声明的因子观察日，再把稀疏
  长表 pivot 成 `(观察日 × 标的)`，并补上同日期的 `paused` 以及小型静态表；
- `selection.py` 的 `FactorSelectionEngine` 是**自包含**的完整选股实现：股票池、
  横截面分位、质量硬门、动量剔尾、综合加权全都写在这一个文件里，按漏斗顺序排列，
  没有继承也没有抽象钩子——新人从上往下读一遍就是完整的选股逻辑。

策略只保留这一套因子实现。`tests/test_cn_equity_factors.py` 用固定合成快照的
因子值和排名基线做回归保护，`tests/test_dividend_lowbeta_selection.py` 覆盖
股票池、质量门、动量剔尾和稳定排序等漏斗语义。

一个易错点：本策略有**两种取行语义**，用错不会报错但会改变选股结果。
`_row_on()` **不做前向填充**（Beta、动量、质量走它）——因子面板本身是逐日的，
某日无值即表示该标的当日不满足计算条件（如上市不足 253 日），应保持缺失；
而 `_row_asof()` **会前向填充**，股息率与停牌两处走它。这个不对称是历史行为，
两处调用点都写了「勿改」注释。

### 用法 C：直接读导出面板（离线分析、外部消费）

```python
panel = pd.read_parquet("runs/factors/panel.parquet")
```

该导出文件不是策略运行依赖，也不是第二套权威缓存。

一次拿到六列，适合做相关性、IC、覆盖率之类的批量分析。

---

## 五、怎么加一个新因子

### 步骤

**1. 判断它能不能做成因子。** 见第一节：依赖当期股票池的量不行。

**2. 选放哪。** 通用 A 股因子 → `cn_equity.py`。
策略专有 → `strategies/<策略>/factors.py`。示例/教学 → `builtin.py`。

**3. 写类。** 骨架：

```python
class MyFactor(BaseFactor):
    """一句话说清算什么，然后说清所有非显然的选择。"""

    name = "my_factor_20d"        # 全局唯一，同时是缓存目录名
    version = "1"                 # 语义版本，改算法含义就 +1
    inputs = ("daily_raw.amount",)  # 声明用到的原始字段（文档性质）

    def __init__(self, window: int = 20):
        self.window = int(window)
        self.lookback = window     # 算 start 当天需要多少交易日历史

    def definition(self):
        # 所有影响结果的参数都必须在这里，否则改参数不会换缓存键
        return {**super().definition(), "window": self.window}

    def compute(self, context, dependencies):
        frame = context.raw(["amount"], self.lookback)   # 多取 lookback 天
        frame["value"] = ...                             # 按标的分组算
        out = frame.loc[
            frame["trade_date"].between(context.start, context.end),
            ["trade_date", "instrument_id", "value"],
        ]
        return _restrict_to_listed(out, context)          # 财务类必须；价格类可省
```

**4. 参数放对位置。** 影响结果的参数（窗口、半衰期、口径）→ 必须进
`definition()`。纯性能参数（`workers`）→ **绝不能**进，否则并行度一变缓存全废。
`SelfBetaFactor` 里就是这么处理的。

**5. 需要并行就按日期分片。** 把逐日计算写成闭包 `one(day)`，交给 `_map_dates`。
**不要用线程**——内层是 Python 绑定，实测 4/8 线程是 0.83× / 0.77×（更慢）。

**6. 写三层测试**（照抄 `tests/test_cn_equity_factors.py`）：

- **固定基线**：在合成快照上锁定关键因子值与排名（`rel=1e-12`）；
- **窗口独立性**：换窄区间重算，共同日期必须相同——这是无泄漏的证据；
- **边界**：窗口中途上市、观测数正好在门槛上下、窗口内缺 1 天、
  全市场某日无有效数据。

> 夹具两条铁律：日期一律**相对生成**不写死；夹具本身必须过契约校验
> （比如 `fiscal_period_end < available_at` 是硬约束，构造不出来的状态
> 该测的是「契约拦住了它」）。

**7. 注册。** 加进 `cn_equity_factors()` 与 `factors/__init__.py` 的 `__all__`。
构建与消费都调 `cn_equity_factors()`，两边参数才一致。

**8. 重建面板。** 新因子不会影响已有因子的缓存（身份键互相独立），
所以可以只算新的：

```bash
python scripts/build_factor_panel.py --root data --dataset <dataset-id> --cache-root .zyquant/cache/factors --workers 8 --only my_factor_20d
```

### 修改一个**已有**因子时

先想清楚要哪种语义：

- **修 bug / 换更好的实现，旧值作废** → 直接改代码。源码哈希变化会自动让缓存
  失效，`version` 可以不动。**但整个面板要重算。**
- **算法含义变了，等于另一个因子** → `version` +1（或干脆换 `name`），
  这样两版可以并存对比。
- **只是加注释、改 docstring** → 也会让缓存失效（源码哈希包含注释）。
  这是设计上的取舍：宁可多算一次，也不要「代码变了但忘记 +1」导致读到旧值。
  本文档配套的注释改动就触发了一次全量重建，约 15 分钟。

---

## 六、已知陷阱

**因子层允许 NaN，信号层不允许。** 因子对未上市/数据不足的格子产出 NaN
是合法信息，但 `signal.py` 遇 NaN 直接失败。策略侧消费时必须自己剔除。

**财务因子必须裁上市窗口。** 财务值是阶梯函数，放任不管会把最后一期报表一路
前向填充下去。修复前有 754,706 行（4.6%）落在上市窗口外、977 个孤儿标的。
价格类因子天然不需要（退市了就没 bar）。`_restrict_to_listed()` 负责这件事。

**`_find_broader` 返回的诊断是宽窗口的**，不是切片后的。别拿它判断切片数据的质量。

**财务 as-of 的排序键是 `(fiscal_period_end, available_at, metric_id)`——
报告期优先于可见日。** 所以不能照直觉对 `available_at` 做 `merge_asof`：
一份晚公告的旧期重述，`available_at` 更晚但报告期更早，不该盖掉已在册的
更新报告期。写错在正常数据上完全测不出来。

**Beta 的分母是个股相关的。** 所有加权和都在「个股与市场同时有效」的子集上求，
每只股票的缺失日不同（实测一只股票缺 4 天，分母相对差 5.26e-03）。
这是当初决定**不做向量化改写**的主要原因：掩码逻辑写错时，
全有效的测试用例完全发现不了。

**动量不屏蔽停牌日，Beta 屏蔽。** 长期停牌的股票收盘价冻结，
「120 天全有效」会被冻结价满足，复牌跳空后动量可以高到 3272
（实测最大值）。这与参照策略一致（`skip_paused=False, fill_paused=True`），
属忠实复现，但使用方要知道：**长期停牌复牌的股票会拿到最高动量分位**，
而剔尾只剔低端。

**新股的 Beta 不可信。** `minimum_observations=21` 配在 252 日窗口上，
上市 21 个交易日就能出 Beta。`|beta| > 5` 的 674 行里 98.5% 是上市不满两个月的
新股。对低波策略无害（排最差端被剔掉），但**参与分位计算**。

**负所有者权益 → 正 ROE。** 资不抵债的公司 `net_profit / equity` 负÷负得正，
拿到极高正 ROE（实测最高 121.14，净利 −7.44 亿）。全历史 204,369 行（1.41%）、
955 只标的属于此类。任何按 ROE 分位排序的质量筛都会把它们排在最前，
消费方必须自己加净利润硬门。

**不要用 `corporate_actions` 手搓股息率。** 厂商的 `dividend_yield` 会**按后续
送转重述每股分红**，朴素求和不会（送 1.0 差 2 倍、送 3.0 差 4 倍）。
已做全市场对拍确认厂商值正确。

**不走 `FactorSignalGenerator`。** 它逐日 compute + 逐日股票池，
`cutoff` / `instruments` 都在身份键里，缓存完全失效；且它是零调用零测试、
配置驱动路径明确拒绝的代码。

更完整的历史问题清单见 [问题与踩坑记录](issue-log.md)。
