# ZyQuant v1.2 数据字典

机器定义位于 `zyquant.data.FIELD_SPECS` 和 `zyquant.data.ARROW_SCHEMAS`。
`必需=否` 的字段可以整列省略；存在时仍必须满足所列类型和约束。日期均为
无时区自然日 `date32`，来源更新时间为 UTC 时间戳。

## instruments

主键：`instrument_id`。

| 字段 | 类型 | 可空 | 必需 | 单位/枚举 | 含义 |
|---|---|---:|---:|---|---|
| instrument_id | string | 否 | 是 |  | 标准证券代码 |
| symbol | string | 否 | 是 |  | 交易所本地代码 |
| exchange | string | 否 | 是 | XSHG, XSHE, XBEI | 交易所 |
| asset_type | string | 否 | 是 | stock, etf | 资产类型 |
| list_date | date32 | 否 | 是 |  | 上市日 |
| delist_date | date32 | 是 | 是 |  | 退市日 |
| lot_size | int64 | 否 | 是 | 股，≥1 | 最小交易手数 |
| sell_delay_days | int64 | 否 | 是 | 交易日，≥0 | 买入后可卖等待期 |
| name | string | 是 | 否 |  | 证券名称 |
| currency | string | 是 | 否 | CNY | 计价币种 |

## trade_calendar

主键：`(exchange, trade_date)`。

| 字段 | 类型 | 可空 | 单位/枚举 | 含义 |
|---|---|---:|---|---|
| trade_date | date32 | 否 |  | 交易日 |
| exchange | string | 否 | XSHG, XSHE, XBEI | 交易所 |

## daily_raw

主键：`(trade_date, instrument_id)`。价格是不复权成交/会计口径；停牌日保留
记录并通过 `paused` 禁止成交。

| 字段 | 类型 | 可空 | 单位 | 含义 |
|---|---|---:|---|---|
| trade_date | date32 | 否 |  | 交易日 |
| instrument_id | string | 否 |  | 标准证券代码 |
| open, high, low, close | float64 | 否 | CNY/股 | 不复权 OHLC |
| pre_close | float64 | 否 | CNY/股 | 当日市场参考昨收 |
| volume | int64 | 否 | 股 | 成交量 |
| amount | float64 | 否 | CNY | 成交额 |
| paused | bool | 否 |  | 是否停牌 |
| limit_up, limit_down | float64 | 否 | CNY/股 | 当日涨跌停价 |

## daily_post_adjusted

主键与 `daily_raw` 完全一致。该表只能在快照发布时物化。

| 字段 | 类型 | 可空 | 单位/枚举 | 含义 |
|---|---|---:|---|---|
| trade_date | date32 | 否 |  | 交易日 |
| instrument_id | string | 否 |  | 标准证券代码 |
| open_post, high_post, low_post, close_post | float64 | 否 | CNY/股 | 后复权 OHLC |
| pre_close_post | float64 | 否 | CNY/股 | 后复权昨收 |
| adjustment_factor | float64 | 否 | 比率，>0 | 归一化复权因子 |
| factor_source | string | 否 | vendor, corporate_action | 因子来源 |
| adjustment_version | string | 否 |  | 复权算法版本 |

## corporate_actions

主键：`event_id`。每行只表达一种经济事件；同一次派息送转可以拆成多行。

| 字段 | 类型 | 可空 | 单位/枚举 | 含义 |
|---|---|---:|---|---|
| event_id | string | 否 |  | 稳定事件 ID |
| instrument_id | string | 否 |  | 标准证券代码 |
| event_type | string | 否 | cash_dividend, bonus, split, merge, rights_issue | 事件类型 |
| record_date | date32 | 是 |  | 登记日 |
| ex_date | date32 | 否 |  | 除权除息日 |
| pay_date | date32 | 是 |  | 到账日 |
| cash_per_share | float64 | 否 | CNY/股，≥0 | 每股税前现金 |
| share_ratio | float64 | 否 | 股/股，≥0 | 份额变化比例 |
| subscription_price | float64 | 是 | CNY/股，≥0 | 配股认购价 |
| status | string | 否 | active, cancelled | 事件状态 |
| announced_at | date32 | 否 |  | 可见的实施公告日 |

## universe_membership

主键：`(universe_id, instrument_id, effective_from)`，区间端点均包含。

| 字段 | 类型 | 可空 | 含义 |
|---|---|---:|---|
| universe_id | string | 否 | 股票池或指数 ID |
| instrument_id | string | 否 | 标准证券代码 |
| effective_from | date32 | 否 | 生效首日 |
| effective_to | date32 | 是 | 生效末日；空表示仍有效 |
| known_at | date32 | 否 | 研究者可见日期 |

## industry_membership

主键：`(classification, instrument_id, effective_from)`，区间端点均包含。

| 字段 | 类型 | 可空 | 含义 |
|---|---|---:|---|
| classification | string | 否 | 分类体系，如 sw_l1 |
| industry_id | string | 否 | 行业代码 |
| instrument_id | string | 否 | 标准证券代码 |
| effective_from | date32 | 否 | 生效首日 |
| effective_to | date32 | 是 | 生效末日 |
| known_at | date32 | 否 | 研究者可见日期 |

## market_rules

主键：`rule_id`。相同交易所、资产类型的生效区间不能重叠。

| 字段 | 类型 | 可空 | 必需 | 单位/枚举 | 含义 |
|---|---|---:|---:|---|---|
| rule_id | string | 否 | 是 |  | 稳定规则 ID |
| exchange | string | 否 | 是 | XSHG, XSHE | 交易所 |
| asset_type | string | 否 | 是 | stock, etf | 资产类型 |
| effective_from | date32 | 否 | 是 |  | 生效首日 |
| effective_to | date32 | 是 | 是 |  | 生效末日 |
| commission_bps | float64 | 否 | 是 | bps，≥0 | 佣金率 |
| minimum_commission | float64 | 否 | 是 | CNY，≥0 | 单笔最低佣金 |
| sell_tax_bps, buy_tax_bps | float64 | 否 | 是 | bps，≥0 | 买卖税率 |
| transfer_fee_bps | float64 | 否 | 是 | bps，≥0 | 过户费率 |
| currency | string | 否 | 是 | CNY | 币种 |
| source | string | 是 | 否 |  | 规则来源 |
| rule_version | string | 是 | 否 |  | 规则版本 |
| scenario | bool | 是 | 否 |  | 是否为情景假设 |

## 可选财务能力

以下五张表必须成组出现。财务金额统一为 CNY，股份统一为股，比例统一为小数。

### financial_reports

主键：`report_id`。

| 字段 | 类型 | 含义 |
|---|---|---|
| report_id | string | 稳定报告版本 ID |
| instrument_id | string | 标准证券代码 |
| statement_type | string | balance, income, cash_flow |
| fiscal_period_start/end | date32 | 事实所属期间 |
| filing_period_end | date32 | 本次披露报表期 |
| record_kind | string | current, comparative |
| published_at | date32 | 来源公告日 |
| available_at | date32 | 研究可用日，默认下一交易日 |
| revision_sequence | int64 | 同一事实期间的修订序号 |
| currency | string | CNY |
| source_report_type | string | 来源报表类型代码 |

### financial_facts

主键：`(report_id, item_code)`。

| 字段 | 类型 | 含义 |
|---|---|---|
| report_id, instrument_id | string | 报告及证券 |
| statement_type | string | 报表类型 |
| item_code | string | 版本化标准科目代码 |
| fiscal_period_start/end | date32 | 事实所属期间 |
| filing_period_end | date32 | 本次披露报表期 |
| available_at | date32 | 研究可用日 |
| value | float64 | 原始规范数值 |
| unit | string | CNY, CNY/share, shares, ratio |
| value_basis | string | instant, ytd, per_share, ratio |
| source_field | string | 供应商原字段 |

### fundamental_metrics

主键：`metric_id`。指标不按日向前填充，查询视图按每个 as-of 日期选择当时
已知的最新版本。

| 字段 | 类型 | 含义 |
|---|---|---|
| metric_code | string | 标准指标代码 |
| fiscal_period_end | date32 | 指标报告期 |
| basis | string | instant, ytd, single_quarter, ttm |
| value, unit | float64, string | 指标值及单位 |
| available_at | date32 | 指标可用日 |
| calculation_version | string | 指标算法版本 |
| source_report_ids | string | 有序来源报告 ID JSON |
| quality_status | string | complete, not_applicable |

### daily_valuation

主键：`(trade_date, instrument_id)`。包括 PE-TTM、PE-LYR、PB、PS-TTM、
两种 PCF、股息率、换手率，以及总/流通/自由流通/A 股股本和市值。
`available_at` 等于交易日，表示当日收盘后可用。

### share_capital

主键：`capital_event_id`。保存变动生效日、公告/可用日、变动原因、总股本、
非流通股、限售股、流通股及 A/B/H 股数量。

## 通用来源字段

所有源表允许以下可选字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| source_record_id | string | 供应商稳定记录 ID |
| source_batch_id | string | 确定性拉取批次 ID |
| source_updated_at | timestamp[us, UTC] | 供应商记录更新时间 |

凭据、访问令牌和密码禁止出现在任何来源字段或快照 lineage 中。
## Hermes 1.2 扩展

当前框架数据 schema 版本为 1.2。机器可校验定义以
`src/zyquant/data/contracts.py` 为准。

Hermes 全量快照覆盖基础 8 表及财务 5 表，并新增：

- `exchange` 支持 `XSHG`、`XSHE`、`XBEI`。
- `corporate_actions.event_type` 支持 `rights_issue`，配股价保存在
  `subscription_price`。
- `fundamental_metrics.quality_status` 支持 `source_missing`，与业务上的
  `not_applicable` 严格区分。
- `CN_ALL_A` 使用证券上市日至退市日的有效区间，因此任意交易日查询得到的
  股票池会自然变化，不使用当前成分倒填历史。
- `market_rules` 可用 `source=source_missing` 和空费率明确表达来源缺失；
  这种快照必须声明不可回测。

Hermes 的 `DECIMAL` 统一转换为 Arrow `float64`，日期为 `date32`，源更新时间
为 UTC 时间戳。每日估值的市值单位为 CNY、股本为股、股息率和换手率为小数。
三大报表保留所有非空数值科目，原始字段名保存在
`financial_facts.source_field`，公告后的下一交易日才可见。
