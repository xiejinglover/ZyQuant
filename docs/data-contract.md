# ZyQuant 标准数据契约

## 必需表

每个快照必须包含 `instruments`、`trade_calendar`、`daily_raw`、`daily_post_adjusted`、`corporate_actions`、`universe_membership`、`industry_membership` 和 `market_rules`。

财务能力为可选扩展，启用时必须同时包含 `financial_reports`、
`financial_facts`、`fundamental_metrics`、`daily_valuation` 和
`share_capital`，并在 manifest 的 `capabilities.financials` 中声明覆盖范围。
普通行情快照无需生成空财务表。

完整字段类型、空值、枚举和单位以[逐字段数据字典](data-dictionary.md)及
`zyquant.data.ARROW_SCHEMAS` 为准。发布时按同一份机器定义执行安全类型转换；
字段缺失、未知字段、无效枚举和有损转换均拒绝发布。

## 行情主键

原始和后复权行情的主键均为 `(trade_date, instrument_id)`，两张表的主键集合必须完全相同。

`daily_raw` 保存不复权开高低收、昨收、成交量、成交额、停牌和涨跌停价。成交、估值和账户会计只能使用此表。

`daily_post_adjusted` 保存后复权开高低收、昨收、复权因子、因子来源和算法版本。因子、标签、训练和信号直接读取此表，运行阶段不得重新计算。

## 复权口径

每个证券最早有效行情日的后复权因子归一为 1。现金分红、送转、拆分和合并在除权日改变因子。所有 OHLC 使用相同因子，计算使用双精度且不提前舍入。

供应商因子存在时必须标准化并与公司行动交叉校验；冲突时禁止发布。

## 公司行动

公司行动至少包含事件 ID、证券、类型、登记日、除权日、到账日、现金金额、份额比例、状态和公告时间。

v1 支持现金分红、送转、拆分和合并。JQData 股票事件来自 `STK_XR_XD`，
ETF 分红与拆并来自 `FUND_DIVIDEND`。无法可靠标准化的复杂事件不能静默近似。
聚宽的现金分红、送股和转增比例分别从 `bonus_ratio_rmb`、
`dividend_ratio` 和 `transfer_ratio` 标准化；finance 查询按递增 `id`
分页，不允许将单次 5000 行上限当作完整结果。

JQData 请求的 `vendor_factor_mode` 支持三种模式：

- `off` 不下载厂商复权因子，只使用公司行动推导因子。
- `validate` 下载并审计厂商因子，但使用公司行动因子发布；这是
  JQData 的默认值。
- `use` 仅在所有偏离都不超过 `vendor_factor_rtol` 时使用厂商因子，
  否则拒绝发布。

JQData 的默认容差为 `1e-3`。审计结果写入
`manifest.quality.vendor_factors`，包括比较行数、超限行数、超限比例和
相对偏离分位数。通用发布接口为了向后兼容，仍默认为 `use`
和 `5e-7`。

## 时间可见性

行情查询绑定 cutoff。历史股票池和行业表包含生效区间和 known-at 日期；研究运行只能读取当时已经可见的记录。

JQData 发布将价格范围与股票池范围分离。`price_scope=explicit` 只下载请求中的
证券行情，但 `universe_membership` 始终保存指数的完整历史成分，并把每日成分
无损压缩为连续有效区间。manifest 的 `lineage.coverage` 明确记录各表覆盖范围；
部分行情快照请求未覆盖证券时必须失败，不能静默缩小回测股票池。

`price_scope=universe` 才会下载区间内全部历史成分并集的行情和公司行动，并将
快照标记为完整股票池可回测。

正式财务报表以来源公告日为 `published_at`。来源没有日内公告时刻时，
`available_at` 为公告后的下一交易日；比较期记录作为届时才可见的历史修订保存。
利润表与现金流量表保留累计 YTD 原值，单季度和 TTM 由版本化算法物化。
每日估值仅在当日收盘后可用。

## 发布

快照通过临时目录构建，全部校验完成后原子发布。manifest 保存 schema、as-of、复权口径、算法版本、lineage、质量摘要、表级内容哈希、文件行数、大小、SHA-256 和整体 fingerprint。已发布数据集不可覆盖。

## 市场规则

`market_rules` 按交易所、资产类型和生效区间保存佣金、最低佣金、买卖税率、
过户费与币种。相同交易所和资产类型的规则区间不得重叠。执行配置中的费率仅是
显式情景覆盖；未覆盖时，回测必须读取执行日生效的规则。

## 查询视图

所有动态表查询必须显式传入 cutoff。`snapshot.research(cutoff)` 只能读取物化
后复权研究价格；`snapshot.trading(cutoff)` 只能读取原始成交和估值价格。
