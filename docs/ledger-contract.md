# ZyQuant v1 标准账本契约

标准账本 schema 版本为 `1.1`。每条订单和现金事件都包含稳定事件 ID、日期、
账户或策略、证券以及上游事件引用。

主要产物：

- `sleeve_demands`、`demand_residuals`：策略需求和所有未满足原因；
- `internal_crosses`：袖套之间的无费用内部转移；
- `orders`、`fills`、`fill_allocations`：主账户订单、市场结果和袖套分配；
- `cashflows`：外部买卖、内部交叉、佣金、税费和分红到账；
- `positions`、`nav`：袖套每日持仓与 NAV；
- `master_positions`、`master_nav`：主账户每日持仓与 NAV；
- `corporate_actions`：登记权益、应收、份额调整和到账；
- `reconciliations`：逐事件及收盘守恒检查；
- `attribution`：价格、公司行动、现金、费用、滑点和执行残差。

`positions` 与 `master_positions` 在原有核心列后追加
`position_status`、`valuation_source`、`last_observed_date` 和
`stale_sessions`。旧读取方可继续只消费原列。退市处置使用确定性
`delisting_disposal` 公司行动记录；现金结算另写入 `cashflows`，
折价或归零损益计入归因的 `corporate_action` 分量。

每个执行事件和每日收盘后必须满足：

```text
袖套现金合计   = 主账户现金
袖套持仓合计   = 主账户持仓
袖套应收合计   = 主账户应收
袖套 NAV 合计  = 主账户 NAV
```

归因的 `pnl_component` 每日合计必须等于真实主账户 P&L；任何超过配置容差的
差异都会使运行失败。
