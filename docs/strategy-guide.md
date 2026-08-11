# 策略与组合开发约定

## 标准策略

标准策略由调仓日程、股票池、信号、组合构建器和约束组成。研究员可以替换任意组件，而不需要修改数据、撮合或账户模块。

策略描述目标持仓，不描述成交细节。交易手数、停牌、涨跌停、成交容量、费用和资金不足由回测层负责。

## 自定义目标策略

无法自然拆分的轮动或状态机策略应实现代码优先的 `Strategy`
协议，自由生成 `StrategyDecision`。策略可以使用 `prepare_run`
预计算因子面板。

策略最终输出 `TargetPortfolio`。引擎统一检查权重、标的、日期和状态，
再负责 T+1、停牌、涨跌停、费用、成交和账本。研究约束可以使用
`ConstraintEngine`，也可由策略自己产生合法目标。

本地策略通过模块工厂引用，例如
`strategies.my_strategy:create_strategy`，并由 `--project-root` 确定源码根目录。

## 状态

策略需要记忆时使用显式、可序列化状态。状态适合保存持仓进入日期、排名缓冲、冷却期和有限状态机，不适合保存行情矩阵、模型对象或外部连接。

## 跨时点短周期交易

普通策略继续返回单个 `TargetPortfolio`，其执行时点由全局
`ExecutionConfig.timing` 决定。需要在同一策略内混合开盘和收盘成交时，使用
`StrategyDecision.scheduled_targets` 返回一个或多个
`ScheduledTargetPortfolio`。

`session_offset` 从信号日按交易日计数，1 表示下一交易日；`cohort_id` 标识本次
信号建立的独立持仓批次。引擎对 cohort 目标只计算和交易该批次的持仓，因此旧批次
可以在收盘退出，而不影响同日开盘建立的新批次。股票的 T+1 可卖日期仍由市场规则
统一检查。

例如，T 日收盘产生信号，T+1 开盘买入、T+2 收盘卖出：

```python
cohort = context.signal_date.isoformat()
entry = ScheduledTargetPortfolio(
    strategy_id=context.strategy_id,
    signal_date=context.signal_date,
    session_offset=1,
    execution_phase="open",
    cohort_id=cohort,
    weights={"600000.XSHG": 0.5},
    cash_weight=0.5,
    universe_fingerprint="...",
    signal_fingerprint="entry-...",
    state_before_hash="...",
    state_after_hash="...",
)
exit_ = ScheduledTargetPortfolio(
    strategy_id=context.strategy_id,
    signal_date=context.signal_date,
    session_offset=2,
    execution_phase="close",
    cohort_id=cohort,
    weights={},
    cash_weight=1.0,
    universe_fingerprint="...",
    signal_fingerprint="exit-...",
    state_before_hash="...",
    state_after_hash="...",
)
return StrategyDecision(
    None, context.state, signals,
    scheduled_targets=(entry, exit_),
)
```

同一个 cohort 的全部执行腿只要有一个落到回测区间之外，就整组跳过，并写入
`scheduled_target_residuals`，避免回测末尾只建仓而没有退出。逐日
`position_lots` 和 `fill_allocations.cohort_id` 可用于追溯批次。

## 组合约束

硬约束默认失败，不静默放宽。无候选可以选择保持、现金、跳过或失败；数据和协议错误始终失败。

## 多策略

多个策略通过虚拟袖套独立核算。相向需求在主账户层净额，真实成交和费用再确定性分配。策略不能直接读取其他袖套的内部持仓或状态。

## 可复现性

策略使用框架提供的随机数生成器，并确保相同数据、参数、状态和随机种子产生相同目标。信号同分必须使用稳定排序规则。
