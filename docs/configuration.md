# ZyQuant 2.0 配置与运行

配置使用严格 schema：未知字段直接失败，解析后的配置冻结并生成确定性
fingerprint。YAML、CLI 与 Python 都通过 `ResolvedRunConfig` 解析，包含敏感字段名
的值在运行产物中自动脱敏。

## 最小运行流程

```bash
zyq data publish \
  --source canonical-directory \
  --root ./data \
  --request examples/canonical_directory.yaml \
  --dataset-id cn-demo-v1
zyq data validate --root ./data --dataset-id cn-demo-v1
zyq config validate --config examples/v1_config.yaml
zyq factors prepare --project-root . --config examples/v1_config.yaml
zyq factors verify --project-root . --config examples/v1_config.yaml
zyq backtest run --project-root . --config examples/v1_config.yaml
zyq runs list --database runs/experiments.sqlite
```

数据源必须显式指定，不存在默认供应商。`--request` 只传递给被选中的
连接器；密码没有 CLI 参数，也不会进入 manifest。

内置声明式 `pipeline` 策略支持外部 Parquet/CSV 信号、日/周/月/每 N 日日程、
历史股票池、TopK/Dropout/分数/风险平价构建器与核心组合约束。复杂策略应通过
本地 `module.path:factory` 工厂接入，不需要将策略打包进 wheel。

回测不生成 checkpoint，避免用未压缩 pickle 重复保存累计账本和策略
运行时对象。任务中断后需要从头重跑；只有成功完成的运行才会原子发布
标准账本和报告。

详细证券/行业归因默认关闭（`analysis.attribution: false`）。普通回测只生成
账本、基础绩效和报告；需要归因时，可在回测完成后基于已落盘账本单独生成，
无需重新执行策略和撮合。

## 退市持仓假设

`execution.delisting_policy` 控制持仓在回测日历首个不早于
`delist_date` 的交易日如何处置：

- `carry_last_mark`（默认）：保留股份、冻结交易，按最后因果收盘价估值；
- `write_off_zero`：注销全部 lot，不增加现金，将账面价值确认为退市损失；
- `cash_settle_last_close`：按最后收盘价与
  `delisting_recovery_rate`（`0` 至 `1`）合成现金结算。

三种模式都是回测假设，不代表退市末价可在现实中兑付。框架不接入
退市板块行情；正常停牌必须由 `paused: true` bar 表示，未退市持仓
完全缺 bar 仍视为数据错误。

因子缓存默认使用 `factor.cache_policy: require`：正式回测只读已发布
缓存，缺失时立即失败且不写文件。开发或预热任务要计算缓存时，
必须显式使用 `compute`。缓存逻辑根目录为 `.zyquant/cache/factors`，
相对 `--project-root` 解析，而不是相对当前 shell 目录。

权威生产快照的新因子缓存使用 `data.cutoff: 2026-07-24`、
`instruments=None` 和连续全历史区间。正式基线回测可以在 `2026-06-01`
结束，它只从更宽缓存中切片。策略可通过
`FactorEngine.load_view(..., dates=...)` 稀疏读取决策日；`dates` 只影响
返回视图，不产生新 cache key。统一 CLI 第一阶段面向实现
`factor_requirements()` 的新策略；旧策略继续使用原有预热脚本。
