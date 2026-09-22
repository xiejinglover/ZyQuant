# ZyQuant 研究运行 SOP

本文是新策略的现行运行规范。历史实验报告和旧策略专用脚本保留其
原始口径，不因本 SOP 自动改写。

## 一、三个边界

- 缓存位置：`.zyquant/cache/factors`，始终相对 `--project-root`。
- 标准因子 cutoff：`2026-07-24`，用于统一 cache identity。
- 正式基线回测结束日：`2026-06-01`，可从更宽缓存切片读取。

cutoff 不是回测结束日。它是数据的总可见性上限，也是因子缓存身份的
一部分。因子必须通过窗口独立性测试，保证宽缓存中的后续日期不会改变
历史行的值。

## 二、标准配置

```yaml
data:
  root: /data/zzh/ZyQuant/data
  dataset_id: <published-snapshot>
  start_date: 2025-12-10
  end_date: 2026-06-01
  cutoff: 2026-07-24
  verify_hashes: true
factor:
  cache_root: .zyquant/cache/factors
  cache_policy: require
output_root: /data/zzh/ZyQuant/runs
experiment_database: /data/zzh/ZyQuant/runs/experiments.sqlite
```

`compute` 只用于明确的开发或缓存构建任务。正式回测不得使用它在运行中
静默补算。

## 三、执行顺序

```bash
test "$(hostname)" = "E5"
test "$(pwd -P)" = "/data/zzh/ZyQuant"

/data/zzh/envs/zyquant-2.0/bin/zyq data validate \
  --root /data/zzh/ZyQuant/data --dataset-id <published-snapshot>

/data/zzh/envs/zyquant-2.0/bin/zyq factors prepare \
  --project-root /data/zzh/ZyQuant --config <config.yaml>

/data/zzh/envs/zyquant-2.0/bin/zyq factors verify \
  --project-root /data/zzh/ZyQuant --config <config.yaml>

/data/zzh/envs/zyquant-2.0/bin/zyq config validate --config <config.yaml>

/data/zzh/envs/zyquant-2.0/bin/zyq backtest run \
  --project-root /data/zzh/ZyQuant --config <config.yaml>
```

`prepare` 在固定缓存目录中构建缺失项，并在
`<output_root>/factors/<manifest-id>/manifest.json` 发布审计记录。`verify` 只读：
缺失、损坏或 identity 不一致均直接失败，不创建缓存和运行目录。

## 四、新旧策略边界

新策略通过 `factor_requirements()` 纯声明因子集，使 `prepare` 和 `verify`
使用同一份需求。旧策略不自动迁移：继续使用原配置、原 cutoff 和专用
预热脚本。对旧策略调用统一 CLI 时，命令会列出未声明需求的策略 ID，
不会猜测或执行 `prepare_run()` 来反推因子。

## 五、验收

- `verify` 返回的所有因子必须是缓存命中。
- manifest 中的数据指纹、cutoff、`instruments` 和 cache key 必须与正式运行一致。
- 正式回测产物必须原子发布，SQLite 中状态为 `succeeded`。
- 同一配置复跑应命中已发布 run，不得产生新因子缓存。
