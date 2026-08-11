# ml_ema20_momentum_v1（第一阶段）

该包实现 EMA20 上穿事件池与短线 cohort 交易系统，不包含多因子、训练集或模型训练。预测分数必须由调用方以 Parquet 提供。

## 预测协议

预测文件每行为一个信号日和股票，必须包含：

`signal_date`, `instrument_id`, `score`, `model_id`, `model_version`,
`feature_cutoff`, `train_cutoff`, `dataset_id`, `data_fingerprint`,
`feature_set_id`。

策略会校验数据集 ID、快照 fingerprint、时间截止、重复键和非有限分数。只在当日 EMA20 上穿池内按 `score desc, instrument_id asc` 选择。

## 交易口径

- T 日收盘后产生 cohort，T+1 开盘买入，T+2 收盘首次卖出。
- Top3 单股目标 15%；按现金和 1% 费用缓冲计算完整槽位。
- 收盘仍封涨停、停牌、跌停、容量不足或部分成交时保留余仓，下一交易日收盘继续尝试，没有最长持有期。
- 回测结束不强平，未关闭 cohort 在报告中披露。

## 服务器运行

先将符合协议且 fingerprint 与 v4 快照一致的预测写到 YAML 中的 `prediction_path`，再执行：

```bash
cd /data/zzh/ZyQuant
/data/zzh/envs/zyquant-2.0/bin/python -P \
  /data/zzh/ZyQuant/strategies/ml_ema20_momentum_v1/run_backtest.py \
  --config /data/zzh/ZyQuant/strategies/ml_ema20_momentum_v1/server_backtest.yaml
```

没有真实预测文件时会直接失败，不会伪造模型输出。

## 技术因子缓存

策略发布 49 个 `ml_ema20_*` 单股技术因子。历史滚动窗口先删除停牌 bar，
非停牌零成交量 bar 保留；EMA20 事件池和 20 日成交额过滤仍按交易日历计算。

服务器全历史预热：

```bash
/data/zzh/envs/zyquant-2.0/bin/python -P \
  /data/zzh/ZyQuant/strategies/ml_ema20_momentum_v1/build_factor_cache.py \
  --root /data/zzh/ZyQuant/data \
  --dataset hermes-cn-a-2010-20260724-v4 \
  --cache-root /data/zzh/ZyQuant/.zyquant/cache/factors \
  --output /data/zzh/ZyQuant/runs/ml_ema20_momentum_v1/factors
```

缓存构建只覆盖沪深普通 A 股，不输出北交所和 B 股；审计目录记录每个因子的
定义、cache key、覆盖率和耗时。横截面排名将在后续 EMA20 事件样本中计算。

## 3年训练 / 1年样本外数据集

```bash
/data/zzh/envs/zyquant-2.0/bin/python -P \
  /data/zzh/ZyQuant/strategies/ml_ema20_momentum_v1/build_dataset.py \
  --root /data/zzh/ZyQuant/data \
  --dataset hermes-cn-a-2010-20260724-v4 \
  --cache-root /data/zzh/ZyQuant/.zyquant/cache/factors \
  --factor-manifest /data/zzh/ZyQuant/runs/ml_ema20_momentum_v1/factors/cache_manifest.json \
  --output /data/zzh/ZyQuant/runs/ml_ema20_momentum_v1/datasets/rolling_3y_1y_v2_clean
```

标签为 `close_post[T+2] / open_post[T+1] - 1`。正式 labeled panel 和各年 fold
同时剔除无效标签以及任一模型特征为 NaN/±Inf 的行。
`excluded_samples.parquet` 只保留被排除的键和原因，不是可训练面板。

## XGBoost 3年训练 / 1年样本外滚动训练

```bash
/data/zzh/envs/zyquant-2.0/bin/python -P \
  /data/zzh/ZyQuant/strategies/ml_ema20_momentum_v1/train_xgb_walkforward.py \
  --dataset-root /data/zzh/ZyQuant/runs/ml_ema20_momentum_v1/datasets/rolling_3y_1y_v2_clean \
  --model-output /data/zzh/ZyQuant/runs/ml_ema20_momentum_v1/models/xgb_ranker_3y1y_v1 \
  --prediction-output /data/zzh/ZyQuant/runs/ml_ema20_momentum_v1/predictions/xgb_ranker_3y1y_v1.parquet \
  --device auto --first-test-year 2015 --last-test-year 2026
```

训练入口使用每日 EMA20 事件池内去极值与 percentile rank，以0–4档
relevance 训练 `XGBRanker(rank:ndcg)`。正式预测生成后，使用
`server_backtest_xgb_2015_2026.yaml` 运行2015–2026滚动回测。
