# Hermes 全量数据运行手册

Hermes 是全量 A 股快照的唯一来源。采集过程只使用只读事务、服务端游标和
启动时固定的 `UPDATE_TIME` 水位，不会向 Hermes 执行写操作。

## 凭据

凭据只能通过进程环境提供：

```bash
export HERMES_MYSQL_HOST=...
export HERMES_MYSQL_PORT=3306
export HERMES_MYSQL_USER=...
export HERMES_MYSQL_PASSWORD=...
export HERMES_MYSQL_DATABASE=hermes
```

状态库、日志、manifest 和 lineage 只保存主机、库名、用户哈希、查询哈希和
源数据水位，不保存用户名明文或密码。

## 全量采集与恢复

`hermes-acquisition.yaml` 保存非敏感请求参数：

```yaml
root: ./data
job_id: hermes-cn-a-2010-20260724
start_date: 2010-01-01
end_date: 2026-07-24
financial_warmup_start: 2009-01-01
limits:
  max_connections: 8
  target_memory_gib: 78
  hard_memory_gib: 84
```

在项目根目录和已安装 `zyquant[hermes]` 的环境中执行：

```bash
cd /path/to/ZyQuant
source .venv/bin/activate

zyq data acquire \
  --source hermes \
  --action run \
  --request hermes-acquisition.yaml
```

状态查询和中断恢复：

```bash
zyq data acquire --source hermes --action status \
  --request hermes-acquisition.yaml

zyq data acquire --source hermes --action resume \
  --request hermes-acquisition.yaml
```

`resume` 会重新校验已完成文件的 SHA-256，只重拉缺失、损坏、失败或中断的
分块。改变日期、资源参数、查询或源 schema 会被拒绝，必须创建新 job。

## 发布

`hermes-publish.yaml` 只需指定已完成的 job：

```yaml
job_id: hermes-cn-a-2010-20260724
```

采集和标准化全部通过后，使用硬链接构造临时快照目录并原子提交：

```bash
zyq data publish \
  --root ./data \
  --dataset-id hermes-cn-a-2010-20260724-v1 \
  --source hermes \
  --request hermes-publish.yaml

zyq data validate \
  --root ./data \
  --dataset-id hermes-cn-a-2010-20260724-v1
```

已发布目录不可覆盖。Hermes 没有经确认的完整历史佣金、税费和过户费序列
时，快照声明 `backtest_ready=false`；研究、因子和财务查询可用，回测入口
会明确拒绝运行。

## 目录

```text
data/acquisitions/<job_id>/
├── state.sqlite
├── source_schema.json
├── raw/
├── canonical/
├── quarantine/
├── logs/
└── .partial/
```

行情和估值按月分区，财务按 100 个机构分组。所有正式分块均经过临时文件、
Parquet 关闭、SHA-256 校验和原子重命名后才标记成功。
