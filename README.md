# ZyQuant

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/xiejinglover/ZyQuant/actions/workflows/test.yml/badge.svg)](https://github.com/xiejinglover/ZyQuant/actions/workflows/test.yml)

ZyQuant 是面向中国股票和 ETF 日频研究的轻量量化框架。核心边界是：

- 数据发布时一次性物化后复权行情，研究运行只读、不重复复权；
- 策略输出目标组合，交易约束和账户会计由回测层负责；
- 因子、模型、策略、回测和实验产物全部绑定不可变数据快照；
- 多策略通过虚拟袖套核算，在主账户层执行订单净额化。

当前版本为 `2.0.0`，实现了架构文档定义的完整单机日频研究基线：

- v1 manifest、lineage、质量报告、严格 PIT 查询与原始/后复权价格防火墙；
- 声明式因子 DAG、内容缓存、ML 数据集/滚动训练/模型登记；
- 标准策略流水线、显式状态、严格组合约束和可插拔算法；
- 主账户与多袖套守恒、内部交叉、T+1、公司行动和历史市场费率；
- 标准账本、绩效/归因、原子实验产物与多进程搜索。
- 可选的版本化财务报表、基本面指标、每日估值和历史股本 PIT 数据。
- Hermes 只读断点采集、沪深北 A 股、配股复权、流式标准化与原子发布。

## 安装

需要 Python 3.11 或更高版本。普通用户建议在独立虚拟环境中直接安装
[GitHub Release](https://github.com/xiejinglover/ZyQuant/releases/tag/v2.0.0) 的 wheel：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "https://github.com/xiejinglover/ZyQuant/releases/download/v2.0.0/zyquant-2.0.0-py3-none-any.whl"
```

验证安装：

```bash
python -c "import zyquant; print(zyquant.__version__)"
zyq --help
zyq data sources
```

核心数据契约与供应商无关。wheel 已包含规范目录、Hermes、JQData 和 SQL
连接器代码；只有实际使用的供应商 SDK 需要通过 extra 安装：

```bash
WHEEL_URL="https://github.com/xiejinglover/ZyQuant/releases/download/v2.0.0/zyquant-2.0.0-py3-none-any.whl"

python -m pip install "zyquant[hermes] @ ${WHEEL_URL}"
python -m pip install "zyquant[jqdata] @ ${WHEEL_URL}"
python -m pip install "zyquant[sql] @ ${WHEEL_URL}"
# 一次安装全部可选连接器：
python -m pip install "zyquant[connectors] @ ${WHEEL_URL}"
```

参与框架开发时才需要获取源码：

```bash
git clone https://github.com/xiejinglover/ZyQuant.git
cd ZyQuant
python -m pip install -e '.[dev]'
python -m pytest -q
```

全量 A 股的权威来源是 Hermes，凭据只从 `HERMES_MYSQL_*` 环境变量读取。
采集、恢复和发布命令见 [Hermes 全量数据运行手册](docs/hermes-acquisition.md)。

JQData 适配器保留用于显式的小样本数据发布，凭据只从环境变量读取：

```bash
WHEEL_URL="https://github.com/xiejinglover/ZyQuant/releases/download/v2.0.0/zyquant-2.0.0-py3-none-any.whl"
python -m pip install "zyquant[jqdata] @ ${WHEEL_URL}"
export JQDATA_USERNAME='<account>'
export JQDATA_PASSWORD='<password>'
zyq data publish \
  --root ./data \
  --dataset-id jqdata-sample-2025 \
  --source jqdata \
  --request examples/jqdata_sample.yaml
```

该示例只拉取三只证券的 2025 年日频行情和两只股票的财务数据，但保存完整的
历史沪深300成分和相关证券/行业元数据；manifest 会分别标记行情和财务覆盖。
正式获取完整股票池行情时在 JQData 请求文件中使用
`price_scope: universe`。目录中的规范数据可通过
`--source canonical-directory --request examples/canonical_directory.yaml` 发布。

数据快照必须包含 `market_rules`。旧 0.1 快照和实验数据库不会被静默升级，
请使用 v1 数据契约重新发布。

具体策略不进入 wheel。简单研究可使用声明式 `pipeline`；复杂策略直接在用户
项目中实现 `Strategy.decide()`，并以模块引用加载：

```yaml
strategy:
  strategies:
    - plugin: strategies.my_strategy:create_strategy
      capital_weight: 1.0
      parameters: {stock_num: 20}
```

```bash
zyq backtest run --project-root . --config config.yaml
```

## 文档

- [总体架构](docs/architecture.md)
- [标准数据契约](docs/data-contract.md)
- [逐字段数据字典](docs/data-dictionary.md)
- [策略与组合开发约定](docs/strategy-guide.md)
- [2.0 配置与运行](docs/configuration.md)
- [标准账本契约](docs/ledger-contract.md)
- [Hermes 全量数据运行手册](docs/hermes-acquisition.md)
- [因子层指南](docs/factor-guide.md)
- [问题与踩坑记录](docs/issue-log.md)

## 开源许可

ZyQuant 使用 [Apache License 2.0](LICENSE) 开源。允许在保留许可证和
相关声明的前提下使用、修改、商用和再发布。
