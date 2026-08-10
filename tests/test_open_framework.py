from __future__ import annotations

import tempfile
from inspect import signature
from pathlib import Path

import pytest

from zyquant.backtest import BacktestEngine, StrategyBinding
from zyquant.backtest.types import Fill
from zyquant.config import ExecutionConfig
from zyquant.core import plugins
from zyquant.core.exceptions import BacktestError
from zyquant.data import CanonicalBatch, SnapshotPublisher
from zyquant.strategy import (
    DailySchedule, StrategyDecision, StrategyState, TargetPortfolio,
)

from tests.support import canonical_tables


class MappingAdapter:
    def __init__(self, tables):
        self.tables = tables

    def ingest(self, request=None):
        return CanonicalBatch(self.tables, source_metadata={"source": "test"})


def create_data_adapter(request=None):
    tables, _ = canonical_tables()
    return MappingAdapter(tables)


def test_local_data_factory_can_be_loaded_without_packaging_it():
    factory = plugins.resolve(
        "data", "tests.test_open_framework:create_data_adapter", Path.cwd()
    )
    assert isinstance(factory({}).ingest(), CanonicalBatch)


def test_core_data_api_does_not_export_vendor_connectors():
    import zyquant.data as data

    assert not hasattr(data, "JQDataAdapter")
    assert not hasattr(data, "HermesDataAdapter")
    assert not hasattr(data, "SQLDataAdapter")


def test_backtest_checkpoint_api_is_not_exposed():
    import zyquant
    from pydantic import ValidationError

    from zyquant.cli.main import parser
    from zyquant.config import StrategyConfig

    assert not hasattr(zyquant, "CheckpointableStrategy")
    assert "checkpoint_dir" not in signature(BacktestEngine.run).parameters
    assert "resume_from" not in signature(BacktestEngine.run).parameters
    with pytest.raises(ValidationError, match="checkpoint_every_days"):
        StrategyConfig.model_validate({"checkpoint_every_days": 10})
    with pytest.raises(SystemExit):
        parser().parse_args(["backtest", "resume"])


class InvalidTargetStrategy:
    strategy_id = "invalid"
    schedule = DailySchedule()

    def decide(self, context):
        target = TargetPortfolio(
            self.strategy_id,
            context.signal_date,
            context.execution_date,
            context.execution_phase,
            {next(iter(context.data.table("instruments")["instrument_id"])): 1.2},
            0.0,
            "custom",
            "custom",
            "before",
            "after",
        )
        return StrategyDecision(target, StrategyState(), None)


def test_engine_validates_code_first_strategy_targets():
    with tempfile.TemporaryDirectory() as temporary:
        tables, days = canonical_tables()
        snapshot = SnapshotPublisher(temporary).publish("open-v2", tables)
        with pytest.raises(BacktestError, match="sum to one"):
            BacktestEngine().run(
                snapshot,
                days[0],
                days[-1],
                [StrategyBinding(InvalidTargetStrategy(), 1.0)],
                100_000,
            )


class InvalidExecutionModel:
    def execute(self, order, market_row, asset_type, rule=None):
        return Fill(
            order.order_id,
            order.execution_date,
            order.instrument_id,
            order.side,
            order.quantity,
            order.quantity + order.lot_size,
            order.reference_price,
            0.0,
            0.0,
            0.0,
            0.0,
            "filled",
            None,
        )


def test_custom_execution_model_cannot_fabricate_fill_quantity():
    from tests.support import signal_frame
    from tests.test_golden_market import strategy_for

    with tempfile.TemporaryDirectory() as temporary:
        tables, days = canonical_tables()
        snapshot = SnapshotPublisher(temporary).publish("execution-v2", tables)
        config = ExecutionConfig(
            max_participation=1,
            commission_bps=0,
            minimum_commission=0,
            stock_sell_tax_bps=0,
            slippage_bps=0,
            impact_coefficient_bps=0,
        )
        with pytest.raises(BacktestError, match="invalid fill quantity"):
            BacktestEngine(
                config, execution_model=InvalidExecutionModel()
            ).run(
                snapshot,
                days[0],
                days[-1],
                [StrategyBinding(strategy_for(signal_frame(days)), 1.0)],
                100_000,
            )


def test_builtin_data_sources_are_explicit_and_vendor_modules_are_lazy():
    from zyquant.connectors import BUILTIN_DATA_SOURCES

    assert set(BUILTIN_DATA_SOURCES) == {
        "canonical-directory", "hermes", "jqdata", "sql",
    }
    for name in BUILTIN_DATA_SOURCES:
        factory = plugins.resolve("data", name)
        assert factory.plugin_metadata.kind == "data"
