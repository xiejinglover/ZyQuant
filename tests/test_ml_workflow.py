from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from zyquant.backtest import BacktestEngine, StrategyBinding
from zyquant.config import ExecutionConfig
from zyquant.data import SnapshotPublisher
from zyquant.experiment import ExperimentStore
from zyquant.factors import FactorEngine, ReturnFactor
from zyquant.ml import DatasetBuilder, PurgedTimeSeriesSplitter
from zyquant.portfolio import ConstraintEngine, PortfolioConstraints, TopKEqualWeightConstructor
from zyquant.strategy import DailySchedule, ExternalSignalGenerator, PipelineStrategy, StandardUniverseSelector
from zyquant.workflow import WorkflowRunner

from tests.support import canonical_tables, signal_frame


class MlWorkflowTests(unittest.TestCase):
    def test_dataset_purging_and_atomic_workflow_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("sample-v1", tables)
            factors = FactorEngine(Path(temporary) / "cache")
            returns = factors.compute(ReturnFactor(1), snapshot, days[1], days[-1]).frame
            dataset = DatasetBuilder().build(
                snapshot, {"return_1d": returns}, days[1], days[-1], horizon=1
            )
            splits = PurgedTimeSeriesSplitter(folds=3, embargo_periods=1).split(dataset)
            self.assertTrue(splits)
            for train, valid in splits:
                valid_start = min(dataset.index.iloc[valid]["trade_date"])
                self.assertTrue((dataset.label_end_dates.iloc[train] < valid_start).all())

            strategy = PipelineStrategy(
                "alpha", DailySchedule(),
                StandardUniverseSelector("TEST", median_amount_window=1),
                ExternalSignalGenerator(signal_frame(days)),
                TopKEqualWeightConstructor(1),
                ConstraintEngine(PortfolioConstraints()),
            )
            store = ExperimentStore(Path(temporary) / "experiments.sqlite")
            runner = WorkflowRunner(Path(temporary) / "runs", store)
            result = runner.run_backtest(
                BacktestEngine(ExecutionConfig(
                    max_participation=1, commission_bps=0, minimum_commission=0,
                    stock_sell_tax_bps=0, slippage_bps=0,
                    impact_coefficient_bps=0,
                )),
                snapshot, days[0], days[-1], [StrategyBinding(strategy, 1)],
                100_000,
            )
            self.assertTrue((result.run_path / "manifest.json").exists())
            self.assertTrue((result.run_path / "report.html").exists())
            self.assertEqual(store.get_run(result.run_id)["status"], "succeeded")
            comparison = store.compare_runs([result.run_id])
            self.assertEqual(comparison[0]["run_id"], result.run_id)
            store.close()
