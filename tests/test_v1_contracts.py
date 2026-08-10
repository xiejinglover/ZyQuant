from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pydantic import ValidationError

from zyquant.backtest import BacktestEngine, StrategyBinding
from zyquant.config import (
    AccountConfig, DataConfig, ExecutionConfig, ResolvedRunConfig,
)
from zyquant.core import PluginMetadata, PluginRegistry
from zyquant.core.exceptions import (
    ConstraintError, FactorCacheMiss, FutureDataError,
)
from zyquant.data import SnapshotPublisher
from zyquant.factors import FactorEngine, ReturnFactor
from zyquant.experiment import ExperimentStore
from zyquant.ml import ModelRegistry, StandardPreprocessor
from zyquant.optimize import SearchEngine
from zyquant.workflow import WorkflowRunner
from zyquant.portfolio import (
    ConstraintEngine, PortfolioConstraints, TopKEqualWeightConstructor,
)
from zyquant.strategy import (
    DailySchedule, ExternalSignalGenerator, PipelineStrategy,
    StandardUniverseSelector,
)

from tests.support import CODE_A, canonical_tables, signal_frame


def parallel_objective(parameters):
    return {"score": -abs(parameters["x"] - 2)}


class V1ContractTests(unittest.TestCase):
    def test_resolved_config_is_strict_frozen_hashable_and_redacted(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = ResolvedRunConfig(
                data=DataConfig(
                    root=temporary, dataset_id="v1",
                    start_date="2025-01-01", end_date="2025-01-31",
                ),
                account=AccountConfig(initial_cash=1_000_000),
                metadata={"api_token": "secret", "owner": "research"},
            )
            self.assertEqual(len(config.fingerprint), 64)
            self.assertFalse(config.analysis.attribution)
            self.assertEqual(config.factor.cache_policy, "compute")
            self.assertEqual(config.redacted()["metadata"]["api_token"], "***REDACTED***")
            with self.assertRaises(ValidationError):
                ResolvedRunConfig.model_validate({
                    **config.model_dump(), "unexpected": True,
                })
            with self.assertRaises(ValidationError):
                config.seed = 1

    def test_plugin_metadata_contract(self):
        registry = PluginRegistry()
        plugin = object()
        registry.register(
            "models", "demo", plugin,
            PluginMetadata("demo", "1.0.0", "models"),
        )
        self.assertIs(registry.get("models", "demo"), plugin)
        self.assertEqual(registry.metadata("models", "demo").version, "1.0.0")
        with self.assertRaises(Exception):
            registry.register("models", "missing", object())

    def test_factor_preflight_failure_happens_before_engine_run(self):
        class MissingFactorStrategy:
            strategy_id = "missing-factor"
            schedule = DailySchedule()

            @staticmethod
            def prepare_run(snapshot, factor_engine, start, end):
                raise FactorCacheMiss("prewarm missing-factor")

        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish(
                "sample-v1", tables
            )
            engine = BacktestEngine()
            runner = WorkflowRunner(Path(temporary) / "runs")
            with patch.object(engine, "run") as execute:
                with self.assertRaisesRegex(
                    FactorCacheMiss, "missing-factor"
                ):
                    runner.run_backtest(
                        engine, snapshot, days[0], days[-1],
                        [StrategyBinding(MissingFactorStrategy(), 1.0)],
                        100_000,
                    )
                execute.assert_not_called()

    def test_snapshot_requires_cutoff_and_resolves_historical_market_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("sample-v1", tables)
            with self.assertRaises(FutureDataError):
                snapshot.table("industry_membership")
            rule = snapshot.market_rule(days[2], "XSHG", "stock")
            self.assertEqual(rule.rule_id, "XSHG-stock-v1")
            self.assertTrue(snapshot.manifest["quality"]["raw_adjusted_key_match"])

    def test_constraint_protocol_errors_do_not_silently_filter(self):
        from zyquant.strategy.types import CandidateWeights

        engine = ConstraintEngine(PortfolioConstraints())
        with self.assertRaises(ConstraintError):
            engine.apply(
                CandidateWeights({"UNKNOWN": 1.0}, 0.0, "bad"),
                {CODE_A}, {},
            )
        with self.assertRaises(ConstraintError):
            engine.apply(
                CandidateWeights({CODE_A: -0.1}, 1.1, "bad"),
                {CODE_A}, {},
            )

    def test_factor_cache_reuses_broader_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("sample-v1", tables)
            engine = FactorEngine(Path(temporary) / "cache")
            broad = engine.compute(
                ReturnFactor(1), snapshot, days[1], days[-1], cutoff=days[-1]
            )
            narrow = engine.compute(
                ReturnFactor(1), snapshot, days[3], days[-2], cutoff=days[-1]
            )
            self.assertFalse(broad.from_cache)
            self.assertTrue(narrow.from_cache)
            self.assertEqual(broad.cache_key, narrow.cache_key)
            self.assertTrue(
                narrow.frame["trade_date"].between(days[3], days[-2]).all()
            )

    def test_master_and_sleeve_ledgers_are_reconciled(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            snapshot = SnapshotPublisher(temporary).publish("sample-v1", tables)
            strategy = PipelineStrategy(
                "alpha", DailySchedule(),
                StandardUniverseSelector("TEST", median_amount_window=1),
                ExternalSignalGenerator(signal_frame(days)),
                TopKEqualWeightConstructor(1),
                ConstraintEngine(PortfolioConstraints()),
            )
            execution = ExecutionConfig(
                max_participation=1, commission_bps=0, minimum_commission=0,
                stock_sell_tax_bps=0, slippage_bps=0,
                impact_coefficient_bps=0,
            )
            engine = BacktestEngine(execution)
            full = engine.run(
                snapshot, days[0], days[-1],
                [StrategyBinding(strategy, 1.0)], 100_000,
                compute_attribution=True,
            )
            self.assertTrue((full.frames["reconciliations"]["status"] == "passed").all())
            sleeve_nav = full.frames["nav"].groupby("date")["nav"].sum()
            master_nav = full.frames["master_nav"].set_index("date")["nav"]
            self.assertTrue(((sleeve_nav - master_nav).abs() < 1e-7).all())
            components = full.frames["attribution"]
            components = components[components["dimension"] == "pnl_component"]
            component_sum = components.groupby("date")["pnl"].sum()
            account = full.frames["nav"].groupby("date")["nav"].sum().diff().dropna()
            self.assertTrue(((component_sum - account).abs() < 1e-7).all())

    def test_parallel_search_matches_serial_and_registry_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            parameters = [{"x": value} for value in range(5)]
            serial_store = ExperimentStore(Path(temporary) / "serial.sqlite")
            parallel_store = ExperimentStore(Path(temporary) / "parallel.sqlite")
            try:
                serial = SearchEngine(serial_store, workers=1).run(
                    "serial", "data", "code", parameters,
                    parallel_objective, "score",
                )
                parallel = SearchEngine(parallel_store, workers=2).run(
                    "parallel", "data", "code", parameters,
                    parallel_objective, "score",
                )
                self.assertEqual(serial.best.parameters, parallel.best.parameters)
                self.assertEqual(serial.best.objective, parallel.best.objective)
            finally:
                serial_store.close()
                parallel_store.close()

            registry = ModelRegistry(Path(temporary) / "models")
            artifact = registry.register(
                "demo", {"coefficient": 1.0}, {"train_cutoff": "2025-01-01"}
            )
            model, loaded = registry.load("demo")
            self.assertEqual(model["coefficient"], 1.0)
            self.assertEqual(artifact.model_id, loaded.model_id)
            with self.assertRaises(Exception):
                registry.register("demo", {}, {})

            processor = StandardPreprocessor().fit(pd.DataFrame({
                "x": [1.0, 2.0, None],
            }))
            transformed = processor.transform(pd.DataFrame({"x": [100.0]}))
            self.assertGreater(float(transformed.iloc[0, 0]), 1.0)

    def test_declarative_workflow_commits_all_artifacts_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, days = canonical_tables()
            SnapshotPublisher(temporary).publish("sample-v1", tables)
            signal_path = Path(temporary) / "signals.parquet"
            signal_frame(days).to_parquet(signal_path, index=False)
            config = ResolvedRunConfig(
                data=DataConfig(
                    root=temporary, dataset_id="sample-v1",
                    start_date=days[0], end_date=days[-1],
                ),
                strategy={
                    "strategies": [{
                        "plugin": "pipeline",
                        "strategy_id": "declarative",
                        "capital_weight": 1.0,
                        "universe": {
                            "universe_id": "TEST",
                            "median_amount_window": 1,
                        },
                        "signal": {"type": "external", "path": str(signal_path)},
                        "constructor": {"type": "topk_equal", "top_k": 1},
                        "constraints": {},
                    }],
                },
                execution=ExecutionConfig(
                    max_participation=1, commission_bps=0,
                    minimum_commission=0, stock_sell_tax_bps=0,
                    slippage_bps=0, impact_coefficient_bps=0,
                ),
                account=AccountConfig(initial_cash=100_000),
                output_root=Path(temporary) / "runs",
                experiment_database=Path(temporary) / "experiments.sqlite",
            )
            with ExperimentStore(config.experiment_database) as store:
                result = WorkflowRunner(config.output_root, store).run(config)
                record = store.get_run(result.run_id)
                artifacts = list(store.connection.execute(
                    "SELECT name FROM artifacts WHERE run_id=?", (result.run_id,)
                ))
            self.assertEqual(record["status"], "succeeded")
            self.assertTrue((result.run_path / "manifest.json").exists())
            self.assertGreater(len(artifacts), 10)
            manifest = json.loads(
                (result.run_path / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["attribution_generated"])
            self.assertTrue(result.frames["attribution"].empty)


if __name__ == "__main__":
    unittest.main()
