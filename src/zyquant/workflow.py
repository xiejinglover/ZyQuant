from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from zyquant.analysis import write_html_report
from zyquant.backtest import BacktestEngine, BacktestResult, StrategyBinding
from zyquant.config import ResolvedRunConfig, load_config
from zyquant.core import environment_metadata, git_metadata, plugins
from zyquant.core.hashing import hash_payload
from zyquant.core.versioning import (
    FRAMEWORK_VERSION, LEDGER_SCHEMA_VERSION, RUN_SCHEMA_VERSION,
)
from zyquant.data import DataSnapshot, ParquetDataProvider
from zyquant.experiment import ExperimentStore
from zyquant.factors import FactorEngine


class WorkflowRunner:
    def __init__(
        self,
        output_root: str | Path,
        experiment_store: ExperimentStore | None = None,
        project_root: str | Path | None = None,
    ):
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.store = experiment_store
        self.project_root = Path(project_root or Path.cwd()).expanduser().resolve()

    def run(self, config: ResolvedRunConfig | Mapping[str, Any] | str | Path):
        resolved = load_config(config)
        if resolved.run_type == "model":
            return self._run_model(resolved)
        if resolved.run_type == "search":
            return self._run_search(resolved)
        snapshot = ParquetDataProvider(resolved.data.root).open_snapshot(
            resolved.data.dataset_id, resolved.data.verify_hashes
        )
        factor_engine = FactorEngine(
            resolved.factor.cache_root,
            resolved.factor.lock_timeout_seconds,
            resolved.factor.cache_policy,
        )
        cost_model = self._extension(
            "cost_models",
            resolved.execution.cost_model,
            resolved.execution.cost_model_parameters,
            config=resolved.execution,
        )
        execution_model = self._extension(
            "execution_models",
            resolved.execution.model,
            resolved.execution.model_parameters,
            config=resolved.execution,
            cost_model=cost_model,
        )
        capital_allocator = self._extension(
            "capital_allocators", resolved.account.capital_allocator,
            resolved.account.capital_allocator_parameters,
            strategy_weights={
                str(spec.get("strategy_id", dict(spec.get("parameters", {})).get(
                    "strategy_id", spec.get("plugin")
                ))): float(
                    spec["capital_weight"]
                )
                for spec in resolved.strategy.strategies
            },
        )
        engine = BacktestEngine(
            resolved.execution,
            factor_engine,
            execution_model=execution_model,
            cost_model=cost_model,
            capital_allocator=capital_allocator,
        )
        bindings = self._strategy_bindings(resolved)
        report_plugin = self._extension(
            "reports", resolved.analysis.report_plugin,
            resolved.analysis.report_parameters,
        )
        return self.run_backtest(
            engine, snapshot, resolved.data.start_date, resolved.data.end_date,
            bindings, resolved.account.initial_cash, resolved.seed,
            config=resolved.redacted(),
            compute_attribution=resolved.analysis.attribution,
            write_report=resolved.analysis.report,
            report_plugin=report_plugin,
        )

    def _run_model(self, config: ResolvedRunConfig):
        from zyquant.factors import MomentumFactor, ReturnFactor, RollingAmountFactor
        from zyquant.ml import (
            DatasetBuilder, ModelRegistry, PurgedTimeSeriesSplitter,
            RollingModelTrainer, SklearnTrainer,
        )

        snapshot = ParquetDataProvider(config.data.root).open_snapshot(
            config.data.dataset_id, config.data.verify_hashes
        )
        engine = FactorEngine(
            config.factor.cache_root,
            config.factor.lock_timeout_seconds,
            config.factor.cache_policy,
        )
        builtins: dict[str, Any] = {
            "return": ReturnFactor,
            "momentum": MomentumFactor,
            "rolling_amount": RollingAmountFactor,
        }
        feature_frames = {}
        for spec in config.factor.factors:
            plugin_name = str(spec["plugin"])
            factory = builtins.get(plugin_name)
            if factory is None:
                factory = plugins.resolve(
                    "factors", plugin_name, self.project_root
                )
            factor = factory(**dict(spec.get("parameters", {})))
            feature_frames[str(spec.get("name", factor.name))] = engine.compute(
                factor, snapshot, config.data.start_date, config.data.end_date,
                cutoff=config.data.end_date,
            ).frame
        dataset = DatasetBuilder().build(
            snapshot, feature_frames, config.data.start_date, config.data.end_date,
            horizon=int(config.model.label.get("horizon", 1)),
            cutoff=config.data.end_date,
        )
        split = dict(config.model.splitter)
        splitter = PurgedTimeSeriesSplitter(
            int(split.get("folds", 5)),
            int(split.get("embargo_periods", 1)),
        )
        trainer_plugin = plugins.resolve(
            "models", str(config.model.trainer["plugin"]), self.project_root
        )
        trainer = (
            trainer_plugin if hasattr(trainer_plugin, "fit")
            else SklearnTrainer(trainer_plugin)
        )
        registry = ModelRegistry(config.model.trainer["registry_root"])
        prefix = str(config.model.trainer["model_id"])
        run_id = hash_payload({
            "type": "model", "config": config.fingerprint,
            "dataset": dataset.fingerprint,
        })[:20]
        if self.store and self.store.get_run(run_id) is None:
            self.store.start_run(
                run_id, "model", config.redacted(),
                {
                    "data_fingerprint": snapshot.metadata.fingerprint,
                    "dataset_fingerprint": dataset.fingerprint,
                },
            )
        try:
            artifacts = RollingModelTrainer(registry).train(
                dataset, splitter, trainer, prefix,
                {
                    "feature_set": config.model.feature_set,
                    "data_fingerprint": snapshot.metadata.fingerprint,
                },
            )
            if self.store:
                for artifact in artifacts:
                    self.store.log_artifact(
                        run_id, f"model:{artifact.model_id}", artifact.path
                    )
                self.store.finish_run(run_id, {"models": len(artifacts)})
            return artifacts
        except Exception as exc:
            if self.store:
                self.store.fail_run(run_id, str(exc))
            raise

    def _run_search(self, config: ResolvedRunConfig):
        from zyquant.optimize import (
            Categorical, FloatRange, GridSampler, IntRange, RandomSampler,
            SearchEngine, SearchSpace, TrialContext,
        )

        snapshot = ParquetDataProvider(config.data.root).open_snapshot(
            config.data.dataset_id, config.data.verify_hashes
        )
        sampler_config = dict(config.search.sampler)
        values: dict[str, Any] = {}
        for name, spec in sampler_config["space"].items():
            kind = spec.get("type", "categorical")
            if kind == "categorical":
                values[name] = Categorical(tuple(spec["values"]))
            elif kind == "int":
                values[name] = IntRange(
                    spec["low"], spec["high"], spec.get("step", 1)
                )
            elif kind == "float":
                values[name] = FloatRange(
                    spec["low"], spec["high"], spec.get("points"),
                    spec.get("log", False),
                )
            else:
                raise ValueError(f"unknown search parameter type: {kind}")
        space = SearchSpace(values)
        parameters = (
            GridSampler().sample(space)
            if sampler_config.get("type", "grid") == "grid"
            else RandomSampler(
                int(sampler_config["trials"]),
                int(sampler_config.get("seed", config.seed)),
            ).sample(space)
        )
        objective = plugins.resolve(
            "objectives", str(sampler_config["runner_plugin"]), self.project_root
        )
        table_hashes = {
            item["name"]: item["content_hash"]
            for item in snapshot.manifest["tables"]
        }
        store = self.store
        owns_store = store is None
        if store is None:
            store = ExperimentStore(config.experiment_database)
        try:
            source = git_metadata(self.project_root)
            code_fingerprint = source.get(
                "commit", source.get("source_tree_fingerprint", "unknown")
            )
            return SearchEngine(
                store, config.search.workers, config.search.retry_resource_errors,
                config.search.heartbeat_seconds,
            ).run(
                str(sampler_config["search_run_id"]),
                snapshot.metadata.fingerprint,
                str(code_fingerprint),
                parameters, objective, str(config.search.objective),
                config.search.maximize,
                trial_context=TrialContext(
                    snapshot.metadata.adjustment_version,
                    table_hashes["daily_raw"],
                    table_hashes["daily_post_adjusted"],
                    seed=config.seed,
                ),
                keep_full_top_n=config.search.keep_full_top_n,
            )
        finally:
            if owns_store:
                store.close()

    def run_backtest(
        self,
        engine: BacktestEngine,
        snapshot: DataSnapshot,
        start: date,
        end: date,
        strategies: list[StrategyBinding],
        initial_cash: float,
        seed: int = 20260722,
        config: Mapping[str, Any] | None = None,
        compute_attribution: bool = False,
        write_report: bool = True,
        report_plugin=None,
    ):
        run_fingerprint = engine._run_fingerprint(
            snapshot, start, end, strategies, seed
        )
        run_id = run_fingerprint[:20]
        metadata = {
            "schema_version": RUN_SCHEMA_VERSION,
            "framework_version": FRAMEWORK_VERSION,
            "dataset_id": snapshot.metadata.dataset_id,
            "data_fingerprint": snapshot.metadata.fingerprint,
            "environment": environment_metadata(),
            "source": git_metadata(self.project_root),
        }
        existing = self.store.get_run(run_id) if self.store else None
        if existing is not None and existing["status"] == "succeeded":
            final = self.output_root / run_id
            if final.exists():
                return self._load_result(final)
        for binding in strategies:
            prepare = getattr(binding.strategy, "prepare_run", None)
            if callable(prepare):
                prepare(snapshot, engine.factor_engine, start, end)
        if self.store and existing is None:
            self.store.start_run(run_id, "backtest", config or {}, metadata)

        partial = self.output_root / ".incomplete" / run_id
        partial.mkdir(parents=True, exist_ok=True)
        try:
            result = engine.run(
                snapshot, start, end, strategies, initial_cash, seed,
                compute_attribution=compute_attribution,
            )
            staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=self.output_root))
            final = self.output_root / run_id
            try:
                for name, frame in result.frames.items():
                    frame.to_parquet(staging / f"{name}.parquet", index=False)
                strategy_plans = [
                    {"strategy_id": item.strategy.strategy_id, **dict(record)}
                    for item in strategies
                    for record in getattr(item.strategy, "planned", ())
                ]
                if strategy_plans:
                    pd.DataFrame(strategy_plans).to_parquet(
                        staging / "strategy_plans.parquet", index=False
                    )
                (staging / "metrics.json").write_text(
                    json.dumps(
                        result.metrics, ensure_ascii=False, indent=2, default=str
                    ),
                    encoding="utf-8",
                )
                (staging / "resolved_config.json").write_text(
                    json.dumps(config or {}, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                manifest = {
                    **metadata,
                    "run_id": run_id,
                    "run_fingerprint": run_fingerprint,
                    "adjustment_version": snapshot.metadata.adjustment_version,
                    "ledger_schema_version": LEDGER_SCHEMA_VERSION,
                    "config_fingerprint": hash_payload(config or {}),
                    "seed": seed,
                    "attribution_generated": bool(compute_attribution),
                    "cost_model": result.cost_model,
                    "strategy_factor_provenance": {
                        item.strategy.strategy_id: dict(
                            getattr(item.strategy, "factor_provenance", {})
                        )
                        for item in strategies
                        if getattr(item.strategy, "factor_provenance", {})
                    },
                    "artifacts": (
                        sorted(f"{name}.parquet" for name in result.frames)
                        + (
                            ["strategy_plans.parquet"]
                            if strategy_plans else []
                        )
                        + ["metrics.json", "resolved_config.json"]
                        + (["report.html"] if write_report else [])
                    ),
                }
                (staging / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                if write_report:
                    if report_plugin is None:
                        write_html_report(
                            staging / "report.html", result.metrics,
                            result.frames["nav"], result.frames["attribution"],
                        )
                    else:
                        writer = getattr(report_plugin, "write", report_plugin)
                        writer(
                            staging / "report.html", result.metrics,
                            result.frames, manifest,
                        )
                if final.exists():
                    existing_manifest = json.loads(
                        (final / "manifest.json").read_text(encoding="utf-8")
                    )
                    if existing_manifest.get("run_fingerprint") != run_fingerprint:
                        raise ValueError(
                            f"immutable run directory has conflicting content: {final}"
                        )
                    shutil.rmtree(staging)
                else:
                    os.replace(staging, final)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            shutil.rmtree(partial, ignore_errors=True)
            if self.store:
                self.store.finish_run(run_id, result.metrics)
                for path in sorted(final.rglob("*")):
                    if path.is_file():
                        self.store.log_artifact(
                            run_id, path.relative_to(final).as_posix(), path
                        )
            return BacktestResult(
                result.run_id, result.metrics, result.frames, final,
                cost_model=result.cost_model,
            )
        except KeyboardInterrupt:
            if self.store:
                self.store.fail_run(run_id, "run interrupted", "interrupted")
            raise
        except Exception as exc:
            if self.store:
                self.store.fail_run(run_id, str(exc))
            raise

    @staticmethod
    def _load_result(final: Path) -> BacktestResult:
        manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((final / "metrics.json").read_text(encoding="utf-8"))
        frames = {
            path.stem: pd.read_parquet(path)
            for path in final.glob("*.parquet")
        }
        return BacktestResult(
            manifest["run_id"], metrics, frames, final,
            cost_model=manifest.get("cost_model", {}),
        )

    def _strategy_bindings(self, config: ResolvedRunConfig) -> list[StrategyBinding]:
        bindings = []
        for spec in config.strategy.strategies:
            plugin_name = str(spec["plugin"])
            parameters = dict(spec.get("parameters", {}))
            if plugin_name == "pipeline":
                strategy = self._built_in_pipeline(spec)
            else:
                factory = plugins.resolve(
                    "strategies", plugin_name, self.project_root
                )
                strategy = factory(**parameters) if callable(factory) else factory
            bindings.append(StrategyBinding(
                strategy,
                float(spec["capital_weight"]),
                parameters,
            ))
        if not bindings:
            raise ValueError("configuration must declare at least one strategy plugin")
        return bindings

    def _extension(
        self, kind, reference, parameters, **injected,
    ):
        if not reference:
            return None
        factory = plugins.resolve(kind, str(reference), self.project_root)
        if not callable(factory):
            return factory
        return factory(**injected, **dict(parameters or {}))

    @staticmethod
    def _built_in_pipeline(spec):
        from zyquant.portfolio import (
            ConstraintEngine, PortfolioConstraints, RiskParityConstructor,
            ScoreWeightedConstructor, TopKDropoutConstructor,
            TopKEqualWeightConstructor,
        )
        from zyquant.strategy import (
            DailySchedule, EveryNTradingDays, ExplicitDateSchedule,
            ExternalSignalGenerator, MonthlySchedule, PipelineStrategy,
            StandardUniverseSelector, WeeklySchedule,
        )

        schedule_spec = dict(spec.get("schedule", {"type": "daily"}))
        schedule_type = schedule_spec.pop("type", "daily")
        schedules = {
            "daily": DailySchedule,
            "every_n": EveryNTradingDays,
            "explicit": ExplicitDateSchedule,
            "weekly": WeeklySchedule,
            "monthly": MonthlySchedule,
        }
        if schedule_type not in schedules:
            raise ValueError(f"unknown built-in schedule: {schedule_type}")
        if schedule_type == "explicit":
            schedule_spec["dates"] = tuple(
                item if isinstance(item, date) else date.fromisoformat(str(item))
                for item in schedule_spec["dates"]
            )
        schedule = schedules[schedule_type](**schedule_spec)
        universe = StandardUniverseSelector(**dict(spec.get("universe", {})))
        signal_spec = dict(spec["signal"])
        signal_type = signal_spec.pop("type", "external")
        if signal_type != "external":
            raise ValueError(
                "declarative built-in pipeline currently accepts external signals; "
                "use a strategy plugin for custom factor/model signals"
            )
        signal_path = Path(signal_spec.pop("path"))
        signals = (
            pd.read_parquet(signal_path)
            if signal_path.suffix == ".parquet" else pd.read_csv(signal_path)
        )
        signal = ExternalSignalGenerator(signals, **signal_spec)
        constructor_spec = dict(spec.get("constructor", {"type": "topk_equal", "top_k": 10}))
        constructor_type = constructor_spec.pop("type", "topk_equal")
        constructors = {
            "topk_equal": TopKEqualWeightConstructor,
            "topk_dropout": TopKDropoutConstructor,
            "score_weighted": ScoreWeightedConstructor,
            "risk_parity": RiskParityConstructor,
        }
        if constructor_type not in constructors:
            raise ValueError(f"unknown built-in constructor: {constructor_type}")
        constructor = constructors[constructor_type](**constructor_spec)
        constraints = ConstraintEngine(
            PortfolioConstraints(**dict(spec.get("constraints", {})))
        )
        return PipelineStrategy(
            str(spec["strategy_id"]), schedule, universe, signal, constructor,
            constraints,
            str(spec.get("no_candidate_policy", "hold_previous")),
            str(spec.get("constraint_failure_policy", "fail")),
        )
