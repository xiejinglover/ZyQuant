from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from zyquant.config import load_config
from zyquant.core import plugins, source_tree_fingerprint
from zyquant.data import (
    ParquetDataProvider,
    SnapshotPublisher,
)
from zyquant.experiment import ExperimentStore
from zyquant.factors import FactorEngine
from zyquant.ml import (
    DatasetBuilder, ModelRegistry, PurgedTimeSeriesSplitter, SklearnTrainer,
    make_prediction_frame,
)
from zyquant.optimize import (
    Categorical, FloatRange, GridSampler, IntRange, RandomSampler, SearchEngine,
    SearchSpace, TrialContext,
)
from zyquant.workflow import WorkflowRunner


def _print(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="zyq")
    commands = root.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config", help="validate resolved configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    validate_config = config_commands.add_parser("validate")
    validate_config.add_argument("--config", required=True, type=Path)

    data = commands.add_parser("data", help="manage immutable market-data snapshots")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    publish = data_commands.add_parser("publish")
    publish.add_argument("--root", required=True, type=Path)
    publish.add_argument("--dataset-id", required=True)
    publish.add_argument("--source", required=True)
    publish.add_argument(
        "--request", type=Path,
        help="source-specific YAML/JSON request mapping",
    )
    publish.add_argument("--project-root", type=Path, default=Path.cwd())
    data_commands.add_parser("sources", help="list available data connectors")
    validate = data_commands.add_parser("validate")
    validate.add_argument("--root", required=True, type=Path)
    validate.add_argument("--dataset-id", required=True)
    listing = data_commands.add_parser("list")
    listing.add_argument("--root", required=True, type=Path)
    acquire = data_commands.add_parser(
        "acquire", help="run or inspect a resumable data connector"
    )
    acquire.add_argument("--source", required=True)
    acquire.add_argument("--action", choices=("run", "resume", "status"), required=True)
    acquire.add_argument("--request", required=True, type=Path)
    acquire.add_argument("--project-root", type=Path, default=Path.cwd())

    backtest = commands.add_parser("backtest")
    backtest_commands = backtest.add_subparsers(dest="backtest_command", required=True)
    backtest_run = backtest_commands.add_parser("run")
    backtest_run.add_argument("--config", required=True, type=Path)
    backtest_run.add_argument("--project-root", type=Path, default=Path.cwd())

    model = commands.add_parser("model")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_train = model_commands.add_parser("train")
    model_train.add_argument("--config", required=True, type=Path)
    model_train.add_argument("--project-root", type=Path, default=Path.cwd())
    model_predict = model_commands.add_parser("predict")
    for name in (
        "registry", "model-id", "features", "index", "snapshot-root",
        "dataset-id", "feature-set-id", "feature-cutoff", "train-cutoff",
        "strategy-id", "output",
    ):
        model_predict.add_argument(f"--{name}", required=True)

    search = commands.add_parser("search")
    search_commands = search.add_subparsers(dest="search_command", required=True)
    for action in ("run", "resume"):
        current = search_commands.add_parser(action)
        current.add_argument("--config", required=True, type=Path)
        current.add_argument("--project-root", type=Path, default=Path.cwd())

    runs_group = commands.add_parser("runs", help="inspect experiment runs")
    run_commands = runs_group.add_subparsers(dest="runs_command", required=True)
    runs = run_commands.add_parser("list")
    runs.add_argument("--database", required=True, type=Path)
    show = run_commands.add_parser("show")
    show.add_argument("--database", required=True, type=Path)
    show.add_argument("--run-id", required=True)
    compare = run_commands.add_parser("compare")
    compare.add_argument("--database", required=True, type=Path)
    compare.add_argument("--run-id", required=True, action="append")
    return root


def _space(payload):
    result = {}
    for name, spec in payload.items():
        kind = spec.get("type", "categorical")
        if kind == "categorical":
            result[name] = Categorical(tuple(spec["values"]))
        elif kind == "int":
            result[name] = IntRange(spec["low"], spec["high"], spec.get("step", 1))
        elif kind == "float":
            result[name] = FloatRange(
                spec["low"], spec["high"], spec.get("points"), spec.get("log", False)
            )
        else:
            raise ValueError(f"unknown search parameter type: {kind}")
    return SearchSpace(result)


def _read_mapping(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"request file must contain a mapping: {path}")
    return payload


def _data_sources() -> list[dict[str, Any]]:
    from zyquant.connectors import BUILTIN_DATA_SOURCES

    required_modules = {
        "hermes": "pymysql",
        "jqdata": "jqdatasdk",
        "sql": "sqlalchemy",
    }
    names = sorted(set(BUILTIN_DATA_SOURCES) | set(plugins.names("data")))
    output = []
    for name in names:
        factory = plugins.resolve("data", name)
        metadata = getattr(factory, "plugin_metadata", None)
        dependency = required_modules.get(name)
        output.append({
            "name": name,
            "version": getattr(metadata, "version", None),
            "available": dependency is None or importlib.util.find_spec(dependency) is not None,
            "optional_dependencies": list(
                getattr(metadata, "optional_dependencies", ())
            ),
        })
    return output


def _run_search(config_path: Path, project_root: Path = Path.cwd()):
    config = load_config(config_path)
    snapshot = ParquetDataProvider(config.data.root).open_snapshot(
        config.data.dataset_id, config.data.verify_hashes
    )
    sampler_config = dict(config.search.sampler)
    space = _space(sampler_config["space"])
    if sampler_config.get("type", "grid") == "grid":
        parameters = GridSampler().sample(space)
    else:
        parameters = RandomSampler(
            int(sampler_config["trials"]), int(sampler_config.get("seed", config.seed))
        ).sample(space)
    runner = plugins.resolve(
        "objectives", str(sampler_config["runner_plugin"]), project_root
    )
    store = ExperimentStore(config.experiment_database)
    try:
        table_hashes = {
            item["name"]: item["content_hash"]
            for item in snapshot.manifest["tables"]
        }
        result = SearchEngine(
            store, config.search.workers, config.search.retry_resource_errors,
            config.search.heartbeat_seconds,
        ).run(
            str(sampler_config["search_run_id"]),
            snapshot.metadata.fingerprint,
            source_tree_fingerprint(Path.cwd() / "src"),
            parameters, runner, str(config.search.objective),
            config.search.maximize,
            trial_context=TrialContext(
                snapshot.metadata.adjustment_version,
                table_hashes["daily_raw"],
                table_hashes["daily_post_adjusted"],
                seed=config.seed,
            ),
        )
        return {
            "search_run_id": result.search_run_id,
            "trials": len(result.trials),
            "best": result.best.parameters if result.best else None,
            "best_objective": result.best.objective if result.best else None,
        }
    finally:
        store.close()


def _train_model(config_path: Path, project_root: Path = Path.cwd()):
    config = load_config(config_path)
    snapshot = ParquetDataProvider(config.data.root).open_snapshot(
        config.data.dataset_id, config.data.verify_hashes
    )
    engine = FactorEngine(
        config.factor.cache_root,
        config.factor.lock_timeout_seconds,
        config.factor.cache_policy,
    )
    feature_frames = {}
    for spec in config.factor.factors:
        factory = plugins.resolve("factors", str(spec["plugin"]), project_root)
        factor = factory(**dict(spec.get("parameters", {})))
        feature_frames[str(spec.get("name", factor.name))] = engine.compute(
            factor, snapshot, config.data.start_date, config.data.end_date,
            cutoff=config.data.end_date,
        ).frame
    horizon = int(config.model.label.get("horizon", 1))
    dataset = DatasetBuilder().build(
        snapshot, feature_frames, config.data.start_date, config.data.end_date,
        horizon=horizon, cutoff=config.data.end_date,
    )
    split_config = dict(config.model.splitter)
    splits = PurgedTimeSeriesSplitter(
        int(split_config.get("folds", 5)),
        int(split_config.get("embargo_periods", 1)),
    ).split(dataset)
    if not splits:
        raise ValueError("training dataset cannot produce a valid split")
    plugin = plugins.resolve(
        "models", str(config.model.trainer["plugin"]), project_root
    )
    trainer = plugin if hasattr(plugin, "fit") else SklearnTrainer(plugin)
    train_indexes, validation_indexes = splits[-1]
    with tempfile.TemporaryDirectory() as temporary:
        outcome = trainer.fit(
            dataset, train_indexes, validation_indexes,
            Path(temporary) / "model.pkl",
        )
    registry = ModelRegistry(config.model.trainer["registry_root"])
    model_id = str(config.model.trainer["model_id"])
    model = outcome.get("model", outcome)
    artifact = registry.register(model_id, model, {
        "dataset_fingerprint": dataset.fingerprint,
        "data_fingerprint": snapshot.metadata.fingerprint,
        "train_cutoff": config.data.end_date,
        "feature_set": config.model.feature_set,
        "metrics": {
            key: value for key, value in outcome.items() if key != "model"
        } if isinstance(outcome, dict) else {},
    })
    return {"model_id": model_id, "path": artifact.path}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "data":
            plugins.discover(kinds=("data",))
        elif args.command in {"backtest", "model", "search"}:
            plugins.discover(kinds=(
                "factors", "models", "optimizers", "objectives",
                "reports", "execution_models", "cost_models",
                "capital_allocators",
            ))
        if args.command == "config":
            config = load_config(args.config)
            _print({
                "status": "valid", "fingerprint": config.fingerprint,
                "config": config.redacted(),
            })
        elif args.command == "data" and args.data_command == "publish":
            request = _read_mapping(args.request)
            factory = plugins.resolve("data", args.source, args.project_root)
            adapter = factory(request) if callable(factory) else factory
            if hasattr(adapter, "publish"):
                snapshot = adapter.publish(args.root, args.dataset_id, request)
            else:
                snapshot = SnapshotPublisher(args.root).publish_adapter(
                    args.dataset_id, adapter, request=request,
                )
            _print({
                "status": "published", "source": args.source,
                **snapshot.metadata.__dict__,
            })
        elif args.command == "data" and args.data_command == "sources":
            _print({"sources": _data_sources()})
        elif args.command == "data" and args.data_command == "validate":
            snapshot = ParquetDataProvider(args.root).open_snapshot(args.dataset_id, True)
            _print({
                "status": "valid", **snapshot.metadata.__dict__,
                "quality": snapshot.manifest["quality"],
            })
        elif args.command == "data" and args.data_command == "list":
            _print({"datasets": ParquetDataProvider(args.root).list_snapshots()})
        elif args.command == "data" and args.data_command == "acquire":
            request = _read_mapping(args.request)
            factory = plugins.resolve("data", args.source, args.project_root)
            adapter = factory(request) if callable(factory) else factory
            acquire = getattr(adapter, "acquire", None)
            if not callable(acquire):
                raise ValueError(
                    f"data source does not support acquisition: {args.source}"
                )
            _print(acquire(args.action, request))
        elif args.command == "backtest" and args.backtest_command == "run":
            config = load_config(args.config)
            with ExperimentStore(config.experiment_database) as store:
                result = WorkflowRunner(
                    config.output_root, store, args.project_root
                ).run(config)
            _print({"run_id": result.run_id, "path": result.run_path, "metrics": result.metrics})
        elif args.command == "model" and args.model_command == "train":
            _print(_train_model(args.config, args.project_root))
        elif args.command == "model" and args.model_command == "predict":
            registry = ModelRegistry(args.registry)
            model, _ = registry.load(args.model_id)
            snapshot = ParquetDataProvider(args.snapshot_root).open_snapshot(args.dataset_id)
            prediction = make_prediction_frame(
                model, pd.read_parquet(args.features), pd.read_parquet(args.index),
                snapshot, args.model_id, "1.0", args.feature_set_id,
                date.fromisoformat(args.feature_cutoff),
                date.fromisoformat(args.train_cutoff), args.strategy_id,
            )
            prediction.frame.to_parquet(args.output, index=False)
            _print({"output": args.output, "fingerprint": prediction.fingerprint})
        elif args.command == "search":
            _print(_run_search(args.config, args.project_root))
        elif args.command == "runs":
            with ExperimentStore(args.database) as store:
                if args.runs_command == "list":
                    _print({"runs": [dict(item) for item in store.list_runs()]})
                elif args.runs_command == "show":
                    record = store.get_run(args.run_id)
                    _print({"run": dict(record) if record else None})
                else:
                    _print({"runs": store.compare_runs(args.run_id)})
        return 0
    except Exception as exc:
        _print({"status": "error", "type": type(exc).__name__, "message": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
