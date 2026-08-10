from __future__ import annotations

import itertools
import math
import json
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from zyquant.core.hashing import hash_payload
from zyquant.core.exceptions import ResourceError
from zyquant.core.versioning import FRAMEWORK_VERSION
from zyquant.experiment import ExperimentStore


@dataclass(frozen=True)
class Categorical:
    values: tuple[Any, ...]


@dataclass(frozen=True)
class IntRange:
    low: int
    high: int
    step: int = 1


@dataclass(frozen=True)
class FloatRange:
    low: float
    high: float
    points: int | None = None
    log: bool = False


@dataclass(frozen=True)
class SearchSpace:
    parameters: Mapping[str, Any]


class GridSampler:
    def sample(self, space: SearchSpace) -> list[dict[str, Any]]:
        names, values = [], []
        for name, spec in space.parameters.items():
            names.append(name)
            if isinstance(spec, Categorical):
                values.append(spec.values)
            elif isinstance(spec, IntRange):
                values.append(tuple(range(spec.low, spec.high + 1, spec.step)))
            elif isinstance(spec, FloatRange) and spec.points:
                sequence = np.geomspace(spec.low, spec.high, spec.points) if spec.log else np.linspace(spec.low, spec.high, spec.points)
                values.append(tuple(map(float, sequence)))
            else:
                raise ValueError(f"grid parameter {name} needs finite values")
        return [dict(zip(names, combination, strict=True)) for combination in itertools.product(*values)]


@dataclass(frozen=True)
class RandomSampler:
    trials: int
    seed: int = 20260722

    def sample(self, space: SearchSpace) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        result: list[dict[str, Any]] = []
        seen = set()
        attempts = 0
        while len(result) < self.trials and attempts < self.trials * 100:
            attempts += 1
            current = {}
            for name, spec in space.parameters.items():
                if isinstance(spec, Categorical):
                    current[name] = spec.values[int(rng.integers(0, len(spec.values)))]
                elif isinstance(spec, IntRange):
                    choices = np.arange(spec.low, spec.high + 1, spec.step)
                    current[name] = int(rng.choice(choices))
                elif isinstance(spec, FloatRange):
                    if spec.log:
                        current[name] = float(math.exp(rng.uniform(math.log(spec.low), math.log(spec.high))))
                    else:
                        current[name] = float(rng.uniform(spec.low, spec.high))
                else:
                    current[name] = spec
            key = hash_payload(current)
            if key not in seen:
                seen.add(key)
                result.append(current)
        return result


@dataclass(frozen=True)
class TrialResult:
    trial_key: str
    parameters: Mapping[str, Any]
    metrics: Mapping[str, Any]
    objective: float
    status: str
    error: str | None = None
    reused: bool = False


@dataclass(frozen=True)
class TrialContext:
    adjustment_version: str
    raw_table_hash: str
    post_adjusted_table_hash: str
    factor_cache_keys: tuple[str, ...] = ()
    feature_cache_keys: tuple[str, ...] = ()
    engine_version: str = FRAMEWORK_VERSION
    seed: int = 20260722


@dataclass(frozen=True)
class SearchResult:
    search_run_id: str
    trials: tuple[TrialResult, ...]
    best: TrialResult | None
    full_results: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def _execute_runner(runner, parameters, objective_name, maximize):
    try:
        metrics = runner(parameters)
        objective = float(metrics[objective_name])
        if not np.isfinite(objective):
            raise ValueError(f"objective {objective_name} is not finite")
        return metrics, objective, None, False
    except Exception as exc:
        retryable = isinstance(exc, (ResourceError, OSError))
        return {}, float("nan"), traceback.format_exc(), retryable


class SearchEngine:
    def __init__(
        self,
        store: ExperimentStore,
        workers: int = 1,
        retry_resource_errors: int = 1,
        heartbeat_seconds: float = 5.0,
    ):
        self.store = store
        self.workers = max(1, workers)
        self.retry_resource_errors = max(0, retry_resource_errors)
        self.heartbeat_seconds = max(0.1, heartbeat_seconds)

    def run(
        self,
        search_run_id: str,
        dataset_fingerprint: str,
        code_fingerprint: str,
        parameters: Sequence[Mapping[str, Any]],
        runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        objective_name: str,
        maximize: bool = True,
        constraints: Mapping[str, tuple[str, float]] | None = None,
        trial_context: TrialContext | Mapping[str, Any] | None = None,
        full_runner: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        keep_full_top_n: int = 20,
    ) -> SearchResult:
        if self.store.get_run(search_run_id) is None:
            self.store.start_run(search_run_id, "search", {"objective": objective_name})
        results: list[TrialResult] = []
        pending = []
        for index, current in enumerate(parameters):
            key = hash_payload({
                "dataset": dataset_fingerprint, "code": code_fingerprint,
                "parameters": current,
                "context": (
                    trial_context.__dict__
                    if isinstance(trial_context, TrialContext)
                    else trial_context or {}
                ),
                "framework": FRAMEWORK_VERSION,
                "objective": objective_name,
            })
            existing = self.store.get_trial(key)
            if existing is not None and existing["status"] == "succeeded":
                results.append(TrialResult(
                    key, current, json.loads(existing["metrics_json"]),
                    float(existing["objective"]), "succeeded",
                    reused=True,
                ))
                continue
            trial_run_id = f"{search_run_id}-{index:06d}-{key[:8]}"
            self.store.upsert_trial(
                key, search_run_id, trial_run_id, "queued", current, attempts=0
            )
            pending.append((key, trial_run_id, current, 1))
        try:
            completed = self._execute_pending(
                pending, runner, objective_name, maximize, search_run_id
            )
        except KeyboardInterrupt:
            self.store.interrupt_running_trials(search_run_id)
            self.store.fail_run(search_run_id, "search interrupted", "interrupted")
            raise
        for (key, trial_run_id, current, attempts), (
            metrics, objective, error, _,
        ) in completed:
            if error is not None:
                self.store.upsert_trial(
                    key, search_run_id, trial_run_id, "failed", current,
                    error=error, attempts=attempts,
                )
                results.append(TrialResult(key, current, metrics, objective, "failed", error))
                continue
            feasible = self._feasible(metrics, constraints or {})
            status = "succeeded" if feasible else "constrained"
            self.store.upsert_trial(
                key, search_run_id, trial_run_id, status, current, objective,
                metrics=metrics, attempts=attempts,
            )
            results.append(TrialResult(key, current, metrics, objective, status))
        eligible = [item for item in results if item.status == "succeeded" and np.isfinite(item.objective)]
        best = (max(eligible, key=lambda x: x.objective) if maximize else min(eligible, key=lambda x: x.objective)) if eligible else None
        full_results: dict[str, Mapping[str, Any]] = {}
        if full_runner is not None and eligible:
            ranked = sorted(
                eligible, key=lambda item: item.objective, reverse=maximize
            )[:max(0, keep_full_top_n)]
            for item in ranked:
                full_results[item.trial_key] = full_runner(item.parameters)
        self.store.finish_run(search_run_id, {
            "trials": len(results), "succeeded": len(eligible),
            "best_objective": best.objective if best else None,
        })
        return SearchResult(search_run_id, tuple(results), best, full_results)

    def _execute_pending(
        self, pending, runner, objective_name, maximize, search_run_id
    ):
        if self.workers == 1:
            completed = []
            for item in pending:
                key, trial_run_id, current, attempts = item
                while True:
                    self.store.upsert_trial(
                        key, search_run_id, trial_run_id, "running", current,
                        attempts=attempts,
                    )
                    outcome = _execute_runner(
                        runner, current, objective_name, maximize
                    )
                    if not outcome[3] or attempts > self.retry_resource_errors:
                        completed.append(((key, trial_run_id, current, attempts), outcome))
                        break
                    attempts += 1
            return completed

        completed = []
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {}

            def submit(item):
                key, trial_run_id, current, attempts = item
                self.store.upsert_trial(
                    key, search_run_id, trial_run_id, "running", current,
                    attempts=attempts,
                )
                future = pool.submit(
                    _execute_runner, runner, current, objective_name, maximize
                )
                futures[future] = item

            for item in pending:
                submit(item)
            while futures:
                done, _ = wait(
                    futures, timeout=self.heartbeat_seconds,
                    return_when=FIRST_COMPLETED,
                )
                self.store.heartbeat([item[0] for item in futures.values()])
                for future in done:
                    item = futures.pop(future)
                    try:
                        outcome = future.result()
                    except Exception:
                        outcome = ({}, float("nan"), traceback.format_exc(), True)
                    key, trial_run_id, current, attempts = item
                    if outcome[3] and attempts <= self.retry_resource_errors:
                        submit((key, trial_run_id, current, attempts + 1))
                    else:
                        completed.append((item, outcome))
        return completed

    @staticmethod
    def _feasible(metrics, constraints):
        for name, (operator, threshold) in constraints.items():
            value = float(metrics[name])
            if operator == "<=" and value > threshold:
                return False
            if operator == ">=" and value < threshold:
                return False
            if operator not in {"<=", ">="}:
                raise ValueError(f"unsupported constraint operator: {operator}")
        return True
