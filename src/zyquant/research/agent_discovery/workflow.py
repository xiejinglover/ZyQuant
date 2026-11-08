from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml

from zyquant.research.factor_discovery.adapter import load_adapter
from zyquant.research.factor_discovery.evaluator import ExpressionEvaluator
from zyquant.research.factor_discovery.evidence import RollingEvidenceEngine
from zyquant.research.factor_discovery.models import ExpressionSpec

from .executor import (
    ZyQuantExperimentExecutor,
    atomic_write_json,
    redact_payload,
    sha256_file,
    sha256_json,
)
from .models import (
    AgentCampaignPlan,
    AgentCampaignResult,
    AgentResearchConfig,
    FactorCandidateSpec,
    JointExperimentSpec,
    ResearchHypothesis,
)
from .replay import ReplayAgentClient


class AgentClient(Protocol):
    def propose_factors(
        self, *, context: dict[str, Any], count: int, iteration: int
    ) -> tuple[tuple[ResearchHypothesis, ...], tuple[FactorCandidateSpec, ...], dict[str, Any]]: ...

    def propose_experiment(
        self, *, context: dict[str, Any], experiment_kind: str, ordinal: int
    ) -> tuple[JointExperimentSpec, dict[str, Any]]: ...


class _IsolatedAgentClient:
    """Filesystem adapter for the separately installed RD-Agent worker."""

    def __init__(self, plan: AgentCampaignPlan, run_dir: Path) -> None:
        self.plan = plan
        self.run_dir = run_dir
        self.settings = plan.config.agent_executor
        self.calls_dir = run_dir / "agent_requests"
        self.calls_dir.mkdir(parents=True, exist_ok=True)

    def propose_factors(
        self, *, context: dict[str, Any], count: int, iteration: int
    ) -> tuple[tuple[ResearchHypothesis, ...], tuple[FactorCandidateSpec, ...], dict[str, Any]]:
        response = self._call("factor_discovery", iteration, {
            "context": context,
            "count": count,
            "iteration": iteration,
        })
        hypotheses = tuple(
            ResearchHypothesis.model_validate(item) for item in response.get("hypotheses", ())
        )
        candidates = tuple(
            FactorCandidateSpec.model_validate(item) for item in response.get("candidates", ())
        )
        return hypotheses, candidates[:count], self._audit(response)

    def propose_experiment(
        self, *, context: dict[str, Any], experiment_kind: str, ordinal: int
    ) -> tuple[JointExperimentSpec, dict[str, Any]]:
        response = self._call(f"{experiment_kind}_experiment", ordinal, {
            "context": context,
            "experiment_kind": experiment_kind,
            "ordinal": ordinal,
        })
        raw = response.get("experiment") or response.get("experiment_spec")
        if raw is None:
            hypotheses = response.get("hypotheses", ())
            if hypotheses:
                raw = hypotheses[0].get("experiment_spec")
        if raw is None:
            raise ValueError("RD-Agent worker response omitted experiment_spec")
        return JointExperimentSpec.model_validate(raw), self._audit(response)

    def _call(self, phase: str, ordinal: int, payload: dict[str, Any]) -> dict[str, Any]:
        call_id = f"{phase}-{ordinal:04d}"
        directory = self.calls_dir / call_id
        directory.mkdir(parents=True, exist_ok=True)
        request_path = directory / "request.json"
        response_path = directory / "response.json"
        request = {
            "contract_version": "1.0",
            "operation": (
                "propose_factors" if phase == "factor_discovery"
                else "propose_experiment"
            ),
            **payload,
        }
        request_id = sha256_json(request)
        atomic_write_json(request_path, request, immutable=True)
        if not response_path.exists():
            replacements = {
                "request": str(request_path),
                "output_dir": str(directory),
                "experiment_path": str(directory),
                "plan_path": str(self.run_dir / "plan.json"),
                "run_dir": str(self.run_dir),
                "spec_path": str(request_path),
                "config_path": str(self.run_dir / "plan.json"),
            }
            argv = [argument.format(**replacements) for argument in self.settings.argv]
            completed = subprocess.run(
                argv, check=False, capture_output=True, text=True,
                timeout=self.settings.timeout_seconds, shell=False, env=os.environ.copy(),
            )
            atomic_write_json(directory / "worker_audit.json", redact_payload({
                "argv_program": argv[0], "returncode": completed.returncode,
                "stdout": completed.stdout[-20_000:], "stderr": completed.stderr[-20_000:],
            }))
            if completed.returncode:
                raise RuntimeError(f"RD-Agent worker failed: exit {completed.returncode}")
        if not response_path.exists():
            # Workers may use the generic output contract name.
            alternative = directory / "output.json"
            if alternative.exists():
                response_path = alternative
            else:
                raise RuntimeError("RD-Agent worker did not publish response.json")
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("request_id") not in {None, request_id}:
            raise ValueError("RD-Agent response request_id mismatch")
        return response

    @staticmethod
    def _audit(response: dict[str, Any]) -> dict[str, Any]:
        audit = dict(response.get("audit", {}))
        audit.setdefault("provider", "rdagent")
        audit.setdefault("response_id", response.get("response_id"))
        audit.setdefault("usage", response.get("usage", {}))
        audit.setdefault("status", response.get("status", "completed"))
        return audit


@dataclass
class CampaignState:
    phase: str = "factor_discovery"
    factor_rounds: int = 0
    raw_proposals: int = 0
    accepted_unique_factors: int = 0
    completed_incremental_experiments: int = 0
    completed_joint_experiments: int = 0
    locked: bool = False
    diagnostics_complete: bool = False
    reference_complete: bool = False


def load_agent_config(path: str | Path) -> AgentResearchConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("agent research config must contain a mapping")
    return AgentResearchConfig.model_validate(payload)


def _git_commit() -> str:
    configured = os.environ.get("ZYQUANT_BUILD_COMMIT", "").strip()
    if configured:
        return configured
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, shell=False,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown0"


def create_agent_plan(config_path: str | Path, output_dir: str | Path) -> AgentCampaignPlan:
    config = load_agent_config(config_path)
    plan = AgentCampaignPlan.create(
        config,
        git_commit=_git_commit(),
        expression_schema_hash=sha256_json(ExpressionSpec.model_json_schema()),
    )
    destination = Path(output_dir).expanduser().resolve()
    _require_new_or_empty(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        atomic_write_json(
            staging / "frozen_spec.json", plan.model_dump(mode="json"), immutable=True
        )
        atomic_write_json(staging / "input_manifest.json", {
            "schema_version": "2.2",
            "plan_id": plan.plan_id,
            "config_hash": plan.config_hash,
            "expression_schema_hash": plan.expression_schema_hash,
            "dataset_id": config.dataset_id,
            "dataset_fingerprint": config.dataset_fingerprint,
            "pool_id": config.pool_id,
            "rdagent_version": plan.rdagent_version,
            "rdagent_commit": plan.rdagent_commit,
            "feedback_eligible_windows": ["research_feedback"],
        }, immutable=True)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return plan


def load_agent_plan(plan_dir: str | Path) -> AgentCampaignPlan:
    root = Path(plan_dir).expanduser().resolve()
    plan = AgentCampaignPlan.model_validate_json(
        (root / "frozen_spec.json").read_text(encoding="utf-8")
    )
    if sha256_json(plan.config.model_dump(mode="json")) != plan.config_hash:
        raise ValueError("frozen agent plan config hash mismatch")
    if sha256_json(ExpressionSpec.model_json_schema()) != plan.expression_schema_hash:
        raise ValueError("frozen plan expression schema differs from installed ZyQuant")
    return plan


class AgentCampaignRunner:
    def __init__(
        self,
        plan: AgentCampaignPlan,
        run_dir: str | Path,
        *,
        client: AgentClient | None = None,
        executor: ZyQuantExperimentExecutor | None = None,
    ) -> None:
        self.plan = plan
        self.config = plan.config
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.client = client
        self.executor = executor
        self.budgets = self.config.budgets
        self._dataset: Any | None = None
        self._evaluator = ExpressionEvaluator(set(self.config.allowed_fields))
        self._evidence = RollingEvidenceEngine(self.config.evidence)

    def run(self, *, resume: bool = False) -> AgentCampaignResult:
        self._resource_preflight()
        if resume:
            self._verify_identity()
            terminal = self.run_dir / "SUCCESS.json"
            if terminal.exists():
                return AgentCampaignResult.model_validate_json(
                    terminal.read_text(encoding="utf-8")
                )
        else:
            _require_new_or_empty(self.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                self.run_dir / "plan.json", self.plan.model_dump(mode="json"), immutable=True
            )
        state = self._load_state()
        started_at = self._started_at()
        client = self.client or self._make_client()
        executor = self.executor or self._make_executor()
        if isinstance(client, ReplayAgentClient):
            client.seek(
                factor_rounds=(
                    len(list((self.run_dir / "factor_rounds").glob("*.json")))
                    if (self.run_dir / "factor_rounds").exists() else 0
                ),
                experiments=(
                    len(list((self.run_dir / "experiment_proposals").glob("*.json")))
                    if (self.run_dir / "experiment_proposals").exists() else 0
                ),
            )

        if state.phase == "factor_discovery":
            self._run_factor_stage(client, state)
            state.phase = "incremental_experiments"
            state.locked = True
            self._save_state(state)
            atomic_write_json(self.run_dir / "FACTOR_LOCK.json", {
                "plan_id": self.plan.plan_id,
                "accepted_factor_ids": sorted(self._accepted()),
                "raw_proposals": state.raw_proposals,
                "factor_rounds": state.factor_rounds,
                "locked_at": datetime.now(timezone.utc).isoformat(),
            }, immutable=True)

        if state.phase == "incremental_experiments":
            self._run_experiment_stage(client, executor, state, "incremental")
            state.phase = "joint_experiments"
            self._save_state(state)
        if state.phase == "joint_experiments":
            self._run_experiment_stage(client, executor, state, "joint")
            state.phase = "locked_evaluation"
            self._save_state(state)
            self._lock_pareto(state)
        if state.phase == "locked_evaluation":
            self._run_locked_evaluations(executor, state)
            state.phase = "complete"
            self._save_state(state)

        status = (
            "complete"
            if state.accepted_unique_factors >= self.budgets.factor_target
            else "incomplete"
        )
        finished = datetime.now(timezone.utc)
        result_payload = {
            "campaign_id": self.config.campaign_id,
            "plan_id": self.plan.plan_id,
            "status": status,
            "run_dir": self.run_dir,
            "raw_proposals": state.raw_proposals,
            "accepted_unique_factors": state.accepted_unique_factors,
            "completed_incremental_experiments": state.completed_incremental_experiments,
            "completed_joint_experiments": state.completed_joint_experiments,
            "pareto_experiment_ids": tuple(self._pareto_ids()),
            "started_at": started_at,
            "finished_at": finished,
        }
        manifest_hash = self._result_manifest(result_payload)
        result = AgentCampaignResult(**result_payload, result_manifest_hash=manifest_hash)
        atomic_write_json(
            self.run_dir / "SUCCESS.json", result.model_dump(mode="json"), immutable=True
        )
        from .report import write_agent_report
        write_agent_report(self.run_dir)
        return result

    def _run_factor_stage(self, client: AgentClient, state: CampaignState) -> None:
        accepted = self._accepted()
        while (
            len(accepted) < self.budgets.factor_target
            and state.factor_rounds < self.budgets.max_factor_rounds
            and state.raw_proposals < self.budgets.max_raw_proposals
        ):
            iteration = state.factor_rounds + 1
            count = min(
                self.budgets.proposals_per_round,
                self.budgets.max_raw_proposals - state.raw_proposals,
            )
            round_path = self.run_dir / "factor_rounds" / f"{iteration:04d}.json"
            round_cached = round_path.exists()
            if round_cached:
                cached = json.loads(round_path.read_text(encoding="utf-8"))
                hypotheses = tuple(
                    ResearchHypothesis.model_validate(item) for item in cached["hypotheses"]
                )
                candidates = tuple(
                    FactorCandidateSpec.model_validate(item) for item in cached["candidates"]
                )
            else:
                hypotheses, candidates, audit = client.propose_factors(
                    context=self._agent_context(), count=count, iteration=iteration
                )
                atomic_write_json(round_path, {
                    "iteration": iteration,
                    "hypotheses": [item.model_dump(mode="json") for item in hypotheses],
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                }, immutable=True)
                self._append_jsonl("agent_calls.jsonl", redact_payload({
                    **audit, "phase": "factor_discovery", "iteration": iteration,
                }))
            if not round_cached:
                for hypothesis in hypotheses:
                    self._append_jsonl("hypotheses.jsonl", hypothesis.model_dump(mode="json"))
            reserved: set[str] = set()
            recorded_ids = {
                FactorCandidateSpec.model_validate(row["candidate"]).factor_id
                for row in self._read_jsonl("factor_proposals.jsonl")
                if "candidate" in row
            } | set(accepted)
            for candidate in candidates[:count]:
                if candidate.factor_id in recorded_ids:
                    continue
                state.raw_proposals += 1
                record: dict[str, Any] = {
                    "proposal_number": state.raw_proposals,
                    "iteration": iteration,
                    "candidate": candidate.model_dump(mode="json"),
                }
                try:
                    candidate.expression.validate_contract(set(self.config.allowed_fields))
                    if candidate.factor_id in accepted or candidate.factor_id in reserved:
                        raise ValueError("exact_canonical_duplicate")
                    validation = self._validate_numeric(candidate, accepted)
                    if validation.get("maximum_absolute_correlation", 0.0) >= (
                        self.config.exact_duplicate_correlation
                    ):
                        raise ValueError("numeric_duplicate_correlation")
                    record.update({"status": "accepted", "validation": validation})
                    reserved.add(candidate.factor_id)
                    accepted[candidate.factor_id] = candidate
                    self._append_jsonl("accepted_factors.jsonl", {
                        "factor_id": candidate.factor_id,
                        "candidate": candidate.model_dump(mode="json"),
                        "validation": validation,
                    })
                except Exception as exc:
                    record.update({
                        "status": "rejected", "reason": str(exc),
                        "error_type": type(exc).__name__,
                    })
                self._append_jsonl("factor_proposals.jsonl", record)
                self._save_state(state)
            state.factor_rounds = iteration
            state.accepted_unique_factors = len(accepted)
            self._save_state(state)

    def _run_experiment_stage(
        self, client: AgentClient, executor: ZyQuantExperimentExecutor,
        state: CampaignState, kind: str,
    ) -> None:
        limit = (
            self.budgets.max_incremental_experiments
            if kind == "incremental" else self.budgets.max_joint_experiments
        )
        field = (
            "completed_incremental_experiments"
            if kind == "incremental" else "completed_joint_experiments"
        )
        while getattr(state, field) < limit:
            ordinal = getattr(state, field) + 1
            proposal_path = (
                self.run_dir / "experiment_proposals" / f"{kind}-{ordinal:04d}.json"
            )
            if proposal_path.exists():
                spec = JointExperimentSpec.model_validate_json(
                    proposal_path.read_text(encoding="utf-8")
                )
            else:
                spec, audit = client.propose_experiment(
                    context=self._agent_context(), experiment_kind=kind, ordinal=ordinal
                )
                atomic_write_json(
                    proposal_path, spec.model_dump(mode="json"), immutable=True
                )
                self._append_jsonl("agent_calls.jsonl", redact_payload({
                    **audit, "phase": f"{kind}_experiments", "ordinal": ordinal,
                }))
            self._validate_experiment(spec, kind)
            feedback = executor.execute(spec)
            # ExperimentFeedback's literal type is the second leakage barrier.
            if feedback.feedback_window != "research_feedback":
                raise ValueError("non-development evidence cannot enter agent feedback")
            if feedback.experiment_id not in {
                item.get("experiment_id") for item in self._read_jsonl("experiment_feedback.jsonl")
            }:
                self._append_jsonl(
                    "experiment_feedback.jsonl", feedback.model_dump(mode="json")
                )
            setattr(state, field, ordinal)
            self._save_state(state)

    def _validate_experiment(self, spec: JointExperimentSpec, kind: str) -> None:
        if spec.campaign_id != self.config.campaign_id or spec.pool_id != self.config.pool_id:
            raise ValueError("experiment campaign or pool differs from frozen plan")
        if spec.experiment_kind != kind:
            raise ValueError("experiment kind differs from active stage")
        if spec.model_recipe.seeds != self.config.seeds:
            raise ValueError("experiment seeds differ from frozen plan")
        if spec.model_recipe.n_jobs > self.config.resources.threads_per_model:
            raise ValueError("experiment n_jobs exceeds frozen threads_per_model")
        if max(spec.prediction_vintages) > self.config.windows.research_feedback.end.year:
            raise ValueError("experiment vintages exceed research_feedback window")
        accepted = self._accepted()
        unknown = {factor.factor_id for factor in spec.candidate_factors} - set(accepted)
        if unknown:
            raise ValueError(f"experiment references unlocked factors: {sorted(unknown)}")
        allowed_baseline = set(self.config.baseline_factor_ids)
        if not set(spec.baseline_factor_ids).issubset(allowed_baseline):
            raise ValueError("experiment references baseline factors outside frozen catalog")

    def _run_locked_evaluations(
        self, executor: ZyQuantExperimentExecutor, state: CampaignState
    ) -> None:
        specs = self._locked_specs()
        if not state.diagnostics_complete:
            for spec in specs:
                executor.execute_locked(spec, "diagnostic")
            state.diagnostics_complete = True
            self._save_state(state)
        if not state.reference_complete:
            for spec in specs:
                executor.execute_locked(spec, "reference")
            state.reference_complete = True
            self._save_state(state)
        account = self.config.account_adapter
        if account is not None:
            account_executor = ZyQuantExperimentExecutor(
                self.run_dir, worker_argv=account.argv,
                timeout_seconds=account.timeout_seconds,
            )
            for spec in specs:
                account_executor.execute_locked(spec, "formal_account")

    def _locked_specs(self) -> list[JointExperimentSpec]:
        specs = []
        for experiment_id in self._pareto_ids():
            path = self.run_dir / "experiments" / experiment_id / "request.json"
            if not path.exists():
                raise ValueError(f"locked experiment request is missing: {experiment_id}")
            spec = JointExperimentSpec.model_validate_json(path.read_text(encoding="utf-8"))
            if spec.experiment_id != experiment_id:
                raise ValueError("locked experiment request hash mismatch")
            specs.append(spec)
        return specs

    def _validate_numeric(
        self, candidate: FactorCandidateSpec, accepted: dict[str, FactorCandidateSpec]
    ) -> dict[str, Any]:
        dataset = self._research_dataset()
        if dataset is None:
            return {
                "finite_values": None,
                "maximum_absolute_correlation": 0.0,
                "nearest_factor_id": None,
                "redundant": False,
                "validation_mode": "worker_deferred",
            }
        index = dataset.observations[["decision_time", "instrument_id"]].rename(
            columns={"decision_time": "trade_date"}
        ).drop_duplicates()
        values = self._evaluator.evaluate(candidate.expression, dataset.fields, index)
        finite = values["value"].replace([np.inf, -np.inf], np.nan)
        if finite.notna().sum() == 0:
            raise ValueError("candidate_has_no_finite_values")
        sample = pd.DataFrame({
            "trade_date": values["trade_date"],
            "instrument_id": values["instrument_id"],
            candidate.factor_id: finite,
        })
        factor_dir = self.run_dir / "factor_shards"
        factor_dir.mkdir(parents=True, exist_ok=True)
        output = factor_dir / f"{candidate.factor_id}.parquet"
        temporary = output.with_suffix(".tmp.parquet")
        sample.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, output)
        nearest_id: str | None = None
        nearest = 0.0
        for factor_id in accepted:
            path = factor_dir / f"{factor_id}.parquet"
            if not path.exists():
                continue
            other = pd.read_parquet(path)
            merged = sample.merge(other, on=["trade_date", "instrument_id"], how="inner")
            if len(merged) < 3:
                continue
            absolute = _median_cross_sectional_spearman(
                merged, candidate.factor_id, factor_id
            )
            if absolute > nearest:
                nearest, nearest_id = absolute, factor_id
        for factor_id in self.config.baseline_factor_ids:
            field_name = f"factor.{factor_id}"
            if field_name not in dataset.fields:
                continue
            other = dataset.fields[field_name].rename(columns={"value": factor_id})
            merged = sample.merge(other, on=["trade_date", "instrument_id"], how="inner")
            absolute = _median_cross_sectional_spearman(
                merged, candidate.factor_id, factor_id
            )
            if absolute > nearest:
                nearest, nearest_id = absolute, factor_id
        evidence = self._candidate_evidence(candidate, values, dataset)
        return {
            "finite_values": int(finite.notna().sum()),
            "maximum_absolute_correlation": nearest,
            "nearest_factor_id": nearest_id,
            "redundant": nearest >= self.config.redundant_correlation,
            "factor_shard_sha256": sha256_file(output),
            "validation_mode": "local_pit_adapter",
            "evidence": evidence,
        }

    def _candidate_evidence(
        self, candidate: FactorCandidateSpec, values: pd.DataFrame, dataset: Any
    ) -> dict[str, Any]:
        required = RollingEvidenceEngine.OBSERVATION_COLUMNS
        if not required.issubset(dataset.observations.columns):
            return {"status": "unavailable", "reason": "adapter_missing_evidence_columns"}
        mapping = dataset.observations[
            ["observation_id", "decision_time", "instrument_id"]
        ]
        joined = values.merge(
            mapping,
            left_on=["trade_date", "instrument_id"],
            right_on=["decision_time", "instrument_id"],
            how="inner", validate="one_to_one",
        )
        factor_values = pd.DataFrame({
            "observation_id": joined["observation_id"],
            "factor_id": candidate.factor_id,
            "factor_value": joined["value"],
            "factor_known_at": joined["decision_time"],
        })
        result = self._evidence.evaluate(
            dataset.observations,
            factor_values,
            pd.DataFrame([{"factor_id": candidate.factor_id, "factor_type": "continuous"}]),
            dataset.label_metadata,
            dataset.exposures,
        )
        health = result.factor_health.iloc[0].to_dict() if not result.factor_health.empty else {}
        summary = result.factor_summary.iloc[0].to_dict() if not result.factor_summary.empty else {}
        annual_rank_ic: dict[int, float | None] = {}
        annual_spread: dict[int, float | None] = {}
        if not result.daily_metrics.empty:
            daily = result.daily_metrics.assign(
                year=pd.to_datetime(result.daily_metrics["decision_time"]).dt.year
            )
            annual_rank_ic = daily.groupby("year")["rank_ic"].mean().to_dict()
            annual_spread = daily.groupby("year")["spread"].mean().to_dict()
        return redact_payload({
            "status": "computed",
            "coverage": health.get("non_null_ratio"),
            "annual_rank_ic": annual_rank_ic,
            "annual_spread_return": annual_spread,
            "summary": summary,
            "exposure_diagnostics": result.exposure_diagnostics.to_dict("records"),
            "failure_reasons": result.failure_records.get(
                "reason", pd.Series(dtype=str)
            ).dropna().astype(str).tolist(),
        })

    def _research_dataset(self) -> Any | None:
        if self._dataset is not None:
            return self._dataset
        adapter_spec = getattr(self.config, "dataset_adapter", None)
        if adapter_spec is None:
            return None
        adapter = load_adapter(adapter_spec.reference, dict(adapter_spec.parameters))
        metadata = adapter.metadata()
        if metadata.get("dataset_fingerprint") != self.config.dataset_fingerprint:
            raise ValueError("dataset adapter fingerprint differs from frozen plan")
        window = self.config.windows.research_feedback
        self._dataset = adapter.load_period("research_feedback", window.start, window.end)
        unknown = set(self._dataset.fields) - set(self.config.allowed_fields)
        if unknown:
            raise ValueError(f"dataset adapter returned unregistered fields: {sorted(unknown)}")
        return self._dataset

    def _agent_context(self) -> dict[str, Any]:
        feedback = self._read_jsonl("experiment_feedback.jsonl")
        accepted = self._read_jsonl("accepted_factors.jsonl")
        # This method is the only context constructor. It intentionally has no
        # path for diagnostic/reference artifacts.
        return {
            "contract_version": "2.2",
            "campaign_id": self.config.campaign_id,
            "pool_id": self.config.pool_id,
            "feedback_window": "research_feedback",
            "allowed_fields": list(self.config.allowed_fields),
            "baseline_factor_ids": list(self.config.baseline_factor_ids),
            "label_contract": self.config.label_contract.model_dump(mode="json"),
            "strategy_contract": self.config.strategy_contract.model_dump(mode="json"),
            "accepted_factors": accepted[-200:],
            "rejected_proposals": [
                item for item in self._read_jsonl("factor_proposals.jsonl")
                if item.get("status") == "rejected"
            ][-100:],
            "experiment_feedback": feedback[-60:],
        }

    def _make_client(self) -> AgentClient:
        settings = getattr(self.config, "agent", None)
        replay_path = getattr(settings, "replay_path", None)
        provider = getattr(settings, "provider", None)
        if provider == "replay" or replay_path:
            if replay_path is None:
                raise ValueError("replay provider requires replay_path")
            return ReplayAgentClient.from_path(replay_path)
        return _IsolatedAgentClient(self.plan, self.run_dir)

    def _make_executor(self) -> ZyQuantExperimentExecutor:
        settings = getattr(self.config, "experiment_executor", None)
        argv = getattr(settings, "argv", None)
        timeout = getattr(settings, "timeout_seconds", 86_400)
        return ZyQuantExperimentExecutor(
            self.run_dir, worker_argv=argv, timeout_seconds=timeout
        )

    def _resource_preflight(self) -> None:
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        resources = getattr(self.config, "resources", None)
        minimum_disk = float(getattr(resources, "minimum_free_gb", 0.0))
        free_disk = shutil.disk_usage(self.run_dir.parent).free / (1024 ** 3)
        if free_disk < minimum_disk:
            raise RuntimeError(
                f"insufficient free disk: {free_disk:.1f} GiB < {minimum_disk:.1f} GiB"
            )
        minimum_memory = float(getattr(resources, "minimum_free_memory_gb", 0.0))
        available = _available_memory_gb()
        if available is not None and available < minimum_memory:
            raise RuntimeError(
                f"insufficient free memory: {available:.1f} GiB < {minimum_memory:.1f} GiB"
            )

    def _lock_pareto(self, state: CampaignState) -> None:
        del state
        rows = self._read_jsonl("experiment_feedback.jsonl")
        eligible = [row for row in rows if (
            (row.get("median_annual_incremental_active_return") or 0.0) > 0
            and (row.get("positive_year_share") or 0.0)
                >= self.config.minimum_positive_year_share
            and (row.get("maximum_drawdown_degradation") or 0.0)
                <= self.config.maximum_drawdown_degradation
        )]
        eligible.sort(key=lambda row: (
            row.get("median_annual_incremental_active_return") or 0.0,
            row.get("positive_year_share") or 0.0,
        ), reverse=True)
        ids = [row["experiment_id"] for row in eligible[:10]]
        atomic_write_json(self.run_dir / "PARETO_LOCK.json", {
            "plan_id": self.plan.plan_id,
            "experiment_ids": ids,
            "selection_window": "research_feedback",
            "diagnostic_feedback_allowed": False,
            "reference_feedback_allowed": False,
        }, immutable=True)

    def _result_manifest(self, payload: dict[str, Any]) -> str:
        artifacts = {}
        for path in sorted(self.run_dir.rglob("*")):
            if path.is_file() and path.name not in {"SUCCESS.json", "REPORT.md"}:
                artifacts[str(path.relative_to(self.run_dir))] = sha256_file(path)
        manifest = {
            "schema_version": "2.2",
            "plan_id": self.plan.plan_id,
            "result": redact_payload(payload),
            "artifacts": artifacts,
        }
        atomic_write_json(self.run_dir / "result_manifest.json", manifest, immutable=True)
        return sha256_file(self.run_dir / "result_manifest.json")

    def _verify_identity(self) -> None:
        if not self.run_dir.exists():
            raise ValueError(f"run directory does not exist: {self.run_dir}")
        stored = AgentCampaignPlan.model_validate_json(
            (self.run_dir / "plan.json").read_text(encoding="utf-8")
        )
        if stored.plan_id != self.plan.plan_id:
            raise ValueError("run directory belongs to a different frozen plan")

    def _load_state(self) -> CampaignState:
        path = self.run_dir / "state.json"
        if not path.exists():
            state = CampaignState()
        else:
            state = CampaignState(**json.loads(path.read_text(encoding="utf-8")))
        proposal_count = len(self._read_jsonl("factor_proposals.jsonl"))
        accepted_without_proposal = len(
            set(self._accepted())
            - {
                FactorCandidateSpec.model_validate(row["candidate"]).factor_id
                for row in self._read_jsonl("factor_proposals.jsonl")
                if "candidate" in row
            }
        )
        state.raw_proposals = max(
            state.raw_proposals, proposal_count + accepted_without_proposal
        )
        state.accepted_unique_factors = len(self._accepted())
        return state

    def _save_state(self, state: CampaignState) -> None:
        destination = self.run_dir / "state.json"
        temporary = destination.with_name(".state.json.tmp")
        temporary.write_text(json.dumps(asdict(state), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)

    def _started_at(self) -> datetime:
        path = self.run_dir / "STARTED.json"
        if path.exists():
            return datetime.fromisoformat(json.loads(path.read_text())["started_at"])
        started = datetime.now(timezone.utc)
        atomic_write_json(path, {"started_at": started.isoformat()}, immutable=True)
        return started

    def _accepted(self) -> dict[str, FactorCandidateSpec]:
        result = {}
        for row in self._read_jsonl("accepted_factors.jsonl"):
            result[row["factor_id"]] = FactorCandidateSpec.model_validate(row["candidate"])
        return result

    def _pareto_ids(self) -> list[str]:
        path = self.run_dir / "PARETO_LOCK.json"
        return json.loads(path.read_text(encoding="utf-8"))["experiment_ids"] if path.exists() else []

    def _append_jsonl(self, name: str, payload: dict[str, Any]) -> None:
        with (self.run_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(redact_payload(payload), ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _read_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.run_dir / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_agent_campaign(
    plan_dir: str | Path, output_dir: str | Path, *,
    client: AgentClient | None = None,
    executor: ZyQuantExperimentExecutor | None = None,
) -> AgentCampaignResult:
    return AgentCampaignRunner(
        load_agent_plan(plan_dir), output_dir, client=client, executor=executor
    ).run()


def resume_agent_campaign(
    run_dir: str | Path, *, client: AgentClient | None = None,
    executor: ZyQuantExperimentExecutor | None = None,
) -> AgentCampaignResult:
    root = Path(run_dir).expanduser().resolve()
    plan = AgentCampaignPlan.model_validate_json((root / "plan.json").read_text(encoding="utf-8"))
    return AgentCampaignRunner(plan, root, client=client, executor=executor).run(resume=True)


def _available_memory_gb() -> float | None:
    path = Path("/proc/meminfo")
    if path.exists():
        values = {}
        for line in path.read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = float(value.strip().split()[0])
        return values.get("MemAvailable", 0.0) / (1024 ** 2)
    return None


def _median_cross_sectional_spearman(
    frame: pd.DataFrame, left: str, right: str
) -> float:
    correlations = []
    for _, group in frame.groupby("trade_date", sort=False):
        valid = group[[left, right]].dropna()
        if len(valid) < 3 or valid[left].nunique() < 2 or valid[right].nunique() < 2:
            continue
        value = valid[left].corr(valid[right], method="spearman")
        if pd.notna(value):
            correlations.append(abs(float(value)))
    return float(np.median(correlations)) if correlations else 0.0


def _require_new_or_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory must not exist or be empty: {path}")


__all__ = [
    "AgentCampaignRunner",
    "AgentClient",
    "CampaignState",
    "create_agent_plan",
    "load_agent_config",
    "load_agent_plan",
    "resume_agent_campaign",
    "run_agent_campaign",
]
