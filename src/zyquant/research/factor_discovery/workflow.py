from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from zyquant import __version__

from .adapter import DiscoveryDataset, load_adapter
from .evaluator import ExpressionEvaluator
from .evidence import EvidenceResult, RollingEvidenceEngine
from .llm import SYSTEM_PROMPT, LLMClient, ReplayLLMClient, make_llm_client
from .models import (
    CandidateSpec,
    DiscoveryPlan,
    DiscoveryRunResult,
    ExpressionSpec,
    FactorDiscoveryConfig,
    Operator,
)


PROMPT_VERSION = "factor-discovery-v1"


@dataclass
class _CampaignState:
    round_number: int = 0
    raw_proposals: int = 0
    accepted: int = 0
    phase: str = "generation"


def _hash_json(payload: Any) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(value.encode()).hexdigest()


def load_discovery_config(path: str | Path) -> FactorDiscoveryConfig:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("factor discovery config must contain a mapping")
    return FactorDiscoveryConfig.model_validate(payload)


def create_discovery_plan(
    config_path: str | Path,
    output_dir: str | Path,
) -> DiscoveryPlan:
    config = load_discovery_config(config_path)
    adapter = load_adapter(config.adapter, config.adapter_parameters)
    metadata = adapter.metadata()
    if metadata.get("dataset_fingerprint") != config.dataset_fingerprint:
        raise ValueError("adapter dataset fingerprint does not match frozen config")
    if metadata.get("pool_id") != config.pool_id:
        raise ValueError("adapter pool_id does not match frozen config")
    expression_schema = ExpressionSpec.model_json_schema()
    prompt_hash = hashlib.sha256(
        f"{PROMPT_VERSION}\n{SYSTEM_PROMPT}".encode()
    ).hexdigest()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    plan = DiscoveryPlan.create(
        config,
        prompt_hash=prompt_hash,
        expression_schema_hash=_hash_json(expression_schema),
        git_commit=git_commit,
        package_version=__version__,
        adapter_metadata=metadata,
    )
    destination = Path(output_dir).expanduser().resolve()
    _require_new_or_empty(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (staging / "frozen_spec.json").write_text(
            plan.model_dump_json(indent=2), encoding="utf-8"
        )
        manifest = {
            "plan_id": plan.plan_id,
            "config_hash": plan.config_hash,
            "dataset_id": config.dataset_id,
            "dataset_fingerprint": config.dataset_fingerprint,
            "pool_id": config.pool_id,
            "adapter_metadata_hash": _hash_json(metadata),
            "thresholds_confirmed": True,
        }
        (staging / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pd.DataFrame([{
            "pool_id": config.pool_id,
            "target_candidates": config.target_candidates,
            "max_proposals": config.max_proposals,
            "preferred_group_count": config.evidence.preferred_group_count,
            "fallback_group_count": config.evidence.fallback_group_count,
            "minimum_pool_size": config.evidence.minimum_pool_size,
            "thresholds_confirmed": True,
        }]).to_csv(staging / "structural_feasibility.csv", index=False)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return plan


def load_discovery_plan(plan_dir: str | Path) -> DiscoveryPlan:
    path = Path(plan_dir).expanduser().resolve() / "frozen_spec.json"
    plan = DiscoveryPlan.model_validate_json(path.read_text(encoding="utf-8"))
    actual = _hash_json(plan.config.model_dump(mode="json"))
    if actual != plan.config_hash:
        raise ValueError("frozen discovery plan config hash mismatch")
    return plan


class FactorDiscoveryRunner:
    def __init__(self, plan: DiscoveryPlan, run_dir: str | Path, llm: LLMClient | None = None):
        self.plan = plan
        self.config = plan.config
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.llm = llm
        self.adapter = load_adapter(self.config.adapter, self.config.adapter_parameters)
        self.evaluator = ExpressionEvaluator(set(self.config.allowed_fields))
        self.evidence = RollingEvidenceEngine(self.config.evidence)

    def run(self, *, resume: bool = False) -> DiscoveryRunResult:
        self._disk_preflight()
        if resume:
            if not self.run_dir.exists():
                raise ValueError(f"run directory does not exist: {self.run_dir}")
            self._verify_run_identity()
        else:
            _require_new_or_empty(self.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "plan.json").write_text(
                self.plan.model_dump_json(indent=2), encoding="utf-8"
            )
        state = self._load_state()
        accepted = self._load_accepted()
        development: DiscoveryDataset | None = None
        if state.phase == "generation":
            development = self._load_period("development")
            llm = self.llm or make_llm_client(self.config.llm)
            if self.llm is None and isinstance(llm, ReplayLLMClient):
                llm.seek(self._jsonl_count(self.run_dir / "llm_calls.jsonl"))
            self._generate(llm, development, state, accepted)
            state.phase = "locked"
            self._save_state(state)
        if state.phase == "locked":
            for period in ("development", "diagnostic", "holdout"):
                dataset = development if period == "development" and development is not None else (
                    self._load_period(period)
                )
                self._evaluate_period(period, dataset, accepted)
            state.phase = "complete"
            self._save_state(state)
        status = "complete" if len(accepted) >= self.config.target_candidates else "incomplete"
        result = DiscoveryRunResult(
            campaign_id=self.config.campaign_id,
            plan_id=self.plan.plan_id,
            status=status,
            accepted_candidates=len(accepted),
            raw_proposals=state.raw_proposals,
            rounds=state.round_number,
            run_dir=self.run_dir,
            period_outputs={name: str(self.run_dir / name) for name in (
                "development", "diagnostic", "holdout"
            )},
        )
        (self.run_dir / "SUCCESS.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        write_discovery_report(self.run_dir)
        return result

    def _generate(
        self,
        llm: LLMClient,
        development: DiscoveryDataset,
        state: _CampaignState,
        accepted: dict[str, CandidateSpec],
    ) -> None:
        feedback = self._feedback_from_existing()
        while (
            len(accepted) < self.config.target_candidates
            and state.round_number < self.config.max_rounds
            and state.raw_proposals < self.config.max_proposals
        ):
            state.round_number += 1
            remaining = min(
                self.config.proposals_per_call,
                self.config.max_proposals - state.raw_proposals,
            )
            prompt = self._prompt(state.round_number, remaining, feedback, accepted)
            call = llm.propose(
                prompt=prompt, count=remaining, round_number=state.round_number
            )
            audit = {**call.audit, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()}
            self._append_jsonl(self.run_dir / "llm_calls.jsonl", audit)
            round_evidence: list[EvidenceResult] = []
            round_samples: list[pd.DataFrame] = []
            for candidate in call.batch.candidates:
                if state.raw_proposals >= self.config.max_proposals:
                    break
                state.raw_proposals += 1
                record: dict[str, Any] = {
                    "round": state.round_number,
                    "proposal_number": state.raw_proposals,
                    "candidate": candidate.model_dump(mode="json"),
                }
                expression_id = candidate.expression.expression_id
                try:
                    candidate.expression.validate_contract(set(self.config.allowed_fields))
                    if expression_id in accepted:
                        raise ValueError("exact_canonical_duplicate")
                    values = self._candidate_values(candidate, development)
                    finite = values["factor_value"].dropna()
                    if finite.empty or not np.isfinite(finite).all():
                        raise ValueError("candidate_has_no_finite_values")
                    candidate_evidence = self.evidence.evaluate(
                        development.observations,
                        values,
                        self._factor_metadata([candidate]),
                        development.label_metadata,
                        development.exposures,
                    )
                    accepted[expression_id] = candidate
                    state.accepted = len(accepted)
                    record.update({"status": "accepted", "factor_id": expression_id})
                    self._append_jsonl(self.run_dir / "accepted.jsonl", {
                        "factor_id": expression_id,
                        "candidate": candidate.model_dump(mode="json"),
                    })
                    factor_dir = self.run_dir / "factor_shards"
                    factor_dir.mkdir(exist_ok=True)
                    values.to_parquet(factor_dir / f"{expression_id}.parquet", index=False)
                    round_evidence.append(candidate_evidence)
                    round_samples.append(self._sample_factor_values(values))
                except Exception as exc:
                    record.update({"status": "rejected", "reason": str(exc),
                                   "error_type": type(exc).__name__})
                self._append_jsonl(self.run_dir / "proposals.jsonl", record)
                self._save_state(state)
                if len(accepted) >= self.config.target_candidates:
                    break
            if round_evidence:
                evidence = self.evidence.combine(
                    round_evidence,
                    pd.concat(round_samples, ignore_index=True),
                )
                feedback = self._feedback(evidence)
                self._append_jsonl(self.run_dir / "round_feedback.jsonl", {
                    "round": state.round_number,
                    "development_only": True,
                    "feedback": feedback,
                })
            self._save_state(state)

    def _candidate_values(
        self, candidate: CandidateSpec, dataset: DiscoveryDataset
    ) -> pd.DataFrame:
        index = dataset.observations[["decision_time", "instrument_id"]].rename(
            columns={"decision_time": "trade_date"}
        ).drop_duplicates()
        values = self.evaluator.evaluate(candidate.expression, dataset.fields, index)
        observations = dataset.observations[
            ["observation_id", "decision_time", "instrument_id"]
        ]
        values = values.merge(
            observations,
            left_on=["trade_date", "instrument_id"],
            right_on=["decision_time", "instrument_id"],
            how="inner",
            validate="one_to_one",
        )
        return pd.DataFrame({
            "observation_id": values["observation_id"],
            "factor_id": candidate.expression.expression_id,
            "factor_value": values["value"],
            "factor_known_at": values["decision_time"],
        })

    def _evaluate_period(
        self,
        period: str,
        dataset: DiscoveryDataset,
        accepted: dict[str, CandidateSpec],
    ) -> None:
        destination = self.run_dir / period
        if (destination / "SUCCESS.json").exists():
            return
        _require_new_or_empty(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.", dir=destination.parent
        ))
        factor_dir = staging / "factor_values"
        factor_dir.mkdir()
        evidence_results = []
        samples = []
        hashes = {}
        for factor_id, candidate in accepted.items():
            cached = self.run_dir / "factor_shards" / f"{factor_id}.parquet"
            values = (
                pd.read_parquet(cached)
                if period == "development" and cached.exists()
                else self._candidate_values(candidate, dataset)
            )
            shard = factor_dir / f"{factor_id}.parquet"
            values.to_parquet(shard, index=False, compression="zstd")
            hashes[factor_id] = _file_hash(shard)
            evidence_results.append(self.evidence.evaluate(
                dataset.observations,
                values,
                self._factor_metadata([candidate]),
                dataset.label_metadata,
                dataset.exposures,
            ))
            samples.append(self._sample_factor_values(values))
        result = self.evidence.combine(
            evidence_results,
            pd.concat(samples, ignore_index=True) if samples else None,
        )
        outputs = {
            "factor_health.csv": result.factor_health,
            "daily_factor_metrics.parquet": result.daily_metrics,
            "factor_summary.csv": result.factor_summary,
            "period_stability.csv": result.period_stability,
            "exposure_diagnostics.csv": result.exposure_diagnostics,
            "failure_records.csv": result.failure_records,
            "redundancy_pairs.csv": result.redundancy_pairs,
        }
        for name, frame in outputs.items():
            if name.endswith(".parquet"):
                frame.to_parquet(staging / name, index=False)
            else:
                frame.to_csv(staging / name, index=False)
        (staging / "SUCCESS.json").write_text(json.dumps({
            "period": period,
            "plan_id": self.plan.plan_id,
            "dataset_fingerprint": self.config.dataset_fingerprint,
            "candidate_count": len(accepted),
            "factor_value_shards": hashes,
        }, indent=2), encoding="utf-8")
        os.replace(staging, destination)

    def _sample_factor_values(self, values: pd.DataFrame) -> pd.DataFrame:
        sample = (
            values.sample(n=25_000, random_state=self.config.random_seed)
            if len(values) > 25_000 else values
        )
        return sample[["observation_id", "factor_id", "factor_value"]].copy()

    def _load_period(self, period: str) -> DiscoveryDataset:
        window = getattr(self.config.windows, period)
        dataset = self.adapter.load_period(period, window.start, window.end)
        observations = dataset.observations.copy()
        decision = pd.to_datetime(observations["decision_time"]).dt.date
        if not decision.between(window.start, window.end).all():
            raise ValueError(f"adapter returned {period} observations outside frozen window")
        unknown = set(dataset.fields) - set(self.config.allowed_fields)
        if unknown:
            raise ValueError(f"adapter returned unregistered fields: {sorted(unknown)}")
        missing = set(self.config.allowed_fields) - set(dataset.fields)
        if missing:
            raise ValueError(f"adapter omitted allowed fields: {sorted(missing)}")
        if dataset.label_metadata.empty:
            raise ValueError("adapter returned empty label_metadata")
        label = dataset.label_metadata.iloc[0].to_dict()
        mismatches = {
            key: (label.get(key), expected)
            for key, expected in self.config.label_contract.items()
            if label.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"adapter label contract differs from frozen plan: {mismatches}")
        return dataset

    def _prompt(
        self, round_number: int, count: int, feedback: dict[str, Any],
        accepted: dict[str, CandidateSpec],
    ) -> str:
        operators = [item.value for item in Operator]
        payload = {
            "prompt_version": PROMPT_VERSION,
            "campaign_id": self.config.campaign_id,
            "pool_id": self.config.pool_id,
            "pool_description": self.config.pool_description,
            "round": round_number,
            "requested_candidates": count,
            "allowed_fields": list(self.config.allowed_fields),
            "allowed_operators": operators,
            "limits": {"depth": 8, "nodes": 64, "lookback": 252},
            "existing_expression_ids": sorted(accepted)[-200:],
            "development_feedback": feedback,
            "important": "Feedback contains development-period aggregates only.",
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _feedback(result: EvidenceResult) -> dict[str, Any]:
        summary = result.factor_summary
        health = result.factor_health
        return {
            "tested": len(health),
            "median_non_null_ratio": (
                float(health["non_null_ratio"].median()) if not health.empty else None
            ),
            "median_rank_ic": (
                float(summary["rank_ic_mean"].median()) if not summary.empty else None
            ),
            "median_spread": (
                float(summary["spread_mean"].median()) if not summary.empty else None
            ),
            "model_candidates": (
                int(summary["model_candidate"].sum()) if not summary.empty else 0
            ),
            "failure_reasons": result.failure_records.get(
                "reason", pd.Series(dtype=str)
            ).value_counts().head(10).to_dict(),
            "redundancy_pairs": len(result.redundancy_pairs),
        }

    def _feedback_from_existing(self) -> dict[str, Any]:
        path = self.run_dir / "round_feedback.jsonl"
        if not path.exists():
            return {}
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        return json.loads(lines[-1])["feedback"] if lines else {}

    @staticmethod
    def _factor_metadata(candidates: list[CandidateSpec]) -> pd.DataFrame:
        return pd.DataFrame({
            "factor_id": [item.expression.expression_id for item in candidates],
            "factor_type": "continuous",
        })

    def _disk_preflight(self) -> None:
        target = self.run_dir.parent
        target.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(target).free / (1024 ** 3)
        if free_gb < self.config.minimum_free_gb:
            raise RuntimeError(
                f"insufficient free disk: {free_gb:.1f} GiB < "
                f"{self.config.minimum_free_gb:.1f} GiB"
            )

    def _verify_run_identity(self) -> None:
        stored = DiscoveryPlan.model_validate_json(
            (self.run_dir / "plan.json").read_text(encoding="utf-8")
        )
        if stored.plan_id != self.plan.plan_id:
            raise ValueError("run directory belongs to a different frozen plan")

    def _load_state(self) -> _CampaignState:
        path = self.run_dir / "state.json"
        if not path.exists():
            return _CampaignState()
        state = _CampaignState(**json.loads(path.read_text(encoding="utf-8")))
        proposals = self.run_dir / "proposals.jsonl"
        if proposals.exists():
            records = [json.loads(line) for line in proposals.read_text(
                encoding="utf-8"
            ).splitlines() if line]
            if records:
                state.raw_proposals = max(
                    state.raw_proposals, max(int(item["proposal_number"]) for item in records)
                )
                state.round_number = max(
                    state.round_number, max(int(item["round"]) for item in records)
                )
        state.accepted = len(self._load_accepted())
        return state

    def _save_state(self, state: _CampaignState) -> None:
        temporary = self.run_dir / ".state.json.tmp"
        temporary.write_text(json.dumps(state.__dict__, indent=2), encoding="utf-8")
        os.replace(temporary, self.run_dir / "state.json")

    def _load_accepted(self) -> dict[str, CandidateSpec]:
        path = self.run_dir / "accepted.jsonl"
        if not path.exists():
            return {}
        accepted = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                payload = json.loads(line)
                accepted[payload["factor_id"]] = CandidateSpec.model_validate(
                    payload["candidate"]
                )
        return accepted

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _jsonl_count(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def run_discovery(
    plan_dir: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    llm: LLMClient | None = None,
) -> DiscoveryRunResult:
    return FactorDiscoveryRunner(
        load_discovery_plan(plan_dir), output_dir, llm=llm
    ).run(resume=resume)


def write_discovery_report(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    success_path = root / "SUCCESS.json"
    success = json.loads(success_path.read_text(encoding="utf-8")) if success_path.exists() else {}
    rows = [
        "# Factor Discovery Report",
        "",
        f"- Campaign: `{success.get('campaign_id', 'running')}`",
        f"- Status: `{success.get('status', 'running')}`",
        f"- Accepted candidates: {success.get('accepted_candidates', 0)}",
        f"- Raw proposals: {success.get('raw_proposals', 0)}",
        "- Holdout results were not supplied to the LLM.",
        "- High-correlation factors are diagnostics only and remain in the denominator.",
        "",
        "## Periods",
        "",
    ]
    for period in ("development", "diagnostic", "holdout"):
        summary_path = root / period / "factor_summary.csv"
        if not summary_path.exists():
            rows.append(f"- {period}: not completed")
            continue
        summary = pd.read_csv(summary_path)
        economic = int(summary.get("economic_eligible", pd.Series(dtype=bool)).sum())
        model = int(summary.get("model_candidate", pd.Series(dtype=bool)).sum())
        rows.append(
            f"- {period}: {len(summary)} tested, {model} model candidates, "
            f"{economic} economic eligible"
        )
    destination = root / "REPORT.md"
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return destination


def _require_new_or_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory must not exist or be empty: {path}")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
