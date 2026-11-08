from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zyquant.research.factor_discovery.models import EvidenceRecipe, ExpressionSpec


PoolId = Literal["CSI300", "CSI1000"]
CampaignStatus = Literal["planned", "running", "complete", "incomplete", "failed"]
FactorDisposition = Literal[
    "stable_core",
    "vintage_only",
    "rejected",
    "evidence_only",
    "model_candidate",
    "economic_eligible",
]

_FIXED_SEEDS: dict[str, tuple[int, ...]] = {
    "CSI300": (20260722, 20260829, 20260916),
    "CSI1000": (20260722, 20260723, 20260724),
}
_SAFE_PLACEHOLDERS = {
    "config_path", "experiment_path", "output_dir", "plan_path", "request", "run_dir", "spec_path"
}
_SECRET_OR_CODE_KEYS = re.compile(
    r"(^|_)(api_?key|access_?token|password|secret|source_?code|python_?code|script|shell|command)($|_)",
    re.IGNORECASE,
)


def _sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class DateWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> "DateWindow":
        if self.start > self.end:
            raise ValueError("window start must not exceed end")
        return self


class AgentResearchWindows(BaseModel):
    """Partitions with a single feedback-eligible window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    research_feedback: DateWindow
    diagnostic: DateWindow
    reference: DateWindow
    prospective_start: date

    @model_validator(mode="after")
    def _ordered_and_disjoint(self) -> "AgentResearchWindows":
        if not (
            self.research_feedback.end < self.diagnostic.start
            and self.diagnostic.end < self.reference.start
            and self.reference.end < self.prospective_start
        ):
            raise ValueError(
                "research_feedback, diagnostic, reference, and prospective_start "
                "must be ordered and disjoint"
            )
        return self

    def is_feedback_eligible(self, window: Literal["research_feedback", "diagnostic", "reference"]) -> bool:
        return window == "research_feedback"


class AgentProviderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["rdagent", "replay"] = "rdagent"
    model: str = Field(default="gpt-5.6-sol", min_length=1, max_length=200)
    base_url: str | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "medium"
    timeout_seconds: int = Field(default=180, ge=1, le=600)
    max_retries: int = Field(default=3, ge=0, le=8)
    replay_path: Path | None = None

    @model_validator(mode="after")
    def _provider_contract(self) -> "AgentProviderSettings":
        if self.provider == "replay" and self.replay_path is None:
            raise ValueError("replay provider requires replay_path")
        if self.provider == "rdagent" and self.replay_path is not None:
            raise ValueError("replay_path is only valid for replay provider")
        return self


def _reject_secret_or_code_keys(value: Any, *, path: str = "parameters") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET_OR_CODE_KEYS.search(str(key)):
                raise ValueError(f"{path} contains forbidden secret/code key: {key}")
            _reject_secret_or_code_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_secret_or_code_keys(item, path=f"{path}[{index}]")


class DatasetAdapterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _safe_parameters(self) -> "DatasetAdapterSpec":
        _reject_secret_or_code_keys(self.parameters)
        return self


class ExecutorSpec(BaseModel):
    """Direct argv execution contract; it is never passed through a shell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    argv: tuple[str, ...] = Field(min_length=1, max_length=64)
    timeout_seconds: int = Field(default=3600, ge=1, le=604800)

    @model_validator(mode="after")
    def _safe_argv(self) -> "ExecutorSpec":
        executable = Path(self.argv[0]).name.lower()
        if executable in {"sh", "bash", "zsh", "fish", "cmd", "powershell", "pwsh"}:
            raise ValueError("shell executables are forbidden")
        for argument in self.argv:
            if not argument or any(char in argument for char in ";|&><`\n\r$()"):
                raise ValueError("argv contains empty value or forbidden shell syntax")
            placeholders = set(re.findall(r"\{([^{}]+)\}", argument))
            unknown = placeholders - _SAFE_PLACEHOLDERS
            if unknown:
                raise ValueError(f"argv contains unsupported placeholders: {sorted(unknown)}")
            if argument.count("{") != argument.count("}"):
                raise ValueError("argv contains malformed placeholder")
        return self


class ResourceGuards(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_free_gb: float = Field(default=100.0, ge=0.0)
    minimum_free_memory_gb: float = Field(default=50.0, ge=0.0)
    factor_workers: int = Field(default=12, ge=1, le=24)
    model_workers: int = Field(default=3, ge=1, le=4)
    threads_per_model: int = Field(default=16, ge=1, le=20)


class StageBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_target: int = Field(default=200, ge=1, le=2000)
    max_factor_rounds: int = Field(default=100, ge=1, le=100)
    proposals_per_round: int = Field(default=4, ge=1, le=4)
    max_raw_proposals: int = Field(default=400, ge=1, le=400)
    max_repairs: int = Field(default=3, ge=0, le=3)
    max_incremental_experiments: int = Field(default=30, ge=0, le=30)
    max_joint_experiments: int = Field(default=30, ge=0, le=30)

    @model_validator(mode="after")
    def _budget_contract(self) -> "StageBudgets":
        if self.factor_target > self.max_raw_proposals:
            raise ValueError("factor_target must not exceed max_raw_proposals")
        return self


class CostModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, max_length=120)
    buy_cost_bps: float = Field(ge=0.0, le=1000.0)
    sell_cost_bps: float = Field(ge=0.0, le=1000.0)
    slippage_bps: float = Field(default=0.0, ge=0.0, le=1000.0)


class LabelContract(BaseModel):
    """Immutable economic-label timing and cost semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_schedule: str = Field(min_length=3, max_length=500)
    decision_interval_observations: int = Field(ge=1, le=252)
    entry_rule: str = Field(min_length=3, max_length=1000)
    exit_rule: str = Field(min_length=3, max_length=1000)
    holding_horizon_observations: int = Field(ge=1, le=252)
    execution_delay_observations: int = Field(ge=0, le=20)
    cost_model: CostModel
    label_is_net_of_costs: Literal[True] = True
    overlapping_positions: bool

    @model_validator(mode="after")
    def _overlap_contract(self) -> "LabelContract":
        expected_overlap = (
            self.holding_horizon_observations > self.decision_interval_observations
        )
        if self.overlapping_positions != expected_overlap:
            raise ValueError(
                "overlapping_positions must agree with holding horizon and decision interval"
            )
        return self


class StrategyContract(BaseModel):
    """Frozen formal-account semantics; the agent cannot rewrite this contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str = Field(min_length=3, pattern=r"^[A-Za-z0-9._-]+$")
    candidate_pool: PoolId
    benchmark_id: str = Field(min_length=1, max_length=120)
    decision_rule: str = Field(min_length=8, max_length=4000)
    account_adapter: str = Field(min_length=3, max_length=240)
    formal_execution_description: str = Field(min_length=20, max_length=4000)
    frozen_parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _safe_frozen_parameters(self) -> "StrategyContract":
        _reject_secret_or_code_keys(self.frozen_parameters, path="frozen_parameters")
        return self


class AgentResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.2"] = "2.2"
    campaign_id: str = Field(min_length=3, pattern=r"^[A-Za-z0-9._-]+$")
    pool_id: PoolId
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=8)
    dataset_adapter: DatasetAdapterSpec
    label_contract: LabelContract
    strategy_contract: StrategyContract
    baseline_factor_ids: tuple[str, ...] = Field(min_length=1)
    allowed_fields: tuple[str, ...] = Field(min_length=1)
    windows: AgentResearchWindows
    seeds: tuple[int, ...]
    agent: AgentProviderSettings = Field(default_factory=AgentProviderSettings)
    agent_executor: ExecutorSpec
    experiment_executor: ExecutorSpec
    account_adapter: ExecutorSpec | None = None
    resources: ResourceGuards
    budgets: StageBudgets = Field(default_factory=StageBudgets)
    evidence: EvidenceRecipe = Field(default_factory=EvidenceRecipe)
    exact_duplicate_correlation: float = Field(default=0.99, ge=0.99, le=1.0)
    redundant_correlation: float = Field(default=0.95, ge=0.9, lt=0.99)
    minimum_positive_year_share: float = Field(default=0.6, ge=0.5, le=1.0)
    maximum_drawdown_degradation: float = Field(default=0.02, ge=0.0, le=0.2)
    random_seed: int = 20261108

    @model_validator(mode="after")
    def _contract(self) -> "AgentResearchConfig":
        if self.seeds != _FIXED_SEEDS[self.pool_id]:
            raise ValueError(f"{self.pool_id} seeds must be {_FIXED_SEEDS[self.pool_id]}")
        if self.strategy_contract.candidate_pool != self.pool_id:
            raise ValueError("strategy candidate_pool must match config pool_id")
        if len(set(self.allowed_fields)) != len(self.allowed_fields):
            raise ValueError("allowed_fields must be unique")
        if len(set(self.baseline_factor_ids)) != len(self.baseline_factor_ids):
            raise ValueError("baseline_factor_ids must be unique")
        if self.redundant_correlation >= self.exact_duplicate_correlation:
            raise ValueError("redundant_correlation must be below exact_duplicate_correlation")
        if self.pool_id == "CSI300" and (
            self.resources.minimum_free_gb < 100
            or self.resources.factor_workers > 12
            or self.resources.model_workers > 3
            or self.resources.threads_per_model > 20
        ):
            raise ValueError("CSI300 resources exceed the registered disk/12/3/20 limits")
        if self.pool_id == "CSI1000" and (
            self.resources.minimum_free_gb < 200
            or self.resources.minimum_free_memory_gb < 50
            or self.resources.factor_workers > 24
            or self.resources.model_workers > 4
            or self.resources.threads_per_model > 16
        ):
            raise ValueError("CSI1000 resources exceed the registered disk/memory/24/4/16 limits")
        return self


class ResearchHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str = Field(min_length=8, pattern=r"^[A-Za-z0-9._-]+$")
    iteration: int = Field(ge=1, le=100)
    title: str = Field(min_length=8, max_length=240)
    thesis: str = Field(min_length=20, max_length=4000)
    economic_mechanism: str = Field(min_length=20, max_length=4000)
    expected_direction: Literal[-1, 1]
    candidate_fields: tuple[str, ...] = Field(min_length=1)
    parent_hypothesis_ids: tuple[str, ...] = ()


class FactorCandidateSpec(BaseModel):
    """A declarative factor candidate; arbitrary executable code has no field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_name: str = Field(min_length=3, max_length=120, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    hypothesis_id: str = Field(min_length=8)
    expression: ExpressionSpec
    direction: Literal[-1, 1]
    rationale: str = Field(min_length=20, max_length=4000)
    repair_attempt: int = Field(default=0, ge=0, le=3)

    @property
    def factor_id(self) -> str:
        return self.expression.expression_id


class ModelRecipeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pool_id: PoolId
    model_type: Literal["XGBRanker", "XGBRegressor"]
    objective: Literal["rank:ndcg", "rank:pairwise", "reg:squarederror", "reg:pseudohubererror"]
    training_window: Literal["3y", "5y", "expanding"]
    label_transform: Literal[
        "none", "cross_sectional_rank", "cross_sectional_zscore", "winsorized_zscore"
    ] = "none"
    seeds: tuple[int, ...]
    learning_rate: float = Field(default=0.03, ge=0.001, le=0.3)
    max_depth: int = Field(default=5, ge=2, le=12)
    min_child_weight: float = Field(default=10.0, ge=0.0, le=1000.0)
    subsample: float = Field(default=0.8, ge=0.5, le=1.0)
    colsample_bytree: float = Field(default=0.7, ge=0.5, le=1.0)
    reg_alpha: float = Field(default=1.0, ge=0.0, le=1000.0)
    reg_lambda: float = Field(default=10.0, ge=0.0, le=1000.0)
    max_rounds: int = Field(default=2000, ge=10, le=10000)
    early_stopping_rounds: int = Field(default=100, ge=1, le=1000)
    n_jobs: int = Field(default=16, ge=1, le=64)

    @model_validator(mode="after")
    def _model_contract(self) -> "ModelRecipeSpec":
        if self.seeds != _FIXED_SEEDS[self.pool_id]:
            raise ValueError(f"{self.pool_id} seeds must be {_FIXED_SEEDS[self.pool_id]}")
        rank_objectives = {"rank:ndcg", "rank:pairwise"}
        if self.model_type == "XGBRanker" and self.objective not in rank_objectives:
            raise ValueError("XGBRanker requires a ranking objective")
        if self.model_type == "XGBRegressor" and self.objective in rank_objectives:
            raise ValueError("XGBRegressor requires a regression objective")
        if self.early_stopping_rounds >= self.max_rounds:
            raise ValueError("early_stopping_rounds must be below max_rounds")
        return self


class JointExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.2"] = "2.2"
    campaign_id: str = Field(min_length=3)
    hypothesis_id: str = Field(min_length=8)
    experiment_kind: Literal["incremental", "joint"]
    pool_id: PoolId
    baseline_factor_ids: tuple[str, ...] = Field(min_length=1)
    candidate_factors: tuple[FactorCandidateSpec, ...] = Field(min_length=1, max_length=12)
    model_recipe: ModelRecipeSpec
    prediction_vintages: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _experiment_contract(self) -> "JointExperimentSpec":
        if self.model_recipe.pool_id != self.pool_id:
            raise ValueError("model_recipe pool_id must match experiment pool_id")
        if self.experiment_kind == "incremental" and len(self.candidate_factors) > 8:
            raise ValueError("incremental experiments allow at most 8 candidate factors")
        if len(set(self.baseline_factor_ids)) != len(self.baseline_factor_ids):
            raise ValueError("baseline_factor_ids must be unique")
        factor_ids = [factor.factor_id for factor in self.candidate_factors]
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError("candidate_factors must have unique expression IDs")
        if tuple(sorted(set(self.prediction_vintages))) != self.prediction_vintages:
            raise ValueError("prediction_vintages must be sorted and unique")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def experiment_id(self) -> str:
        return _sha256(self.canonical_payload())


class CandidateFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_id: str = Field(min_length=64, max_length=64)
    disposition: FactorDisposition
    coverage: float = Field(ge=0.0, le=1.0)
    missing_ratio: float = Field(ge=0.0, le=1.0)
    annual_rank_ic: dict[int, float | None] = Field(default_factory=dict)
    annual_spread_return: dict[int, float | None] = Field(default_factory=dict)
    nearest_factor_id: str | None = None
    nearest_factor_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    exposure_diagnostics: dict[str, float | None] = Field(default_factory=dict)
    failure_reasons: tuple[str, ...] = ()


class ExperimentFeedback(BaseModel):
    """Only development/research evidence can be serialized back to RD-Agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(min_length=64, max_length=64)
    feedback_window: Literal["research_feedback"] = "research_feedback"
    candidate_feedback: tuple[CandidateFeedback, ...]
    annual_incremental_active_return: dict[int, float] = Field(default_factory=dict)
    median_annual_incremental_active_return: float | None = None
    positive_year_share: float | None = Field(default=None, ge=0.0, le=1.0)
    maximum_drawdown_degradation: float | None = None
    prediction_correlation_to_baseline: float | None = Field(default=None, ge=-1.0, le=1.0)
    training_seconds: float | None = Field(default=None, ge=0.0)
    peak_memory_mb: float | None = Field(default=None, ge=0.0)
    errors: tuple[str, ...] = ()


class AgentCampaignPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=16)
    created_at: datetime
    git_commit: str = Field(min_length=7)
    package_version: Literal["2.2.0"] = "2.2.0"
    rdagent_version: Literal["1.0.0"] = "1.0.0"
    rdagent_commit: Literal["484776c211e4fbbeef03e0ec00d6bbee7362a4f4"] = (
        "484776c211e4fbbeef03e0ec00d6bbee7362a4f4"
    )
    config_hash: str = Field(min_length=64, max_length=64)
    expression_schema_hash: str = Field(min_length=64, max_length=64)
    config: AgentResearchConfig

    @classmethod
    def create(
        cls,
        config: AgentResearchConfig,
        *,
        git_commit: str,
        expression_schema_hash: str,
        created_at: datetime | None = None,
    ) -> "AgentCampaignPlan":
        config_hash = _sha256(config.model_dump(mode="json"))
        plan_id = _sha256(
            {
                "config_hash": config_hash,
                "git_commit": git_commit,
                "expression_schema_hash": expression_schema_hash,
                "package_version": "2.2.0",
                "rdagent_commit": "484776c211e4fbbeef03e0ec00d6bbee7362a4f4",
            }
        )[:20]
        return cls(
            plan_id=plan_id,
            created_at=created_at or datetime.now(timezone.utc),
            git_commit=git_commit,
            config_hash=config_hash,
            expression_schema_hash=expression_schema_hash,
            config=config,
        )


class AgentCampaignResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    campaign_id: str
    plan_id: str
    status: CampaignStatus
    run_dir: Path
    raw_proposals: int = Field(ge=0, le=400)
    accepted_unique_factors: int = Field(ge=0)
    completed_incremental_experiments: int = Field(ge=0, le=30)
    completed_joint_experiments: int = Field(ge=0, le=30)
    pareto_experiment_ids: tuple[str, ...] = Field(default=(), max_length=10)
    result_manifest_hash: str = Field(min_length=64, max_length=64)
    started_at: datetime
    finished_at: datetime | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _result_contract(self) -> "AgentCampaignResult":
        if self.status in {"complete", "incomplete", "failed"} and self.finished_at is None:
            raise ValueError("terminal campaign status requires finished_at")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed campaign requires failure_reason")
        if self.status != "failed" and self.failure_reason is not None:
            raise ValueError("failure_reason is only valid for failed campaigns")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.status in {"planned", "running"} and self.finished_at is not None:
            raise ValueError("non-terminal campaign status cannot have finished_at")
        return self


__all__ = [
    "AgentCampaignPlan",
    "AgentCampaignResult",
    "AgentResearchConfig",
    "AgentResearchWindows",
    "AgentProviderSettings",
    "CandidateFeedback",
    "DateWindow",
    "DatasetAdapterSpec",
    "ExecutorSpec",
    "ExperimentFeedback",
    "FactorCandidateSpec",
    "JointExperimentSpec",
    "LabelContract",
    "ModelRecipeSpec",
    "ResourceGuards",
    "ResearchHypothesis",
    "StageBudgets",
    "CostModel",
    "StrategyContract",
]
