from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Operator(str, Enum):
    CONST = "const"
    FIELD = "field"
    NEG = "neg"
    ABS = "abs"
    SIGN = "sign"
    LOG_ABS = "log_abs"
    SQRT_ABS = "sqrt_abs"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    SAFE_DIV = "safe_div"
    MIN = "min"
    MAX = "max"
    LAG = "lag"
    DELTA = "delta"
    RETURN = "return"
    MEAN = "mean"
    STD = "std"
    SUM = "sum"
    ROLLING_MIN = "rolling_min"
    ROLLING_MAX = "rolling_max"
    RANK = "rank"
    CORR = "corr"
    EMA = "ema"


_UNARY = {
    Operator.NEG, Operator.ABS, Operator.SIGN, Operator.LOG_ABS,
    Operator.SQRT_ABS,
}
_BINARY = {
    Operator.ADD, Operator.SUB, Operator.MUL, Operator.SAFE_DIV,
    Operator.MIN, Operator.MAX,
}
_WINDOW_UNARY = {
    Operator.LAG, Operator.DELTA, Operator.RETURN, Operator.MEAN,
    Operator.STD, Operator.SUM, Operator.ROLLING_MIN, Operator.ROLLING_MAX,
    Operator.RANK, Operator.EMA,
}
_COMMUTATIVE = {Operator.ADD, Operator.MUL, Operator.MIN, Operator.MAX}


class ExpressionSpec(BaseModel):
    """A closed, declarative factor expression. It is never evaluated as Python."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Operator
    source: Literal["raw", "post_adjusted", "valuation", "factor"] | None = None
    field: str | None = None
    value: float | None = None
    window: int | None = None
    args: tuple["ExpressionSpec", ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> "ExpressionSpec":
        if self.op == Operator.CONST:
            if self.value is None or self.args or self.field or self.source or self.window:
                raise ValueError("const requires only a finite value")
            if not (-1e100 < self.value < 1e100):
                raise ValueError("const value must be finite and bounded")
        elif self.op == Operator.FIELD:
            if not self.field or not self.source:
                raise ValueError("field requires source and field")
            if self.args or self.value is not None or self.window is not None:
                raise ValueError("field cannot have args, value, or window")
        elif self.op in _UNARY:
            self._require_args(1)
            if self.window is not None:
                raise ValueError(f"{self.op.value} does not accept window")
        elif self.op in _BINARY:
            self._require_args(2)
            if self.window is not None:
                raise ValueError(f"{self.op.value} does not accept window")
        elif self.op in _WINDOW_UNARY:
            self._require_args(1)
            self._require_window()
        elif self.op == Operator.CORR:
            self._require_args(2)
            self._require_window()
        else:  # pragma: no cover - Enum makes this defensive
            raise ValueError(f"unsupported operator: {self.op}")
        if self.op not in {Operator.CONST, Operator.FIELD} and (
            self.field is not None or self.source is not None or self.value is not None
        ):
            raise ValueError(f"{self.op.value} accepts only args and optional window")
        return self

    def _require_args(self, count: int) -> None:
        if len(self.args) != count:
            raise ValueError(f"{self.op.value} requires {count} args")

    def _require_window(self) -> None:
        if self.window is None or not 1 <= self.window <= 252:
            raise ValueError(f"{self.op.value} window must be in [1, 252]")

    @property
    def depth(self) -> int:
        return 1 + max((arg.depth for arg in self.args), default=0)

    @property
    def node_count(self) -> int:
        return 1 + sum(arg.node_count for arg in self.args)

    @property
    def lookback(self) -> int:
        child = max((arg.lookback for arg in self.args), default=0)
        if self.op in _WINDOW_UNARY or self.op == Operator.CORR:
            return child + int(self.window or 0)
        return child

    @property
    def leaf_key(self) -> str:
        if self.op != Operator.FIELD:
            raise ValueError("leaf_key is only defined for field nodes")
        return f"{self.source}.{self.field}"

    def fields(self) -> set[str]:
        if self.op == Operator.FIELD:
            return {self.leaf_key}
        return set().union(*(arg.fields() for arg in self.args)) if self.args else set()

    def validate_contract(
        self,
        allowed_fields: set[str],
        *,
        max_depth: int = 8,
        max_nodes: int = 64,
        max_lookback: int = 252,
    ) -> None:
        unknown = self.fields() - allowed_fields
        if unknown:
            raise ValueError(f"expression uses fields outside whitelist: {sorted(unknown)}")
        if self.depth > max_depth:
            raise ValueError(f"expression depth {self.depth} exceeds {max_depth}")
        if self.node_count > max_nodes:
            raise ValueError(f"expression nodes {self.node_count} exceeds {max_nodes}")
        if self.lookback > max_lookback:
            raise ValueError(f"expression lookback {self.lookback} exceeds {max_lookback}")

    def canonical_payload(self) -> dict[str, Any]:
        args = [arg.canonical_payload() for arg in self.args]
        if self.op in _COMMUTATIVE:
            args.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        payload: dict[str, Any] = {"op": self.op.value}
        if self.source is not None:
            payload["source"] = self.source
        if self.field is not None:
            payload["field"] = self.field
        if self.value is not None:
            payload["value"] = self.value
        if self.window is not None:
            payload["window"] = self.window
        if args:
            payload["args"] = args
        return payload

    @property
    def expression_id(self) -> str:
        raw = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hashlib.sha256(raw).hexdigest()


class CandidateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(min_length=8, max_length=1000)
    direction: Literal[-1, 1]
    expression: ExpressionSpec
    rationale: str = Field(min_length=8, max_length=2000)


class CandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateSpec] = Field(min_length=1, max_length=20)


class DateWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> "DateWindow":
        if self.start > self.end:
            raise ValueError("window start must not exceed end")
        return self


class DiscoveryWindows(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    development: DateWindow
    diagnostic: DateWindow
    holdout: DateWindow

    @model_validator(mode="after")
    def _disjoint(self) -> "DiscoveryWindows":
        if not (
            self.development.end < self.diagnostic.start
            and self.diagnostic.end < self.holdout.start
        ):
            raise ValueError("development, diagnostic, and holdout must be ordered and disjoint")
        return self


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openai", "replay"] = "openai"
    model: str = "gpt-5.6-sol"
    base_url: str | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "medium"
    timeout_seconds: int = Field(default=180, ge=1, le=600)
    max_retries: int = Field(default=3, ge=0, le=8)
    replay_path: Path | None = None

    @model_validator(mode="after")
    def _replay_path(self) -> "LLMSettings":
        if self.provider == "replay" and self.replay_path is None:
            raise ValueError("replay provider requires replay_path")
        return self


class EvidenceRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    preferred_group_count: int = Field(default=5, ge=2, le=10)
    fallback_group_count: int = Field(default=3, ge=2, le=10)
    minimum_pool_size: int = Field(default=30, ge=2)
    minimum_group_size: int = Field(default=10, ge=1)
    factor_non_null_ratio: float = Field(default=0.8, gt=0, le=1)
    valid_decision_coverage: float = Field(default=0.7, gt=0, le=1)
    minimum_valid_decisions: int = Field(default=30, ge=2)
    valid_decision_cap: int = Field(default=126, ge=2)
    fdr_alpha: float = Field(default=0.05, gt=0, lt=1)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    hac_primary_lag: int = Field(default=4, ge=0)
    hac_sensitivity_lags: tuple[int, ...] = (8, 12)
    minimum_effect: float = Field(default=0.001, ge=0)
    positive_period_share: float = Field(default=0.6, ge=0.5, le=1)
    monotonicity_threshold: float = Field(default=0.6, ge=0, le=1)
    redundancy_correlation: float = Field(default=0.95, gt=0, le=1)


class FactorDiscoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    campaign_id: str = Field(min_length=3, pattern=r"^[A-Za-z0-9._-]+$")
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=8)
    pool_id: str = Field(min_length=1)
    timezone: str = "Asia/Shanghai"
    adapter: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    adapter_parameters: dict[str, Any] = Field(default_factory=dict)
    allowed_fields: tuple[str, ...] = Field(min_length=1)
    windows: DiscoveryWindows
    llm: LLMSettings = Field(default_factory=LLMSettings)
    evidence: EvidenceRecipe = Field(default_factory=EvidenceRecipe)
    target_candidates: int = Field(default=200, ge=1, le=2000)
    proposals_per_call: int = Field(default=20, ge=1, le=20)
    max_rounds: int = Field(default=20, ge=1, le=100)
    max_proposals: int = Field(default=400, ge=1, le=10000)
    evaluation_workers: int = Field(default=1, ge=1, le=20)
    minimum_free_gb: float = Field(default=100.0, ge=0)
    random_seed: int = 20261107
    label_contract: dict[str, Any]
    pool_description: str = Field(min_length=8, max_length=4000)

    @model_validator(mode="after")
    def _limits(self) -> "FactorDiscoveryConfig":
        if len(set(self.allowed_fields)) != len(self.allowed_fields):
            raise ValueError("allowed_fields must be unique")
        if self.max_proposals < self.target_candidates:
            raise ValueError("max_proposals must be >= target_candidates")
        required = {
            "decision_schedule", "decision_interval_observations", "entry_rule",
            "exit_rule", "holding_horizon_observations", "execution_delay_observations",
            "cost_model", "label_is_net_of_costs", "overlapping_positions",
        }
        missing = required - set(self.label_contract)
        if missing:
            raise ValueError(f"label_contract missing fields: {sorted(missing)}")
        return self

    def redacted(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DiscoveryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    plan_id: str
    created_at: datetime
    config_hash: str
    prompt_hash: str
    expression_schema_hash: str
    git_commit: str
    package_version: str
    adapter_metadata: dict[str, Any]
    config: FactorDiscoveryConfig

    @classmethod
    def create(
        cls,
        config: FactorDiscoveryConfig,
        *,
        prompt_hash: str,
        expression_schema_hash: str,
        git_commit: str,
        package_version: str,
        adapter_metadata: dict[str, Any],
    ) -> "DiscoveryPlan":
        config_json = json.dumps(
            config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        identity = json.dumps({
            "config_hash": config_hash,
            "prompt_hash": prompt_hash,
            "expression_schema_hash": expression_schema_hash,
            "git_commit": git_commit,
            "package_version": package_version,
            "adapter_metadata": adapter_metadata,
        }, sort_keys=True, separators=(",", ":"), default=str)
        return cls(
            plan_id=hashlib.sha256(identity.encode()).hexdigest()[:20],
            created_at=datetime.now(timezone.utc),
            config_hash=config_hash,
            prompt_hash=prompt_hash,
            expression_schema_hash=expression_schema_hash,
            git_commit=git_commit,
            package_version=package_version,
            adapter_metadata=adapter_metadata,
            config=config,
        )


class DiscoveryRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    campaign_id: str
    plan_id: str
    status: Literal["complete", "incomplete", "failed"]
    accepted_candidates: int
    raw_proposals: int
    rounds: int
    run_dir: Path
    period_outputs: dict[str, str] = Field(default_factory=dict)
