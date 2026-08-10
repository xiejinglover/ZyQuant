from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from zyquant.core.hashing import hash_payload


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class DataConfig(StrictModel):
    root: Path
    dataset_id: str
    start_date: date
    end_date: date
    cutoff: date | None = None
    verify_hashes: bool = True
    validation_level: Literal["strict", "standard"] = "strict"

    @model_validator(mode="after")
    def validate_dates(self) -> "DataConfig":
        if self.start_date > self.end_date:
            raise ValueError("start_date must not exceed end_date")
        effective = self.cutoff or self.end_date
        if effective < self.end_date:
            raise ValueError("data cutoff must cover end_date")
        return self


class FactorConfig(StrictModel):
    cache_root: Path = Path(".zyquant/cache/factors")
    factors: tuple[Mapping[str, Any], ...] = ()
    lock_timeout_seconds: float = Field(default=120.0, gt=0)
    cache_policy: Literal["compute", "require"] = "compute"
    strict_quality: bool = True


class ModelConfig(StrictModel):
    enabled: bool = False
    feature_set: Mapping[str, Any] = Field(default_factory=dict)
    label: Mapping[str, Any] = Field(default_factory=dict)
    splitter: Mapping[str, Any] = Field(default_factory=dict)
    trainer: Mapping[str, Any] = Field(default_factory=dict)
    rolling: bool = False


class StrategyConfig(StrictModel):
    strategies: tuple[Mapping[str, Any], ...] = ()
    state_schema_version: str = "1.0"


class PortfolioConfig(StrictModel):
    constructor: Mapping[str, Any] = Field(default_factory=dict)
    constraints: Mapping[str, Any] = Field(default_factory=dict)


class CostOverride(StrictModel):
    """Per-asset-type cost rates. Unset fields fall through to the snapshot."""

    commission_bps: float | None = Field(default=None, ge=0)
    minimum_commission: float | None = Field(default=None, ge=0)
    sell_tax_bps: float | None = Field(default=None, ge=0)
    buy_tax_bps: float | None = Field(default=None, ge=0)
    transfer_fee_bps: float | None = Field(default=None, ge=0)


class ExecutionConfig(StrictModel):
    model: str | None = None
    model_parameters: Mapping[str, Any] = Field(default_factory=dict)
    cost_model: str | None = None
    cost_model_parameters: Mapping[str, Any] = Field(default_factory=dict)
    timing: Literal["next_open", "same_close", "next_close"] = "next_open"
    max_participation: float = Field(default=0.05, gt=0, le=1)
    slippage_bps: float = Field(default=2.0, ge=0)
    impact_coefficient_bps: float = Field(default=50.0, ge=0)
    max_impact_bps: float = Field(default=100.0, ge=0)
    # Explicit scenario overrides. None means use the historical market_rules
    # table, falling back to DEFAULT_MARKET_COSTS where it carries no value.
    commission_bps: float | None = Field(default=None, ge=0)
    minimum_commission: float | None = Field(default=None, ge=0)
    stock_sell_tax_bps: float | None = Field(default=None, ge=0)
    # Finer-grained overrides, keyed by asset_type ("stock", "etf"). These
    # reach buy_tax_bps and transfer_fee_bps, which no scalar above covers.
    cost_overrides: Mapping[str, CostOverride] = Field(default_factory=dict)


class ConstraintConfig(StrictModel):
    min_cash_weight: float = Field(default=0.0, ge=0, le=1)
    max_cash_weight: float = Field(default=1.0, ge=0, le=1)
    max_instrument_weight: float = Field(default=1.0, gt=0, le=1)
    max_stock_weight: float = Field(default=1.0, ge=0, le=1)
    max_etf_weight: float = Field(default=1.0, ge=0, le=1)
    max_industry_weight: float | None = Field(default=None, gt=0, le=1)
    max_one_way_turnover: float | None = Field(default=None, gt=0, le=1)
    min_target_weight: float = Field(default=0.0, ge=0, le=1)
    min_holdings: int = Field(default=0, ge=0)
    max_holdings: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_constraints(self) -> "ConstraintConfig":
        if self.min_cash_weight > self.max_cash_weight:
            raise ValueError("min_cash_weight must not exceed max_cash_weight")
        if self.max_holdings is not None and self.min_holdings > self.max_holdings:
            raise ValueError("min_holdings must not exceed max_holdings")
        return self


class AccountConfig(StrictModel):
    initial_cash: float = Field(gt=0)
    sleeve_weights: Mapping[str, float] = Field(default_factory=dict)
    capital_allocator: str | None = None
    capital_allocator_parameters: Mapping[str, Any] = Field(default_factory=dict)
    accounting_tolerance: float = Field(default=1e-7, gt=0)

    @model_validator(mode="after")
    def validate_sleeves(self) -> "AccountConfig":
        if self.sleeve_weights:
            if any(value <= 0 for value in self.sleeve_weights.values()):
                raise ValueError("sleeve weights must be positive")
            if abs(sum(self.sleeve_weights.values()) - 1.0) > 1e-12:
                raise ValueError("sleeve weights must sum to one")
        return self


class AnalysisConfig(StrictModel):
    benchmark_id: str | None = None
    risk_free_rate: float = 0.0
    attribution_tolerance: float = Field(default=1e-7, gt=0)
    # Detailed security/industry attribution is an optional post-processing
    # step. Ordinary backtests should not pay its cost.
    attribution: bool = False
    report: bool = True
    report_plugin: str | None = None
    report_parameters: Mapping[str, Any] = Field(default_factory=dict)


class SearchConfig(StrictModel):
    enabled: bool = False
    workers: int = Field(default=1, ge=1)
    sampler: Mapping[str, Any] = Field(default_factory=dict)
    objective: str | None = None
    maximize: bool = True
    retry_resource_errors: int = Field(default=1, ge=0)
    heartbeat_seconds: float = Field(default=5.0, gt=0)
    keep_full_top_n: int = Field(default=20, ge=0)


class ResolvedRunConfig(StrictModel):
    schema_version: str = "1.0"
    run_type: Literal["backtest", "search", "model"] = "backtest"
    data: DataConfig
    factor: FactorConfig = Field(default_factory=FactorConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    account: AccountConfig
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    seed: int = 20260722
    output_root: Path = Path("runs")
    experiment_database: Path = Path("runs/experiments.sqlite")
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return hash_payload(self.model_dump(mode="json"))

    def redacted(self) -> dict[str, Any]:
        sensitive = ("password", "passwd", "token", "secret", "credential", "api_key")

        def visit(value: Any, key: str = "") -> Any:
            if any(marker in key.lower() for marker in sensitive):
                return "***REDACTED***"
            if isinstance(value, Mapping):
                return {str(k): visit(v, str(k)) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [visit(item) for item in value]
            return value

        return visit(self.model_dump(mode="json"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ResolvedRunConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("configuration root must be a mapping")
        return cls.model_validate(payload)


# Backward-compatible construction for the old direct engine API.
class BacktestConfig(StrictModel):
    data: DataConfig
    initial_cash: float = Field(gt=0)
    seed: int = 20260722
    output_root: Path = Path("runs")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


def load_config(value: str | Path | Mapping[str, Any] | ResolvedRunConfig) -> ResolvedRunConfig:
    if isinstance(value, ResolvedRunConfig):
        return value
    if isinstance(value, (str, Path)):
        return ResolvedRunConfig.from_yaml(value)
    return ResolvedRunConfig.model_validate(value)
