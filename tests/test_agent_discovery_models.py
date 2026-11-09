from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from zyquant.research.agent_discovery import (
    AgentCampaignPlan,
    AgentCampaignResult,
    AgentProviderSettings,
    AgentResearchConfig,
    AgentResearchWindows,
    CandidateFeedback,
    CostModel,
    DateWindow,
    DatasetAdapterSpec,
    ExecutorSpec,
    ExperimentFeedback,
    FactorCandidateSpec,
    JointExperimentSpec,
    LabelContract,
    ModelRecipeSpec,
    ResourceGuards,
    ResearchHypothesis,
    StrategyContract,
)
from zyquant.research.factor_discovery import EvidenceRecipe, ExpressionSpec, Operator


CSI300_SEEDS = (20260722, 20260829, 20260916)
CSI1000_SEEDS = (20260722, 20260723, 20260724)


def windows() -> AgentResearchWindows:
    return AgentResearchWindows(
        research_feedback=DateWindow(start=date(2010, 1, 1), end=date(2022, 12, 31)),
        diagnostic=DateWindow(start=date(2023, 1, 1), end=date(2025, 12, 31)),
        reference=DateWindow(start=date(2026, 1, 1), end=date(2026, 7, 24)),
        prospective_start=date(2026, 7, 25),
    )


def config(pool_id: str = "CSI1000") -> AgentResearchConfig:
    seeds = CSI1000_SEEDS if pool_id == "CSI1000" else CSI300_SEEDS
    return AgentResearchConfig(
        campaign_id=f"{pool_id.lower()}-agent-v1",
        pool_id=pool_id,
        dataset_id="pit-panel-v1",
        dataset_fingerprint="a" * 64,
        dataset_adapter=DatasetAdapterSpec(reference="tests.adapters:PITAdapter"),
        label_contract=LabelContract(
            decision_schedule="every fifth trading observation",
            decision_interval_observations=5,
            entry_rule="enter at the first buyable T+1 open",
            exit_rule="exit at fifth-session close or first sellable close",
            holding_horizon_observations=5,
            execution_delay_observations=1,
            cost_model=CostModel(
                model_id="cn-equity-v1", buy_cost_bps=4.6, sell_cost_bps=9.6
            ),
            overlapping_positions=False,
        ),
        strategy_contract=StrategyContract(
            strategy_id=f"ml-{pool_id.lower()}-enhancement",
            candidate_pool=pool_id,
            benchmark_id=pool_id,
            decision_rule="Use the frozen production ranking and portfolio construction rule.",
            account_adapter="zyquant.production.frozen-account-v1",
            formal_execution_description=(
                "Run the existing PIT-aware formal account engine with frozen trade rules."
            ),
            frozen_parameters={"top_n": 100, "buffer": 20},
        ),
        baseline_factor_ids=("quality", "momentum"),
        allowed_fields=("raw.close", "raw.volume"),
        windows=windows(),
        seeds=seeds,
        agent_executor=ExecutorSpec(
            argv=("python", "-m", "zyquant_rdagent.worker", "{request}", "{output_dir}")
        ),
        experiment_executor=ExecutorSpec(
            argv=("python", "-m", "zyquant.worker", "--spec", "{experiment_path}")
        ),
        resources=ResourceGuards(
            minimum_free_gb=200 if pool_id == "CSI1000" else 100,
            factor_workers=24 if pool_id == "CSI1000" else 12,
            model_workers=4 if pool_id == "CSI1000" else 3,
            threads_per_model=16 if pool_id == "CSI1000" else 20,
        ),
    )


def expression(field: str = "close") -> ExpressionSpec:
    return ExpressionSpec(
        op=Operator.MEAN,
        window=20,
        args=(ExpressionSpec(op=Operator.FIELD, source="raw", field=field),),
    )


def candidate(field: str = "close") -> FactorCandidateSpec:
    return FactorCandidateSpec(
        candidate_name=f"mean_20_{field}",
        hypothesis_id="hyp-0001",
        expression=expression(field),
        direction=1,
        rationale="Persistent trading behavior may contain information about future returns.",
    )


def recipe(pool_id: str = "CSI1000") -> ModelRecipeSpec:
    seeds = CSI1000_SEEDS if pool_id == "CSI1000" else CSI300_SEEDS
    return ModelRecipeSpec(
        pool_id=pool_id,
        model_type="XGBRanker",
        objective="rank:ndcg",
        training_window="3y",
        label_transform="cross_sectional_rank",
        seeds=seeds,
    )


def test_csi1000_single_host_capacity_profile_is_explicit_and_bounded() -> None:
    payload = config("CSI1000").model_dump(mode="json")
    payload["resources"] = ResourceGuards(
        minimum_free_gb=100,
        minimum_free_memory_gb=8,
        capacity_profile="single_host_legacy",
        factor_workers=12,
        factor_execution_mode="process",
        model_workers=3,
        threads_per_model=16,
    ).model_dump(mode="json")

    value = AgentResearchConfig.model_validate(payload)
    assert value.resources.capacity_profile == "single_host_legacy"
    assert value.resources.factor_execution_mode == "process"

    payload["resources"]["factor_workers"] = 13
    with pytest.raises(ValidationError, match="single-host resources"):
        AgentResearchConfig.model_validate(payload)


def experiment() -> JointExperimentSpec:
    return JointExperimentSpec(
        campaign_id="csi1000-agent-v1",
        hypothesis_id="hyp-0001",
        experiment_kind="incremental",
        pool_id="CSI1000",
        baseline_factor_ids=("quality", "momentum"),
        candidate_factors=(candidate(),),
        model_recipe=recipe(),
        prediction_vintages=(2020, 2021, 2022),
    )


def test_research_windows_are_strictly_ordered_and_only_development_can_feed_back() -> None:
    value = windows()
    assert value.is_feedback_eligible("research_feedback")
    assert not value.is_feedback_eligible("diagnostic")
    assert not value.is_feedback_eligible("reference")

    with pytest.raises(ValidationError, match="ordered and disjoint"):
        AgentResearchWindows(
            research_feedback=DateWindow(start=date(2020, 1, 1), end=date(2023, 1, 1)),
            diagnostic=DateWindow(start=date(2023, 1, 1), end=date(2024, 1, 1)),
            reference=DateWindow(start=date(2025, 1, 1), end=date(2025, 12, 31)),
            prospective_start=date(2026, 1, 1),
        )


@pytest.mark.parametrize(
    ("pool_id", "seeds"),
    [("CSI300", CSI300_SEEDS), ("CSI1000", CSI1000_SEEDS)],
)
def test_config_accepts_only_pool_fixed_seeds(pool_id: str, seeds: tuple[int, ...]) -> None:
    assert config(pool_id).seeds == seeds
    payload = config(pool_id).model_dump()
    payload["seeds"] = tuple(reversed(seeds))
    with pytest.raises(ValidationError, match="seeds must be"):
        AgentResearchConfig.model_validate(payload)


def test_config_enforces_unique_catalog_and_proposal_budget() -> None:
    payload = config().model_dump()
    payload["allowed_fields"] = ("raw.close", "raw.close")
    with pytest.raises(ValidationError, match="allowed_fields must be unique"):
        AgentResearchConfig.model_validate(payload)

    payload = config().model_dump()
    payload["budgets"]["factor_target"] = 200
    payload["budgets"]["max_raw_proposals"] = 199
    with pytest.raises(ValidationError, match="factor_target"):
        AgentResearchConfig.model_validate(payload)

    payload = config("CSI1000").model_dump()
    payload["resources"]["threads_per_model"] = 17
    with pytest.raises(ValidationError, match="CSI1000 resources"):
        AgentResearchConfig.model_validate(payload)


def test_provider_replay_and_safe_process_contracts_do_not_accept_secrets_or_shell() -> None:
    with pytest.raises(ValidationError, match="requires replay_path"):
        AgentProviderSettings(provider="replay")
    replay = AgentProviderSettings(provider="replay", replay_path=Path("trace.jsonl"))
    assert replay.replay_path == Path("trace.jsonl")

    with pytest.raises(ValidationError, match="shell executables are forbidden"):
        ExecutorSpec(argv=("bash", "-c", "python worker.py"))
    with pytest.raises(ValidationError, match="forbidden shell syntax"):
        ExecutorSpec(argv=("python", "worker.py;rm", "-rf"))
    with pytest.raises(ValidationError, match="unsupported placeholders"):
        ExecutorSpec(argv=("python", "worker.py", "{api_key}"))
    with pytest.raises(ValidationError, match="forbidden secret/code key"):
        DatasetAdapterSpec(
            reference="tests.adapters:PITAdapter",
            parameters={"api_key": "must-not-be-in-config"},
        )


def test_label_and_strategy_contracts_freeze_economic_semantics() -> None:
    label = config().label_contract.model_dump()
    label["overlapping_positions"] = True
    with pytest.raises(ValidationError, match="must agree"):
        LabelContract.model_validate(label)
    label["overlapping_positions"] = False
    label["label_is_net_of_costs"] = False
    with pytest.raises(ValidationError):
        LabelContract.model_validate(label)

    strategy = config().strategy_contract.model_dump()
    strategy["frozen_parameters"] = {"python_code": "do_not_execute()"}
    with pytest.raises(ValidationError, match="forbidden secret/code key"):
        StrategyContract.model_validate(strategy)

    payload = config().model_dump()
    payload["strategy_contract"]["candidate_pool"] = "CSI300"
    with pytest.raises(ValidationError, match="candidate_pool must match"):
        AgentResearchConfig.model_validate(payload)


def test_evidence_recipe_is_frozen_for_reporting_not_technical_acceptance() -> None:
    payload = config().model_dump()
    payload["evidence"] = EvidenceRecipe(
        monotonicity_threshold=1.0,
        fdr_alpha=0.000001,
        confidence_level=0.999999,
        hac_primary_lag=12,
    ).model_dump()
    value = AgentResearchConfig.model_validate(payload)

    # These values configure RollingEvidenceEngine output. Technical validity remains
    # exclusively the expression/PIT/numeric/newness contract and has no significance gate.
    assert value.evidence.monotonicity_threshold == 1.0
    assert value.evidence.fdr_alpha == 0.000001
    assert value.evidence.hac_primary_lag == 12
    value_candidate = candidate()
    value_candidate.expression.validate_contract(set(value.allowed_fields))
    assert value_candidate.factor_id

def test_hypothesis_and_candidate_are_strict_code_free_contracts() -> None:
    hypothesis = ResearchHypothesis(
        hypothesis_id="hyp-0001",
        iteration=1,
        title="Turnover persistence hypothesis",
        thesis="Persistent turnover may proxy for gradual information diffusion in the pool.",
        economic_mechanism="Investor attention changes slowly and may produce delayed price adjustment.",
        expected_direction=1,
        candidate_fields=("raw.volume", "raw.close"),
    )
    assert hypothesis.iteration == 1
    assert candidate().factor_id == expression().expression_id

    payload = candidate().model_dump()
    payload["python_code"] = "__import__('os').system('id')"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FactorCandidateSpec.model_validate(payload)


@pytest.mark.parametrize("training_window", ["3y", "5y", "expanding"])
def test_model_recipe_supports_only_registered_models_and_windows(training_window: str) -> None:
    value = recipe().model_copy(update={"training_window": training_window})
    assert value.training_window == training_window

    with pytest.raises(ValidationError):
        ModelRecipeSpec.model_validate({**recipe().model_dump(), "model_type": "LightGBM"})
    with pytest.raises(ValidationError):
        ModelRecipeSpec.model_validate({**recipe().model_dump(), "training_window": "10y"})


def test_model_recipe_objective_bounds_and_seeds_are_validated() -> None:
    with pytest.raises(ValidationError, match="ranking objective"):
        ModelRecipeSpec.model_validate({**recipe().model_dump(), "objective": "reg:squarederror"})
    with pytest.raises(ValidationError, match="below max_rounds"):
        ModelRecipeSpec.model_validate(
            {**recipe().model_dump(), "max_rounds": 100, "early_stopping_rounds": 100}
        )
    with pytest.raises(ValidationError):
        ModelRecipeSpec.model_validate({**recipe().model_dump(), "learning_rate": 1.0})
    with pytest.raises(ValidationError, match="seeds must be"):
        ModelRecipeSpec.model_validate({**recipe().model_dump(), "seeds": (1, 2, 3)})

    regressor = ModelRecipeSpec.model_validate(
        {
            **recipe().model_dump(),
            "model_type": "XGBRegressor",
            "objective": "reg:pseudohubererror",
        }
    )
    assert regressor.objective == "reg:pseudohubererror"


def test_joint_experiment_id_is_stable_and_content_addressed() -> None:
    first = experiment()
    second = JointExperimentSpec.model_validate(first.model_dump(mode="json"))
    assert len(first.experiment_id) == 64
    assert first.experiment_id == second.experiment_id

    changed = first.model_copy(update={"prediction_vintages": (2020, 2021)})
    assert changed.experiment_id != first.experiment_id


def test_joint_experiment_rejects_pool_mismatch_duplicates_and_unsorted_vintages() -> None:
    payload = experiment().model_dump()
    payload["pool_id"] = "CSI300"
    with pytest.raises(ValidationError, match="pool_id must match"):
        JointExperimentSpec.model_validate(payload)

    payload = experiment().model_dump()
    payload["candidate_factors"] = (candidate(), candidate())
    with pytest.raises(ValidationError, match="unique expression IDs"):
        JointExperimentSpec.model_validate(payload)

    payload = experiment().model_dump()
    payload["prediction_vintages"] = (2021, 2020)
    with pytest.raises(ValidationError, match="sorted and unique"):
        JointExperimentSpec.model_validate(payload)


def test_diagnostic_or_reference_evidence_cannot_be_agent_feedback() -> None:
    candidate_result = CandidateFeedback(
        factor_id=candidate().factor_id,
        disposition="model_candidate",
        coverage=0.95,
        missing_ratio=0.05,
    )
    feedback = ExperimentFeedback(
        experiment_id=experiment().experiment_id,
        candidate_feedback=(candidate_result,),
        annual_incremental_active_return={2020: 0.01},
    )
    assert feedback.feedback_window == "research_feedback"

    payload = feedback.model_dump()
    payload["feedback_window"] = "diagnostic"
    with pytest.raises(ValidationError):
        ExperimentFeedback.model_validate(payload)


def test_campaign_plan_is_reproducible_and_pins_rdagent() -> None:
    created_at = datetime(2026, 11, 8, tzinfo=timezone.utc)
    first = AgentCampaignPlan.create(
        config(), git_commit="ad9c29d4d3581", expression_schema_hash="b" * 64,
        created_at=created_at,
    )
    second = AgentCampaignPlan.create(
        config(), git_commit="ad9c29d4d3581", expression_schema_hash="b" * 64,
        created_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    assert first.plan_id == second.plan_id
    assert first.rdagent_version == "1.0.0"
    assert first.rdagent_commit == "484776c211e4fbbeef03e0ec00d6bbee7362a4f4"


def test_campaign_result_terminal_state_contract() -> None:
    base = {
        "campaign_id": "csi1000-agent-v1",
        "plan_id": "1" * 20,
        "status": "complete",
        "run_dir": Path("/tmp/campaign"),
        "raw_proposals": 400,
        "accepted_unique_factors": 200,
        "completed_incremental_experiments": 30,
        "completed_joint_experiments": 30,
        "result_manifest_hash": "c" * 64,
        "started_at": datetime(2026, 11, 8, tzinfo=timezone.utc),
    }
    with pytest.raises(ValidationError, match="requires finished_at"):
        AgentCampaignResult.model_validate(base)

    result = AgentCampaignResult.model_validate(
        {**base, "finished_at": datetime(2026, 11, 9, tzinfo=timezone.utc)}
    )
    assert result.status == "complete"

    with pytest.raises(ValidationError, match="requires failure_reason"):
        AgentCampaignResult.model_validate(
            {
                **base,
                "status": "failed",
                "finished_at": datetime(2026, 11, 9, tzinfo=timezone.utc),
            }
        )
