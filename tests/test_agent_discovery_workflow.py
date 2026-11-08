from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from zyquant.research.agent_discovery import (
    AgentProviderSettings,
    AgentResearchConfig,
    DatasetAdapterSpec,
    ResourceGuards,
    StageBudgets,
)
from zyquant.research.agent_discovery.report import verify_agent_campaign
from zyquant.research.agent_discovery.workflow import (
    create_agent_plan,
    resume_agent_campaign,
    run_agent_campaign,
)
from zyquant.research.factor_discovery.adapter import DiscoveryDataset

from tests.test_agent_discovery_models import config as contract_config
import zyquant.research.agent_discovery.workflow as agent_workflow


@pytest.fixture(autouse=True)
def _sufficient_campaign_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay tests exercise contracts, not the host running the test suite."""

    monkeypatch.setattr(
        agent_workflow.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=512 * 1024**3),
    )
    monkeypatch.setattr(agent_workflow, "_available_memory_gb", lambda: 256.0)


class SyntheticAdapter:
    def metadata(self):
        return {"dataset_fingerprint": "a" * 64, "pool_id": "CSI300"}

    def load_period(self, period: str, start: date, end: date) -> DiscoveryDataset:
        del period, start, end
        rows = []
        for day_number, day in enumerate(pd.date_range("2020-01-01", periods=4)):
            for instrument_number, instrument in enumerate(("A", "B", "C")):
                rows.append({
                    "trade_date": day,
                    "instrument_id": instrument,
                    "value": float(day_number * 3 + instrument_number + 1),
                })
        field = pd.DataFrame(rows)
        observations = field[["trade_date", "instrument_id"]].rename(
            columns={"trade_date": "decision_time"}
        )
        observations["observation_id"] = range(len(observations))
        observations["active_return"] = 0.01
        labels = pd.DataFrame([{"contract": "synthetic"}])
        return DiscoveryDataset(observations, {"raw.close": field}, labels)


def create_adapter() -> SyntheticAdapter:
    return SyntheticAdapter()


def _candidate() -> dict:
    return {
        "candidate_name": "close_level",
        "hypothesis_id": "hyp-0001",
        "expression": {"op": "field", "source": "raw", "field": "close"},
        "direction": 1,
        "rationale": "Closing price level provides a deterministic workflow test candidate.",
        "repair_attempt": 0,
    }


def _hypothesis() -> dict:
    return {
        "hypothesis_id": "hyp-0001",
        "iteration": 1,
        "title": "Closing price persistence hypothesis",
        "thesis": "Persistent price levels may represent slowly changing information in the pool.",
        "economic_mechanism": "Gradual information diffusion can produce a stable cross-sectional ordering.",
        "expected_direction": 1,
        "candidate_fields": ["raw.close"],
        "parent_hypothesis_ids": [],
    }


def workflow_config(replay_path: Path) -> AgentResearchConfig:
    base = contract_config("CSI300")
    payload = base.model_dump(mode="json")
    payload["dataset_adapter"] = DatasetAdapterSpec(
        reference="tests.test_agent_discovery_workflow:create_adapter"
    ).model_dump(mode="json")
    payload["allowed_fields"] = ["raw.close"]
    payload["agent"] = AgentProviderSettings(
        provider="replay", model="replay", replay_path=replay_path
    ).model_dump(mode="json")
    payload["resources"] = ResourceGuards(
        minimum_free_gb=100, minimum_free_memory_gb=50,
        factor_workers=1, model_workers=1, threads_per_model=1,
    ).model_dump(mode="json")
    payload["budgets"] = StageBudgets(
        factor_target=1, max_factor_rounds=1, proposals_per_round=1,
        max_raw_proposals=1, max_incremental_experiments=0,
        max_joint_experiments=0,
    ).model_dump(mode="json")
    return AgentResearchConfig.model_validate(payload)


def test_replay_workflow_is_content_addressed_and_resume_is_idempotent(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps({
        "factor_rounds": [{"hypotheses": [_hypothesis()], "candidates": [_candidate()]}],
        "experiments": [],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(workflow_config(replay_path).model_dump(mode="json")),
        encoding="utf-8",
    )
    plan_dir = tmp_path / "plan"
    run_dir = tmp_path / "run"
    plan = create_agent_plan(config_path, plan_dir)
    result = run_agent_campaign(plan_dir, run_dir)

    assert result.status == "complete"
    assert result.accepted_unique_factors == 1
    assert plan.config_hash
    assert (run_dir / "FACTOR_LOCK.json").exists()
    assert (run_dir / "PARETO_LOCK.json").exists()
    assert verify_agent_campaign(run_dir)["status"] == "valid"
    resumed = resume_agent_campaign(run_dir)
    assert resumed == result


def test_agent_context_never_contains_diagnostic_or_reference_evidence(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps({
        "factor_rounds": [{"hypotheses": [_hypothesis()], "candidates": [_candidate()]}],
        "experiments": [],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(workflow_config(replay_path).model_dump(mode="json")),
        encoding="utf-8",
    )
    plan_dir, run_dir = tmp_path / "plan", tmp_path / "run"
    create_agent_plan(config_path, plan_dir)
    run_agent_campaign(plan_dir, run_dir)
    calls = (run_dir / "agent_calls.jsonl").read_text(encoding="utf-8")
    assert "diagnostic" not in calls
    assert "reference" not in calls
    assert "OPENAI_API_KEY" not in "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in run_dir.rglob("*") if path.is_file() and path.suffix != ".parquet"
    )
