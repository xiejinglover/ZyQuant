from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from zyquant.research.factor_discovery import (
    CandidateBatch,
    EvidenceRecipe,
    ExpressionEvaluator,
    ExpressionSpec,
    ReplayLLMClient,
    RollingEvidenceEngine,
)
from zyquant.research.factor_discovery.adapter import DiscoveryDataset
from zyquant.research.factor_discovery.workflow import (
    create_discovery_plan,
    run_discovery,
)


FIELD_CLOSE = {"op": "field", "source": "post_adjusted", "field": "close"}
FIELD_VOLUME = {"op": "field", "source": "raw", "field": "volume"}


def test_expression_is_canonical_safe_and_bounded():
    left = ExpressionSpec.model_validate({
        "op": "add", "args": [FIELD_CLOSE, FIELD_VOLUME],
    })
    right = ExpressionSpec.model_validate({
        "op": "add", "args": [FIELD_VOLUME, FIELD_CLOSE],
    })
    assert left.expression_id == right.expression_id
    assert left.depth == 2
    assert left.node_count == 3
    left.validate_contract({"post_adjusted.close", "raw.volume"})
    with pytest.raises(ValueError, match="outside whitelist"):
        left.validate_contract({"post_adjusted.close"})
    with pytest.raises(ValueError, match="window must be"):
        ExpressionSpec.model_validate({"op": "mean", "window": 0, "args": [FIELD_CLOSE]})
    with pytest.raises(ValueError):
        ExpressionSpec.model_validate({"op": "field", "source": "raw", "field": "close",
                                       "python": "open('/etc/passwd')"})


def test_evaluator_keeps_instrument_histories_isolated_and_division_safe():
    days = pd.date_range("2026-01-01", periods=4, tz="Asia/Shanghai")
    frame = pd.DataFrame({
        "trade_date": list(days) * 2,
        "instrument_id": ["A"] * 4 + ["B"] * 4,
        "value": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0],
    })
    volume = frame.copy()
    volume["value"] = [0.0, 2.0, 0.0, 4.0, 10.0, 0.0, 30.0, 40.0]
    evaluator = ExpressionEvaluator({"post_adjusted.close", "raw.volume"})
    lag = ExpressionSpec.model_validate({"op": "lag", "window": 1, "args": [FIELD_CLOSE]})
    result = evaluator.evaluate(lag, {
        "post_adjusted.close": frame, "raw.volume": volume,
    })
    assert result[result["instrument_id"] == "B"]["value"].iloc[0] != 4.0
    assert np.isnan(result[result["instrument_id"] == "B"]["value"].iloc[0])
    division = ExpressionSpec.model_validate({
        "op": "safe_div", "args": [FIELD_CLOSE, FIELD_VOLUME],
    })
    divided = evaluator.evaluate(division, {
        "post_adjusted.close": frame, "raw.volume": volume,
    })
    assert np.isnan(divided.iloc[0]["value"])
    assert np.isfinite(divided["value"].dropna()).all()


def test_evaluator_uses_pre_observation_history_but_returns_only_requested_rows():
    days = pd.date_range("2026-01-01", periods=4, tz="Asia/Shanghai")
    frame = pd.DataFrame({
        "trade_date": days,
        "instrument_id": "A",
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    requested = frame.iloc[[-1]][["trade_date", "instrument_id"]]
    spec = ExpressionSpec.model_validate({
        "op": "mean", "window": 3, "args": [FIELD_CLOSE],
    })
    result = ExpressionEvaluator({"post_adjusted.close"}).evaluate(
        spec, {"post_adjusted.close": frame}, requested
    )
    assert len(result) == 1
    assert result.iloc[0]["value"] == pytest.approx(3.0)


def test_evaluator_materializes_only_expression_leaves():
    days = pd.date_range("2026-01-01", periods=2, tz="Asia/Shanghai")
    used = pd.DataFrame({
        "trade_date": days,
        "instrument_id": ["A", "A"],
        "value": [1.0, 2.0],
    })
    # Wide/lazy adapters may advertise hundreds of fields. An unrelated field
    # must not be normalized merely because it exists in the mapping.
    unused_invalid = pd.DataFrame({"not_a_factor_column": [1.0]})
    spec = ExpressionSpec.model_validate(FIELD_CLOSE)
    result = ExpressionEvaluator({"post_adjusted.close", "raw.unused"}).evaluate(
        spec,
        {"post_adjusted.close": used, "raw.unused": unused_invalid},
    )
    assert result["value"].tolist() == [1.0, 2.0]


def test_vectorized_rolling_rank_matches_pandas_average_pct_reference():
    days = pd.date_range("2026-01-01", periods=8, tz="Asia/Shanghai")
    frame = pd.DataFrame({
        "trade_date": list(days) * 2,
        "instrument_id": ["A"] * 8 + ["B"] * 8,
        "value": [1.0, 2.0, 2.0, 4.0, np.nan, 3.0, 3.0, 5.0,
                  8.0, 7.0, 7.0, 6.0, 5.0, 5.0, 4.0, 3.0],
    })
    spec = ExpressionSpec.model_validate({
        "op": "rank", "window": 4, "args": [FIELD_CLOSE],
    })

    actual = ExpressionEvaluator({"post_adjusted.close"}).evaluate(
        spec, {"post_adjusted.close": frame}
    )
    reference = frame.groupby("instrument_id", group_keys=False, sort=False)["value"].apply(
        lambda values: values.rolling(4, min_periods=4).apply(
            lambda window: pd.Series(window).rank(method="average", pct=True).iloc[-1],
            raw=False,
        )
    )
    expected = frame[["trade_date", "instrument_id"]].copy()
    expected["value"] = reference
    expected = expected.sort_values(["trade_date", "instrument_id"]).reset_index(drop=True)

    assert actual["value"].to_numpy() == pytest.approx(
        expected["value"].to_numpy(), nan_ok=True
    )


def _evidence_tables(decisions=40, instruments=40):
    rows = []
    values = []
    rng = np.random.default_rng(7)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for day_index in range(decisions):
        decision = start + timedelta(days=day_index)
        signal = np.linspace(-1, 1, instruments)
        labels = signal * (0.01 + day_index * 0.00001) + rng.normal(0, 0.002, instruments)
        for instrument in range(instruments):
            observation_id = f"{day_index}-{instrument}"
            rows.append({
                "observation_id": observation_id, "pool_id": "TEST",
                "decision_time": decision, "instrument_id": f"S{instrument:03d}",
                "label_id": "net_5d", "label_value": labels[instrument],
                "label_exit_time": decision + timedelta(days=5),
            })
            values.append({
                "observation_id": observation_id, "factor_id": "signal",
                "factor_value": signal[instrument], "factor_known_at": decision,
            })
    observations = pd.DataFrame(rows)
    factor_values = pd.DataFrame(values)
    metadata = pd.DataFrame({"factor_id": ["signal"], "factor_type": ["continuous"]})
    label_metadata = pd.DataFrame([{
        "label_id": "net_5d", "decision_schedule": "daily",
        "decision_interval_observations": 1, "entry_rule": "next_open",
        "exit_rule": "fifth_close", "holding_horizon_observations": 5,
        "execution_delay_observations": 1, "cost_model": "test",
        "label_is_net_of_costs": True, "overlapping_positions": True,
    }])
    return observations, factor_values, metadata, label_metadata


def test_evidence_engine_computes_hac_fdr_and_fixed_classification():
    tables = _evidence_tables()
    recipe = EvidenceRecipe(
        minimum_pool_size=20, minimum_group_size=5,
        minimum_valid_decisions=10, valid_decision_cap=20,
        minimum_effect=0.001, positive_period_share=0.5,
        monotonicity_threshold=0.5,
    )
    result = RollingEvidenceEngine(recipe).evaluate(*tables)
    summary = result.factor_summary.iloc[0]
    assert summary["valid_decisions"] == 40
    assert summary["rank_ic_mean"] > 0.8
    assert summary["rank_ic_q_value"] < 0.05
    assert bool(summary["model_candidate"])
    assert bool(summary["economic_eligible"])
    broken = tables[1].copy()
    broken.loc[0, "factor_known_at"] = broken.loc[0, "factor_known_at"] + timedelta(days=1)
    with pytest.raises(ValueError, match="factor_known_at"):
        RollingEvidenceEngine(recipe).evaluate(tables[0], broken, tables[2], tables[3])


def create_adapter(**_):
    return _TestAdapter()


class _TestAdapter:
    def metadata(self):
        return {"dataset_fingerprint": "test-fingerprint-0001", "pool_id": "TEST"}

    def load_period(self, period: str, start: date, end: date):
        days = pd.date_range(start, periods=4, tz="Asia/Shanghai")
        observations = []
        close = []
        volume = []
        for day_index, day in enumerate(days):
            for instrument in range(12):
                observation_id = f"{period}-{day_index}-{instrument}"
                observations.append({
                    "observation_id": observation_id, "pool_id": "TEST",
                    "decision_time": day, "instrument_id": f"S{instrument:02d}",
                    "label_id": "net_5d", "label_value": instrument / 100 + day_index / 1000,
                    "label_exit_time": day + pd.Timedelta(days=5),
                })
                close.append({"trade_date": day, "instrument_id": f"S{instrument:02d}",
                              "value": float(instrument + day_index + 1)})
                volume.append({"trade_date": day, "instrument_id": f"S{instrument:02d}",
                               "value": float((instrument + 1) * 100)})
        label = pd.DataFrame([{
            "label_id": "net_5d", "decision_schedule": "daily",
            "decision_interval_observations": 1, "entry_rule": "next_open",
            "exit_rule": "fifth_close", "holding_horizon_observations": 5,
            "execution_delay_observations": 1, "cost_model": "test",
            "label_is_net_of_costs": True, "overlapping_positions": True,
        }])
        return DiscoveryDataset(
            pd.DataFrame(observations),
            {"post_adjusted.close": pd.DataFrame(close), "raw.volume": pd.DataFrame(volume)},
            label,
        )


def test_replay_campaign_freezes_plan_and_never_feeds_holdout(tmp_path: Path):
    replay = [{"candidates": [
        {"hypothesis": "price level proxy has a stable ordering effect", "direction": 1,
         "expression": FIELD_CLOSE,
         "rationale": "Use the permitted adjusted close as a deterministic test expression."},
        {"hypothesis": "trading activity contains incremental participation information",
         "direction": 1, "expression": FIELD_VOLUME,
         "rationale": "Use the permitted raw volume as a second deterministic expression."},
    ]}]
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    config = {
        "campaign_id": "test-campaign", "dataset_id": "test-data",
        "dataset_fingerprint": "test-fingerprint-0001", "pool_id": "TEST",
        "adapter": "tests.test_factor_discovery:create_adapter",
        "allowed_fields": ["post_adjusted.close", "raw.volume"],
        "windows": {
            "development": {"start": "2020-01-01", "end": "2020-01-04"},
            "diagnostic": {"start": "2021-01-01", "end": "2021-01-04"},
            "holdout": {"start": "2022-01-01", "end": "2022-01-04"},
        },
        "llm": {"provider": "replay", "model": "replay", "replay_path": str(replay_path)},
        "evidence": {"minimum_pool_size": 10, "minimum_group_size": 2,
                     "minimum_valid_decisions": 2, "valid_decision_cap": 4,
                     "factor_non_null_ratio": 0.5, "valid_decision_coverage": 0.5,
                     "positive_period_share": 0.5, "monotonicity_threshold": 0.0},
        "target_candidates": 2, "proposals_per_call": 2, "max_rounds": 1,
        "max_proposals": 2, "minimum_free_gb": 0,
        "label_contract": {
            "decision_schedule": "daily", "decision_interval_observations": 1,
            "entry_rule": "next_open", "exit_rule": "fifth_close",
            "holding_horizon_observations": 5, "execution_delay_observations": 1,
            "cost_model": "test", "label_is_net_of_costs": True,
            "overlapping_positions": True,
        },
        "pool_description": "Synthetic point-in-time test pool for workflow verification.",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    plan_dir = tmp_path / "plan"
    run_dir = tmp_path / "run"
    plan = create_discovery_plan(config_path, plan_dir)
    result = run_discovery(plan_dir, run_dir)
    assert result.status == "complete"
    assert result.accepted_candidates == 2
    assert plan.config_hash
    assert (run_dir / "holdout" / "SUCCESS.json").exists()
    calls = (run_dir / "llm_calls.jsonl").read_text(encoding="utf-8")
    assert "holdout" not in calls
    assert "OPENAI_API_KEY" not in "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in run_dir.rglob("*") if path.is_file() and path.suffix != ".parquet"
    )

    parallel_config = dict(config)
    parallel_config["campaign_id"] = "test-campaign-parallel"
    parallel_config["evaluation_workers"] = 2
    parallel_path = tmp_path / "parallel-config.yaml"
    parallel_path.write_text(yaml.safe_dump(parallel_config), encoding="utf-8")
    parallel_plan = tmp_path / "parallel-plan"
    parallel_run = tmp_path / "parallel-run"
    create_discovery_plan(parallel_path, parallel_plan)
    parallel_result = run_discovery(parallel_plan, parallel_run)
    assert parallel_result.accepted_candidates == result.accepted_candidates
    for period in ("development", "diagnostic", "holdout"):
        serial_summary = pd.read_csv(run_dir / period / "factor_summary.csv").sort_values(
            "factor_id"
        ).reset_index(drop=True)
        parallel_summary = pd.read_csv(
            parallel_run / period / "factor_summary.csv"
        ).sort_values("factor_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(serial_summary, parallel_summary)


def test_replay_client_is_bounded():
    batch = CandidateBatch.model_validate({"candidates": [{
        "hypothesis": "a sufficiently descriptive economic hypothesis",
        "direction": 1, "expression": FIELD_CLOSE,
        "rationale": "a sufficiently descriptive implementation rationale",
    }]})
    client = ReplayLLMClient([batch])
    assert len(client.propose(prompt="x", count=1, round_number=1).batch.candidates) == 1
    with pytest.raises(RuntimeError, match="exhausted"):
        client.propose(prompt="x", count=1, round_number=2)
