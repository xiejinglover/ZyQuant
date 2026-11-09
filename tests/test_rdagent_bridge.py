from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from zyquant.research.agent_discovery import FactorCandidateSpec, ResearchHypothesis

from zyquant_rdagent import (
    RDAGENT_COMMIT,
    RDAGENT_VERSION,
    BridgeExperiment,
    BridgeFeedback,
    OpenAIResponsesBackend,
    RDAgentCompatibilityError,
    RDAgentProcessClient,
    RDAgentUnavailableError,
    ZyQuantExperimentSummarizer,
    ZyQuantFactorModelRunner,
    ZyQuantHypothesis2Experiment,
    ZyQuantScenario,
    StructuredCompletion,
    StructuredRDAgentHandler,
    assert_no_qlib_modules_loaded,
    create_rdagent_loop,
)


def test_module_import_is_optional_and_pinned() -> None:
    assert RDAGENT_VERSION == "1.0.0"
    assert RDAGENT_COMMIT == "484776c211e4fbbeef03e0ec00d6bbee7362a4f4"
    assert not any(name.startswith("rdagent.app.qlib") for name in sys.modules)
    assert not any(name == "qlib" or name.startswith("qlib.") for name in sys.modules)


def test_scenario_is_redacted_and_matches_rdagent_protocol() -> None:
    scenario = ZyQuantScenario(
        {
            "background": "CSI1000 PIT research",
            "dataset": {"id": "panel-v1", "api_key": "must-not-leak"},
            "runtime": {"python": "3.11", "engine": "zyquant"},
        }
    )
    assert scenario.background == "CSI1000 PIT research"
    assert "must-not-leak" not in scenario.rich_style_description
    assert "[REDACTED]" in scenario.get_source_data_desc()
    assert "zyquant" in scenario.get_runtime_environment()


def test_converter_accepts_json_and_rejects_generated_code() -> None:
    converter = ZyQuantHypothesis2Experiment()
    hypothesis = SimpleNamespace(
        experiment_id="trial-001",
        experiment_spec={"kind": "factor", "factors": [{"op": "field", "field": "close"}]},
    )
    experiment = converter.convert(hypothesis)
    assert experiment.experiment_id == "trial-001"
    assert experiment.spec["kind"] == "factor"

    with pytest.raises(ValueError, match="generated code"):
        converter.convert({"spec": {"kind": "factor", "python_code": "open('/tmp/x')"}})
    with pytest.raises(ValueError, match="structured experiment_spec"):
        converter.convert({"hypothesis": "write a Python factor"})


def test_runner_uses_immutable_json_and_safe_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = BridgeExperiment("trial-safe", {"kind": "factor", "seed": 7})
    runner = ZyQuantFactorModelRunner(
        run_dir=tmp_path,
        executable="/opt/zyquant/bin/zyq",
        timeout_seconds=30,
    )
    observed: dict[str, object] = {}

    def fake_run(argv: object, **kwargs: object) -> SimpleNamespace:
        observed["argv"] = argv
        observed.update(kwargs)
        assert isinstance(argv, tuple)
        response = Path(argv[-1])
        response.write_text(
            json.dumps(
                {
                    "experiment_id": "trial-safe",
                    "status": "complete",
                    "decision": True,
                    "candidate_feedback": {"rank_ic": 0.03},
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    completed = runner.develop(experiment)
    assert observed["shell"] is False
    assert observed["check"] is False
    assert observed["argv"] == (
        "/opt/zyquant/bin/zyq",
        "research",
        "agent",
        "worker",
        "--request",
        str(tmp_path / "requests" / "trial-safe.json"),
        "--response",
        str(tmp_path / "responses" / "trial-safe.json"),
    )
    request = json.loads(completed.request_path.read_text(encoding="utf-8"))
    assert request["spec"] == {"kind": "factor", "seed": 7}
    assert completed.request_sha256
    assert completed.result["decision"] is True
    assert "OPENAI_API_KEY" not in observed["env"]

    with pytest.raises(FileExistsError, match="immutable contract"):
        runner.develop(BridgeExperiment("trial-safe", {"kind": "factor", "seed": 8}))


def test_summarizer_works_without_rdagent(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = importlib.import_module("zyquant_rdagent.bridge")
    monkeypatch.setattr(
        bridge,
        "_load_rdagent",
        lambda: (_ for _ in ()).throw(RDAgentUnavailableError("not installed")),
    )
    experiment = BridgeExperiment("summary-1", {"kind": "factor"})
    experiment.result = {
        "experiment_id": "summary-1",
        "status": "complete",
        "decision": True,
        "candidate_feedback": {"coverage": 0.98, "rank_ic_by_year": {"2022": 0.04}},
    }
    feedback = ZyQuantExperimentSummarizer().generate_feedback(experiment)
    assert isinstance(feedback, BridgeFeedback)
    assert feedback.decision is True
    assert "rank_ic_by_year" in feedback.observations


def test_process_client_uses_isolated_file_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = RDAgentProcessClient(
        tmp_path,
        command=("/opt/rdagent/bin/python", "-m", "zyquant_rdagent.worker"),
    )

    def fake_run(argv: object, **kwargs: object) -> SimpleNamespace:
        assert isinstance(argv, tuple)
        assert argv[:3] == (
            "/opt/rdagent/bin/python",
            "-m",
            "zyquant_rdagent.worker",
        )
        request_path = Path(argv[-3])
        response_path = Path(argv[-1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request_id = __import__("hashlib").sha256(
            json.dumps(
                request,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        response_path.write_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "hypotheses": [],
                    "candidates": [],
                    "audit": {"response_id": "rdagent-1"},
                }
            ),
            encoding="utf-8",
        )
        assert kwargs["shell"] is False
        assert "OPENAI_API_KEY" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    hypotheses, candidates, audit = client.propose_factors(
        context={"campaign_id": "csi1000"}, count=4, iteration=1
    )
    assert hypotheses == ()
    assert candidates == ()
    assert audit["provider"] == "rdagent"

    with pytest.raises(ValueError, match="secret-bearing"):
        client.propose_factors(
            context={"OPENAI_API_KEY": "do-not-write"}, count=4, iteration=2
        )


def test_no_qlib_guard_and_loop_factory_with_fake_rdagent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = importlib.import_module("zyquant_rdagent.bridge")
    fake_rdagent = SimpleNamespace(__version__="1.0.0", __commit__=RDAGENT_COMMIT)
    monkeypatch.setattr(bridge.importlib, "import_module", lambda name: fake_rdagent)
    monkeypatch.setattr(
        bridge.importlib.metadata,
        "version",
        lambda name: "1.0.0",
    )

    class FakeRDLoop:
        __module__ = "rdagent.components.workflow.rd_loop"

        def __init__(self, setting: object) -> None:
            self.setting = setting

    setting = object()
    loop = create_rdagent_loop(setting, loop_class=FakeRDLoop)
    assert loop.setting is setting

    with pytest.raises(RDAgentCompatibilityError, match="Qlib modules"):
        assert_no_qlib_modules_loaded({"qlib": object()})
    with pytest.raises(RDAgentCompatibilityError, match="Qlib modules"):
        assert_no_qlib_modules_loaded({"rdagent.utils.qlib": object()})

    class QlibLoop:
        __module__ = "rdagent.app.qlib_rd_loop.quant"

    with pytest.raises(RDAgentCompatibilityError, match="Qlib loop class"):
        create_rdagent_loop(setting, loop_class=QlibLoop)


def test_structured_handler_uses_mock_llm_and_upstream_trace_types() -> None:
    class FakeHypothesis:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class FakeFeedback:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class FakeTrace:
        def __init__(self, scen: object) -> None:
            self.scen = scen
            self.hist: list[tuple[object, object]] = []

        def sync_dag_parent_and_hist(self, item: tuple[object, object], loop_id: int) -> None:
            assert loop_id in {0, 2}
            self.hist.append(item)

    class MockStructuredBackend:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def complete(self, **kwargs: object) -> StructuredCompletion:
            self.calls.append(kwargs)
            response_model = kwargs["response_model"]
            if len(self.calls) == 1:
                response_model.model_validate(
                    {
                        "hypotheses": [],
                        "candidates": [
                            {
                                "candidate_name": "invalid_rank",
                                "hypothesis_id": "hyp_0002",
                                "expression": {
                                    "op": "rank",
                                    "args": [
                                        {"op": "field", "source": "raw", "field": "close"}
                                    ],
                                },
                                "direction": 1,
                                "rationale": "This deliberately invalid first response tests repair.",
                            }
                        ],
                    }
                )
            value = response_model.model_validate(
                {
                    "hypotheses": [
                        {
                            "hypothesis_id": "hyp_0002",
                            "iteration": 2,
                            "title": "Volume-price exhaustion reversal",
                            "thesis": "Abnormal volume without matching price progress may indicate exhausted demand.",
                            "economic_mechanism": "Crowded trading consumes near-term liquidity and is followed by price reversal.",
                            "expected_direction": -1,
                            "candidate_fields": ["raw.close", "raw.volume"],
                            "parent_hypothesis_ids": [],
                        }
                    ],
                    "candidates": [
                        {
                            "candidate_name": "volume_price_exhaustion",
                            "hypothesis_id": "hyp_0002",
                            "expression": {
                                "op": "safe_div",
                                "args": [
                                    {
                                        "op": "mean",
                                        # Semantically outside the campaign
                                        # contract, but still valid structured
                                        # output. The core workflow must reject
                                        # it without crashing the Agent worker.
                                        "window": 200,
                                        "args": [
                                            {
                                                "op": "lag",
                                                "window": 115,
                                                "args": [
                                                    {
                                                        "op": "field",
                                                        "source": "raw",
                                                        "field": "volume",
                                                    }
                                                ],
                                            }
                                        ],
                                    },
                                    {
                                        "op": "abs",
                                        "args": [
                                            {
                                                "op": "return",
                                                "window": 20,
                                                "args": [
                                                    {"op": "field", "source": "raw", "field": "close"}
                                                ],
                                            }
                                        ],
                                    },
                                ],
                            },
                            "direction": -1,
                            "rationale": "High participation with little price progress measures potential exhaustion.",
                            "repair_attempt": 0,
                        }
                    ],
                }
            )
            return StructuredCompletion(value, response_id="resp_mock", usage={"input_tokens": 10})

    backend = MockStructuredBackend()
    handler = StructuredRDAgentHandler(
        backend=backend,
        proposal_types=(FakeTrace, FakeHypothesis, FakeFeedback),
    )
    hypotheses, candidates, audit = handler.propose_factors(
        context={
            "campaign_id": "csi1000-agent",
            "pool_id": "CSI1000",
            "feedback_window": "research_feedback",
            "allowed_fields": ["raw.close", "raw.volume"],
            "baseline_factor_ids": ["baseline_1"],
            "accepted_factors": [],
            "experiment_feedback": [
                {"factor_id": "old", "failure_reasons": ["numeric_duplicate_correlation"]}
            ],
            "agent": {"model": "gpt-5.6-sol", "reasoning_effort": "medium"},
            "budgets": {"max_repairs": 3},
        },
        count=1,
        iteration=2,
    )
    assert hypotheses[0].hypothesis_id == "hyp_0002"
    assert candidates[0].candidate_name == "volume_price_exhaustion"
    with pytest.raises(ValueError, match="lookback 315 exceeds 252"):
        candidates[0].expression.validate_contract({"raw.close", "raw.volume"})
    assert audit["response_id"] == "resp_mock"
    assert audit["repair_attempts"] == 1
    assert len(backend.calls) == 2
    assert backend.calls[1]["model"] == "gpt-5.6-sol"
    assert backend.calls[1]["reasoning_effort"] == "medium"
    assert "failed schema validation" in str(backend.calls[1]["user_prompt"])
    assert not any(name == "qlib" or name.startswith("qlib.") for name in sys.modules)


def test_structured_handler_repairs_unambiguous_double_field_prefix() -> None:
    candidate = FactorCandidateSpec.model_validate(
        {
            "candidate_name": "double_prefix",
            "hypothesis_id": "hyp_0001",
            "expression": {
                "op": "field",
                "source": "factor",
                "field": "factor.ema20x_alpha_20",
            },
            "direction": 1,
            "rationale": "The duplicated namespace is an unambiguous structured-output typo.",
        }
    )
    repaired, count = StructuredRDAgentHandler._repair_field_prefixes(
        (candidate,), {"factor.ema20x_alpha_20"}
    )
    assert count == 1
    assert repaired[0].expression.leaf_key == "factor.ema20x_alpha_20"
    assert repaired[0].repair_attempt == 1


def test_structured_handler_repairs_hypothesis_field_prefixes() -> None:
    hypothesis = ResearchHypothesis(
        hypothesis_id="hyp_0001",
        iteration=1,
        title="Canonical field metadata",
        thesis="Metadata must use the same registered field namespace.",
        economic_mechanism="Canonical names keep the research trace auditable.",
        expected_direction=1,
        candidate_fields=("factor.factor.alpha", "beta"),
    )
    repaired, count = StructuredRDAgentHandler._repair_hypothesis_field_prefixes(
        (hypothesis,), {"factor.alpha", "factor.beta"}
    )
    assert count == 1
    assert repaired[0].candidate_fields == ("factor.alpha", "factor.beta")


def test_structured_handler_binds_hypothesis_metadata_to_candidate_ast() -> None:
    hypothesis = ResearchHypothesis(
        hypothesis_id="hyp_0001",
        iteration=1,
        title="Alias-safe metadata",
        thesis="The executable AST is authoritative for registered fields.",
        economic_mechanism="A friendly narrative alias must not abort a valid round.",
        expected_direction=1,
        candidate_fields=("friendly_cashflow_alias",),
    )
    candidate = FactorCandidateSpec.model_validate(
        {
            "candidate_name": "registered_cashflow",
            "hypothesis_id": "hyp_0001",
            "expression": {
                "op": "field",
                "source": "factor",
                "field": "registered_cashflow",
            },
            "direction": 1,
            "rationale": "Use the registered PIT factor referenced by the AST.",
        }
    )
    repaired, count = StructuredRDAgentHandler._repair_hypothesis_fields_from_candidates(
        (hypothesis,), (candidate,), {"factor.registered_cashflow"}
    )
    assert count == 1
    assert repaired[0].candidate_fields == ("factor.registered_cashflow",)


def test_default_loop_uses_only_upstream_loopbase_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = importlib.import_module("zyquant_rdagent.bridge")

    class FakeLoopMeta(type):
        pass

    class FakeLoopBase:
        EXCEPTION_KEY = "_EXCEPTION"
        LOOP_IDX_KEY = "_LOOP_IDX"

        def __init__(self) -> None:
            self.loop_base_initialized = True

    class FakeTrace:
        def __init__(self, scen: object) -> None:
            self.scen = scen
            self.hist: list[object] = []

        def sync_dag_parent_and_hist(self, item: object, loop_id: int) -> None:
            self.hist.append((item, loop_id))

    real_import = bridge.importlib.import_module

    def fake_import(name: str) -> object:
        if name == "rdagent.utils.workflow":
            return SimpleNamespace(LoopBase=FakeLoopBase, LoopMeta=FakeLoopMeta)
        if name == "rdagent.core.proposal":
            return SimpleNamespace(Trace=FakeTrace)
        return real_import(name)

    monkeypatch.setattr(bridge, "_load_rdagent", lambda: SimpleNamespace(__version__="1.0.0"))
    monkeypatch.setattr(bridge.importlib, "import_module", fake_import)

    class Generator:
        def gen(self, trace: object) -> str:
            return "hypothesis"

    class Converter:
        def convert(self, hypothesis: object, trace: object) -> BridgeExperiment:
            return BridgeExperiment("loop-exp", {"kind": "factor"}, hypothesis=hypothesis)

    class Runner:
        def develop(self, experiment: BridgeExperiment) -> BridgeExperiment:
            return experiment

    class Summarizer:
        def generate_feedback(self, *args: object, **kwargs: object) -> BridgeFeedback:
            return BridgeFeedback("ok", True, "ok", "ok", "next")

    setting = SimpleNamespace(
        scenario=ZyQuantScenario({"background": "test"}),
        hypothesis_gen=Generator(),
        hypothesis2experiment=Converter(),
        runner=Runner(),
        summarizer=Summarizer(),
    )
    loop = create_rdagent_loop(setting)
    assert loop.loop_base_initialized is True
    assert isinstance(loop.trace, FakeTrace)
    assert "rdagent.utils.qlib" not in sys.modules


def test_worker_accepts_core_phase_contract_and_preserves_request_id() -> None:
    worker = importlib.import_module("zyquant_rdagent.worker")

    class Handler:
        def propose_factors(self, **kwargs: object) -> tuple[list[object], list[object], dict[str, object]]:
            assert kwargs["count"] == 4
            assert kwargs["iteration"] == 1
            return [], [], {"response_id": "upstream-response", "usage": {"input_tokens": 3}}

    response = worker.execute(
        {
            "contract_version": "2.2",
            "request_id": "f" * 64,
            "phase": "factor_discovery",
            "requested_candidates": 4,
            "context": {"feedback_window": "research_feedback"},
        },
        Handler(),
    )
    assert response["request_id"] == "f" * 64
    assert response["response_id"] == "upstream-response"
    assert response["usage"] == {"input_tokens": 3}
    assert response["audit"]["qlib_loaded"] is False


def test_openai_backend_uses_responses_parse_with_pydantic_schema() -> None:
    class Output(BaseModel):
        value: int

    observed: dict[str, object] = {}

    class Responses:
        def parse(self, **kwargs: object) -> SimpleNamespace:
            observed.update(kwargs)
            return SimpleNamespace(
                id="resp_strict",
                output_parsed=Output(value=7),
                usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 12}),
            )

    backend = OpenAIResponsesBackend(SimpleNamespace(responses=Responses()))
    completion = backend.complete(
        system_prompt="system",
        user_prompt="user",
        response_model=Output,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
    )
    assert completion.value == Output(value=7)
    assert completion.response_id == "resp_strict"
    assert observed["text_format"] is Output
    assert observed["reasoning"] == {"effort": "medium"}
