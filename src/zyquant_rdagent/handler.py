"""Production structured handler powered by RD-Agent's upstream research core."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from .bridge import (
    BridgeExperiment,
    ZyQuantScenario,
    _canonical_json,
    _json_object,
    _load_rdagent,
    _redact,
    assert_no_qlib_modules_loaded,
)


_SYSTEM_PROMPT = """You are the research component of a ZyQuant factor campaign.
Use the supplied point-in-time research evidence to propose economically motivated,
testable experiments. Return exactly one JSON object conforming to the supplied
schema. Never return Python, source code, commands, imports, file paths, network
actions, or changes to the frozen universe, label, costs, holding path, or seeds.
Diagnostic/reference evidence is unavailable by design. Prefer genuinely distinct
hypotheses after reflecting on candidate-level failures, redundancy, annual RankIC,
spread returns, exposure diagnostics, and model incremental evidence.
"""


@dataclass(frozen=True, slots=True)
class StructuredCompletion:
    value: Any
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    repair_attempts: int = 0


class OpenAIResponsesBackend:
    """Strict Structured Outputs backend; the API key remains environment-only."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - isolated env packaging failure
                raise RuntimeError("OpenAI SDK is required in the RD-Agent worker") from exc
            base_url = os.environ.get("OPENAI_BASE_URL")
            client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self.client = client

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        model: str,
        reasoning_effort: str,
    ) -> StructuredCompletion:
        response = self.client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=response_model,
            reasoning={"effort": reasoning_effort},
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("OpenAI Responses returned no parsed structured output")
        usage = getattr(response, "usage", None)
        if usage is not None and hasattr(usage, "model_dump"):
            usage = usage.model_dump(mode="json")
        return StructuredCompletion(
            parsed,
            response_id=getattr(response, "id", None),
            usage=usage if isinstance(usage, dict) else {},
        )


class RDAgentAPIBackendAdapter:
    """Optional adapter for an injected upstream APIBackend (mainly offline parity)."""

    def __init__(self, backend: Any | None = None) -> None:
        if backend is None:
            _load_rdagent()
            backend = importlib.import_module("rdagent.oai.llm_utils").APIBackend()
        self.backend = backend

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        model: str,  # noqa: ARG002
        reasoning_effort: str,  # noqa: ARG002
    ) -> StructuredCompletion:
        raw = self.backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
            json_target_type=dict[str, Any],
        )
        value = response_model.model_validate(json.loads(raw))
        return StructuredCompletion(value)


class StructuredRDAgentHandler:
    """Generate strictly validated ZyQuant contracts with RD-Agent APIBackend.

    RD-Agent imports are delayed until construction. Tests can inject a backend
    and the three upstream proposal classes without installing RD-Agent.
    """

    def __init__(
        self,
        *,
        backend: Any | None = None,
        proposal_types: tuple[type[Any], type[Any], type[Any]] | None = None,
    ) -> None:
        if proposal_types is None:
            _load_rdagent()
        if backend is None:
            backend = OpenAIResponsesBackend()
        if proposal_types is None:
            proposal = importlib.import_module("rdagent.core.proposal")
            proposal_types = (
                proposal.Trace,
                proposal.Hypothesis,
                proposal.HypothesisFeedback,
            )
        self.backend = backend
        self._trace_class, self._hypothesis_class, self._feedback_class = proposal_types
        self._traces: dict[str, Any] = {}
        self._trace_event_ids: dict[str, set[str]] = {}
        assert_no_qlib_modules_loaded()

    def propose_factors(
        self,
        *,
        context: Mapping[str, Any],
        count: int,
        iteration: int,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
        from zyquant.research.agent_discovery.models import (
            FactorCandidateSpec,
            ResearchHypothesis,
        )

        clean_context = self._context(context)
        trace = self._trace(clean_context)
        self._hydrate_trace(clean_context, trace)
        user_prompt = self._factor_prompt(
            clean_context,
            trace,
            count=count,
            iteration=iteration,
            hypothesis_schema=ResearchHypothesis.model_json_schema(),
            candidate_schema=FactorCandidateSpec.model_json_schema(),
        )
        class FactorResponse(BaseModel):
            model_config = ConfigDict(extra="forbid")

            hypotheses: tuple[ResearchHypothesis, ...]
            candidates: tuple[FactorCandidateSpec, ...]

        completion = self._complete(user_prompt, FactorResponse, clean_context)
        response = completion.value
        hypotheses = TypeAdapter(tuple[ResearchHypothesis, ...]).validate_python(
            response.hypotheses
        )
        candidates = TypeAdapter(tuple[FactorCandidateSpec, ...]).validate_python(
            response.candidates
        )
        if not hypotheses or not candidates:
            raise ValueError("RD-Agent must return non-empty hypotheses and candidates")
        if len(candidates) > count:
            raise ValueError("RD-Agent returned more candidates than requested")
        if any(hypothesis.iteration != iteration for hypothesis in hypotheses):
            raise ValueError("RD-Agent hypothesis iteration differs from request")
        hypothesis_ids = {hypothesis.hypothesis_id for hypothesis in hypotheses}
        allowed_fields = set(clean_context.get("allowed_fields", ()))
        candidates, deterministic_repairs = self._repair_field_prefixes(
            candidates, allowed_fields
        )
        if deterministic_repairs:
            completion = StructuredCompletion(
                completion.value,
                response_id=completion.response_id,
                usage=completion.usage,
                repair_attempts=completion.repair_attempts + deterministic_repairs,
            )
        for hypothesis in hypotheses:
            if not set(hypothesis.candidate_fields).issubset(allowed_fields):
                raise ValueError("RD-Agent hypothesis references fields outside whitelist")
        for candidate in candidates:
            if candidate.hypothesis_id not in hypothesis_ids:
                raise ValueError("candidate references an unknown hypothesis")
            candidate.expression.validate_contract(allowed_fields)
        self._record(
            trace,
            hypotheses[0],
            {"hypotheses": hypotheses, "candidates": candidates},
            decision=True,
            reason="Structured factor hypotheses passed ZyQuant schema validation.",
            loop_id=iteration,
        )
        return hypotheses, candidates, self._audit(completion, "factor", iteration)

    @staticmethod
    def _repair_field_prefixes(
        candidates: tuple[Any, ...], allowed_fields: set[str]
    ) -> tuple[tuple[Any, ...], int]:
        """Repair the unambiguous ``factor.factor.foo`` structured-output typo."""

        from zyquant.research.factor_discovery.models import ExpressionSpec, Operator

        def repair(node: ExpressionSpec) -> tuple[ExpressionSpec, bool]:
            changed = False
            args = []
            for child in node.args:
                repaired, child_changed = repair(child)
                args.append(repaired)
                changed = changed or child_changed
            updates: dict[str, Any] = {}
            if tuple(args) != node.args:
                updates["args"] = tuple(args)
            if (
                node.op == Operator.FIELD
                and node.source is not None
                and node.field is not None
                and node.leaf_key not in allowed_fields
                and node.field in allowed_fields
                and node.field.startswith(f"{node.source}.")
            ):
                updates["field"] = node.field.split(".", 1)[1]
                changed = True
            return (node.model_copy(update=updates) if updates else node), changed

        repaired_candidates = []
        repairs = 0
        for candidate in candidates:
            expression, changed = repair(candidate.expression)
            if changed:
                repairs += 1
                candidate = candidate.model_copy(
                    update={
                        "expression": expression,
                        "repair_attempt": min(3, candidate.repair_attempt + 1),
                    }
                )
            repaired_candidates.append(candidate)
        return tuple(repaired_candidates), repairs

    def propose_experiment(
        self,
        *,
        context: Mapping[str, Any],
        experiment_kind: str,
        ordinal: int,
    ) -> tuple[Any, dict[str, Any]]:
        from zyquant.research.agent_discovery.models import JointExperimentSpec

        if experiment_kind not in {"incremental", "joint"}:
            raise ValueError("experiment_kind must be incremental or joint")
        clean_context = self._context(context)
        trace = self._trace(clean_context)
        self._hydrate_trace(clean_context, trace)
        user_prompt = self._experiment_prompt(
            clean_context,
            trace,
            experiment_kind=experiment_kind,
            ordinal=ordinal,
            schema=JointExperimentSpec.model_json_schema(),
        )
        class ExperimentResponse(BaseModel):
            model_config = ConfigDict(extra="forbid")

            experiment: JointExperimentSpec

        completion = self._complete(user_prompt, ExperimentResponse, clean_context)
        spec = JointExperimentSpec.model_validate(completion.value.experiment)
        if spec.experiment_kind != experiment_kind:
            raise ValueError("RD-Agent experiment kind differs from request")
        if spec.campaign_id != clean_context.get("campaign_id"):
            raise ValueError("RD-Agent experiment campaign differs from request")
        if spec.pool_id != clean_context.get("pool_id"):
            raise ValueError("RD-Agent experiment pool differs from request")
        accepted_ids = {
            row.get("factor_id")
            for row in clean_context.get("accepted_factors", ())
            if isinstance(row, Mapping)
        }
        if {factor.factor_id for factor in spec.candidate_factors} - accepted_ids:
            raise ValueError("RD-Agent experiment references factors outside the locked catalog")
        hypothesis = self._make_hypothesis(
            title=f"{experiment_kind} experiment {ordinal}",
            reason=f"Evaluate structured experiment {spec.experiment_id} against development evidence.",
        )
        self._record(
            trace,
            hypothesis,
            {"experiment": spec},
            decision=True,
            reason="Structured joint experiment passed ZyQuant schema validation.",
            loop_id=ordinal,
        )
        return spec, self._audit(completion, experiment_kind, ordinal)

    def _complete(
        self,
        user_prompt: str,
        response_model: type[BaseModel],
        context: Mapping[str, Any],
    ) -> StructuredCompletion:
        agent_settings = context.get("agent", {})
        if not isinstance(agent_settings, Mapping):
            agent_settings = {}
        model = str(
            agent_settings.get("model")
            or context.get("model")
            or os.environ.get("ZYQUANT_RDAGENT_MODEL", "gpt-5.6-sol")
        )
        reasoning_effort = str(
            agent_settings.get("reasoning_effort")
            or context.get("reasoning_effort")
            or os.environ.get("ZYQUANT_RDAGENT_REASONING_EFFORT", "medium")
        )
        budgets = context.get("budgets", {})
        if not isinstance(budgets, Mapping):
            budgets = {}
        max_repairs = int(budgets.get("max_repairs", 3))
        prompt = user_prompt
        result: Any = None
        repairs = 0
        while True:
            try:
                result = self.backend.complete(
                    user_prompt=prompt,
                    system_prompt=_SYSTEM_PROMPT,
                    response_model=response_model,
                    model=model,
                    reasoning_effort=reasoning_effort,
                )
                break
            except ValidationError as exc:
                if repairs >= max_repairs:
                    raise
                repairs += 1
                prompt = (
                    user_prompt
                    + "\n\nThe previous structured response failed schema validation. "
                    + "Return a corrected full response. Validation errors: "
                    + json.dumps(
                        exc.errors(include_input=False),
                        ensure_ascii=False,
                        default=str,
                    )
                )
        assert_no_qlib_modules_loaded()
        if isinstance(result, StructuredCompletion):
            return StructuredCompletion(
                result.value,
                response_id=result.response_id,
                usage=result.usage,
                repair_attempts=repairs,
            )
        if isinstance(result, response_model):
            return StructuredCompletion(result, repair_attempts=repairs)
        return StructuredCompletion(
            response_model.model_validate(result), repair_attempts=repairs
        )

    @staticmethod
    def _context(context: Mapping[str, Any]) -> dict[str, Any]:
        clean = _redact(_json_object(context, name="agent context"))
        if clean.get("feedback_window") != "research_feedback":
            raise ValueError("only research_feedback evidence may enter RD-Agent")
        forbidden = {"diagnostic", "reference", "holdout"} & set(clean)
        if forbidden:
            raise ValueError(f"non-development context is forbidden: {sorted(forbidden)}")
        return clean

    def _trace(self, context: Mapping[str, Any]) -> Any:
        campaign_id = str(context.get("campaign_id", "unknown"))
        trace = self._traces.get(campaign_id)
        if trace is None:
            scenario = ZyQuantScenario(
                {
                    "background": f"ZyQuant {context.get('pool_id', '')} PIT research",
                    "source_data": {
                        "feedback_window": "research_feedback",
                        "allowed_fields": context.get("allowed_fields", ()),
                        "baseline_factor_ids": context.get("baseline_factor_ids", ()),
                    },
                }
            )
            trace = self._trace_class(scen=scenario)
            self._traces[campaign_id] = trace
            self._trace_event_ids[campaign_id] = set()
        return trace

    def _hydrate_trace(self, context: Mapping[str, Any], trace: Any) -> None:
        """Rebuild RD-Agent memory from immutable candidate-level feedback."""

        campaign_id = str(context.get("campaign_id", "unknown"))
        seen = self._trace_event_ids.setdefault(campaign_id, set())
        rows = context.get("experiment_feedback", ())
        if not isinstance(rows, (list, tuple)):
            return
        for row in rows[-60:]:
            if not isinstance(row, Mapping):
                continue
            event_id = str(row.get("experiment_id") or hashlib.sha256(_canonical_json(row)).hexdigest())
            if event_id in seen:
                continue
            errors = row.get("errors", ())
            decision = not bool(errors)
            hypothesis = self._make_hypothesis(
                title=f"Historical experiment {event_id[:16]}",
                reason="Reconstructed from immutable ZyQuant research_feedback evidence.",
            )
            experiment = BridgeExperiment(
                experiment_id=hashlib.sha256(f"trace:{event_id}".encode()).hexdigest(),
                spec={"kind": "historical_feedback", "feedback": _jsonable(row)},
                hypothesis=hypothesis,
            )
            feedback = self._feedback_class(
                reason=(
                    "Historical experiment completed without worker errors."
                    if decision
                    else f"Historical worker errors: {errors}"
                ),
                decision=decision,
                observations=json.dumps(_redact(row), ensure_ascii=False, sort_keys=True),
                hypothesis_evaluation="Use the candidate-level annual and exposure evidence for reflection.",
                new_hypothesis="Prefer a distinct mechanism that addresses observed failures.",
                code_change_summary="",
            )
            trace.sync_dag_parent_and_hist((experiment, feedback), len(trace.hist))
            seen.add(event_id)

    def _make_hypothesis(self, *, title: str, reason: str) -> Any:
        return self._hypothesis_class(
            hypothesis=title,
            reason=reason,
            concise_reason=reason[:240],
            concise_observation="Candidate-level ZyQuant development evidence was reviewed.",
            concise_justification="The proposal is a code-free, strictly validated contract.",
            concise_knowledge="Only research_feedback evidence is eligible.",
        )

    def _record(
        self,
        trace: Any,
        hypothesis: Any,
        result: Mapping[str, Any],
        *,
        decision: bool,
        reason: str,
        loop_id: int,
    ) -> None:
        rd_hypothesis = (
            hypothesis
            if isinstance(hypothesis, self._hypothesis_class)
            else self._make_hypothesis(
                title=str(getattr(hypothesis, "title", "Structured hypothesis")),
                reason=str(getattr(hypothesis, "thesis", reason)),
            )
        )
        experiment = BridgeExperiment(
            experiment_id=hashlib.sha256(_canonical_json(_jsonable(result))).hexdigest(),
            spec={"kind": "rdagent_trace", "result": _jsonable(result)},
            hypothesis=rd_hypothesis,
        )
        feedback = self._feedback_class(
            reason=reason,
            decision=decision,
            observations="ZyQuant strict validation completed.",
            hypothesis_evaluation=reason,
            new_hypothesis="Reflect on candidate-level evidence before the next proposal.",
            code_change_summary="",
        )
        trace.sync_dag_parent_and_hist((experiment, feedback), loop_id)

    @staticmethod
    def _trace_summary(trace: Any) -> list[dict[str, Any]]:
        summary = []
        for experiment, feedback in trace.hist[-12:]:
            summary.append(
                {
                    "hypothesis": getattr(getattr(experiment, "hypothesis", None), "hypothesis", None),
                    "decision": bool(getattr(feedback, "decision", False)),
                    "reason": getattr(feedback, "reason", None),
                }
            )
        return summary

    def _factor_prompt(
        self,
        context: Mapping[str, Any],
        trace: Any,
        *,
        count: int,
        iteration: int,
        hypothesis_schema: Mapping[str, Any],
        candidate_schema: Mapping[str, Any],
    ) -> str:
        payload = {
            "task": "propose_factors",
            "iteration": iteration,
            "candidate_count": count,
            "context": context,
            "rdagent_trace": self._trace_summary(trace),
            "output_contract": {
                "shape": {"hypotheses": ["ResearchHypothesis"], "candidates": ["FactorCandidateSpec"]},
                "ResearchHypothesis": hypothesis_schema,
                "FactorCandidateSpec": candidate_schema,
            },
            "reflection": [
                "Use accepted_factors and experiment_feedback candidate by candidate.",
                "Avoid exact or economically equivalent prior expressions.",
                "Explain an economic mechanism before formalizing its expression.",
                "Use only allowed_fields and declarative operators in the schema.",
                "A field node requires source and field, with no args or window.",
                "For allowed field factor.foo use source='factor' and field='foo'; never put 'factor.' inside field.",
                "rank is a rolling time-series rank, never a cross-sectional rank, and requires window 1..252.",
                "lag, delta, return, mean, std, sum, rolling_min, rolling_max, rank, ema, and corr require window 1..252.",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _experiment_prompt(
        self,
        context: Mapping[str, Any],
        trace: Any,
        *,
        experiment_kind: str,
        ordinal: int,
        schema: Mapping[str, Any],
    ) -> str:
        payload = {
            "task": "propose_experiment",
            "experiment_kind": experiment_kind,
            "ordinal": ordinal,
            "context": context,
            "rdagent_trace": self._trace_summary(trace),
            "output_contract": {"shape": {"experiment": "JointExperimentSpec"}, "schema": schema},
            "fixed_pool_seeds": {
                "CSI300": [20260722, 20260829, 20260916],
                "CSI1000": [20260722, 20260723, 20260724],
            },
            "reflection": [
                "Select only locked accepted_factors present in context.",
                "For incremental experiments keep the frozen baseline and add at most eight factors.",
                "For joint experiments use at most twelve factors and a schema-allowed model recipe.",
                "Use prediction vintages 2020, 2021, and 2022 only.",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _audit(
        completion: StructuredCompletion, action: str, ordinal: int
    ) -> dict[str, Any]:
        canonical = _canonical_json(_jsonable(completion.value))
        return {
            "provider": "rdagent",
            "response_id": completion.response_id or hashlib.sha256(canonical).hexdigest(),
            "status": "completed",
            "usage": completion.usage or {},
            "action": action,
            "ordinal": ordinal,
            "repair_attempts": completion.repair_attempts,
        }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


__all__ = [
    "OpenAIResponsesBackend",
    "RDAgentAPIBackendAdapter",
    "StructuredCompletion",
    "StructuredRDAgentHandler",
]
