from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from .models import FactorCandidateSpec, JointExperimentSpec, ResearchHypothesis


class ReplayAgentClient:
    """Deterministic RD-Agent substitute for tests, review, and resume drills.

    The replay file is a JSON object with optional ``factor_rounds`` and
    ``experiments`` arrays. Each factor round contains ``hypotheses`` and
    ``candidates``. Consumed positions are explicit so a resumed workflow can
    seek to the state journal without repeating proposals.
    """

    def __init__(
        self,
        factor_rounds: Sequence[Mapping[str, Any]] = (),
        experiments: Sequence[Mapping[str, Any] | JointExperimentSpec] = (),
    ) -> None:
        self._rounds = [dict(item) for item in factor_rounds]
        self._experiments = list(experiments)
        self._round_position = 0
        self._experiment_position = 0

    @classmethod
    def from_path(cls, path: str | Path) -> "ReplayAgentClient":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("agent replay must contain a JSON object")
        return cls(payload.get("factor_rounds", ()), payload.get("experiments", ()))

    def seek(self, *, factor_rounds: int = 0, experiments: int = 0) -> None:
        if not 0 <= factor_rounds <= len(self._rounds):
            raise ValueError("factor replay position is outside available rounds")
        if not 0 <= experiments <= len(self._experiments):
            raise ValueError("experiment replay position is outside available experiments")
        self._round_position = factor_rounds
        self._experiment_position = experiments

    def propose_factors(
        self, *, context: Mapping[str, Any], count: int, iteration: int
    ) -> tuple[tuple[ResearchHypothesis, ...], tuple[FactorCandidateSpec, ...], dict[str, Any]]:
        del context
        if self._round_position >= len(self._rounds):
            raise RuntimeError("factor replay responses exhausted")
        payload = self._rounds[self._round_position]
        self._round_position += 1
        hypotheses = TypeAdapter(tuple[ResearchHypothesis, ...]).validate_python(
            payload.get("hypotheses", ())
        )
        candidates = TypeAdapter(tuple[FactorCandidateSpec, ...]).validate_python(
            payload.get("candidates", ())
        )[:count]
        return hypotheses, candidates, {
            "provider": "replay",
            "iteration": iteration,
            "response_id": f"replay-factor-{self._round_position}",
            "status": "completed",
            "usage": {},
        }

    def propose_experiment(
        self, *, context: Mapping[str, Any], experiment_kind: str, ordinal: int
    ) -> tuple[JointExperimentSpec, dict[str, Any]]:
        del context, experiment_kind
        if self._experiment_position >= len(self._experiments):
            raise RuntimeError("experiment replay responses exhausted")
        spec = JointExperimentSpec.model_validate(
            self._experiments[self._experiment_position]
        )
        self._experiment_position += 1
        return spec, {
            "provider": "replay",
            "ordinal": ordinal,
            "response_id": f"replay-experiment-{self._experiment_position}",
            "status": "completed",
            "usage": {},
        }


__all__ = ["ReplayAgentClient"]
