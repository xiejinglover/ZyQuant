from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import CandidateBatch, LLMSettings


SYSTEM_PROMPT = """You are a quantitative factor research agent.
Return only the requested structured candidate batch. Every factor must express a
clear economic hypothesis using only the supplied field whitelist and operators.
Never emit Python, SQL, shell commands, cross-sectional transforms, future data,
portfolio rules, or performance claims. Prefer parsimonious expressions and vary
economic families rather than making cosmetic parameter changes.
"""


@dataclass(frozen=True)
class LLMCallResult:
    batch: CandidateBatch
    audit: dict[str, Any]


class LLMClient(Protocol):
    def propose(
        self,
        *,
        prompt: str,
        count: int,
        round_number: int,
    ) -> LLMCallResult: ...


class ReplayLLMClient:
    """Deterministic client for CI, review, and resumability tests."""

    def __init__(self, batches: Sequence[CandidateBatch]):
        self._batches = list(batches)
        self._position = 0

    @classmethod
    def from_path(cls, path: str | Path) -> "ReplayLLMClient":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_batches = payload if isinstance(payload, list) else payload["batches"]
        return cls([CandidateBatch.model_validate(item) for item in raw_batches])

    def propose(self, *, prompt: str, count: int, round_number: int) -> LLMCallResult:
        if self._position >= len(self._batches):
            raise RuntimeError("replay responses exhausted")
        batch = self._batches[self._position]
        self._position += 1
        trimmed = CandidateBatch(candidates=batch.candidates[:count])
        return LLMCallResult(trimmed, {
            "provider": "replay",
            "round": round_number,
            "response_id": f"replay-{self._position}",
            "status": "completed",
            "usage": {},
        })

    def seek(self, position: int) -> None:
        if position < 0 or position > len(self._batches):
            raise ValueError("replay position is outside available responses")
        self._position = position


class OpenAIResponsesClient:
    """OpenAI Responses API client using SDK-native Pydantic structured output."""

    def __init__(self, settings: LLMSettings):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be configured in the host environment")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install zyquant[ai] to use the OpenAI provider") from exc
        kwargs: dict[str, Any] = {
            "api_key": os.environ["OPENAI_API_KEY"],
            "timeout": settings.timeout_seconds,
            "max_retries": 0,
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._client = OpenAI(**kwargs)
        self._settings = settings

    def propose(self, *, prompt: str, count: int, round_number: int) -> LLMCallResult:
        last_error: Exception | None = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                response = self._client.responses.parse(
                    model=self._settings.model,
                    reasoning={"effort": self._settings.reasoning_effort},
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    text_format=CandidateBatch,
                )
                if getattr(response, "status", None) != "completed":
                    raise RuntimeError(
                        f"OpenAI response incomplete: {getattr(response, 'status', None)}"
                    )
                parsed = getattr(response, "output_parsed", None)
                if parsed is None:
                    refusal = self._refusal(response)
                    raise RuntimeError(f"OpenAI response has no parsed output: {refusal}")
                batch = CandidateBatch.model_validate(parsed)
                if len(batch.candidates) > count:
                    batch = CandidateBatch(candidates=batch.candidates[:count])
                usage = getattr(response, "usage", None)
                if usage is not None and hasattr(usage, "model_dump"):
                    usage = usage.model_dump(mode="json")
                return LLMCallResult(batch, {
                    "provider": "openai",
                    "model": self._settings.model,
                    "round": round_number,
                    "attempt": attempt + 1,
                    "response_id": getattr(response, "id", None),
                    "status": getattr(response, "status", None),
                    "usage": usage or {},
                })
            except Exception as exc:  # SDK error classes remain an optional dependency
                last_error = exc
                if attempt >= self._settings.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(
            f"OpenAI request failed after {self._settings.max_retries + 1} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    @staticmethod
    def _refusal(response: Any) -> str:
        for output in getattr(response, "output", ()):
            for item in getattr(output, "content", ()):
                refusal = getattr(item, "refusal", None)
                if refusal:
                    return str(refusal)
        return "unknown refusal or schema failure"


def make_llm_client(settings: LLMSettings) -> LLMClient:
    if settings.provider == "replay":
        return ReplayLLMClient.from_path(settings.replay_path or "")
    return OpenAIResponsesClient(settings)
