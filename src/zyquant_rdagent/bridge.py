"""Filesystem bridge between Microsoft RD-Agent and a ZyQuant worker.

Only RD-Agent's generic workflow is loaded, and only on demand.  Qlib's
application/scenario modules are never imported.  Experiments cross the
process boundary as canonical JSON instead of generated Python code.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

RDAGENT_VERSION = "1.0.0"
RDAGENT_COMMIT = "484776c211e4fbbeef03e0ec00d6bbee7362a4f4"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FORBIDDEN_SPEC_KEYS = frozenset(
    {"argv", "command", "code", "executable", "python", "python_code", "script", "shell"}
)
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "auth_token",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "openai_api_key",
    }
)


class RDAgentUnavailableError(RuntimeError):
    """RD-Agent is absent from the isolated worker environment."""


class RDAgentCompatibilityError(RuntimeError):
    """The installed RD-Agent does not match the audited upstream release."""


def _normalise_version(version: str) -> str:
    return version.strip().removeprefix("v").split("+", 1)[0]


def _load_rdagent() -> ModuleType:
    """Load the package lazily and enforce the audited release."""

    try:
        module = importlib.import_module("rdagent")
    except ModuleNotFoundError as exc:
        if exc.name == "rdagent":
            raise RDAgentUnavailableError(
                "RD-Agent is not installed. Install Microsoft RD-Agent v1.0.0 "
                f"from commit {RDAGENT_COMMIT} in the isolated Python 3.11 worker."
            ) from exc
        raise

    try:
        installed = importlib.metadata.version("rdagent")
    except importlib.metadata.PackageNotFoundError:
        installed = str(getattr(module, "__version__", ""))
    if _normalise_version(installed) != RDAGENT_VERSION:
        shown = installed or "unknown"
        raise RDAgentCompatibilityError(
            f"RD-Agent {shown} is installed; exactly {RDAGENT_VERSION} "
            f"({RDAGENT_COMMIT}) is required."
        )
    return module


def assert_no_qlib_modules_loaded(modules: Mapping[str, object] | None = None) -> None:
    """Fail if the Python process imported Qlib or an RD-Agent Qlib application."""

    loaded = modules if modules is not None else sys.modules
    forbidden = sorted(
        name
        for name in loaded
        if name == "qlib"
        or name.startswith("qlib.")
        or (
            name.startswith("rdagent.")
            and re.search(r"(?:^|\.)qlib(?:[_.]|$)", name) is not None
        )
    )
    if forbidden:
        raise RDAgentCompatibilityError(
            "Qlib modules are forbidden in the ZyQuant RD-Agent worker: " + ", ".join(forbidden)
        )


def verify_rdagent_install() -> dict[str, Any]:
    """Verify the pin and return an auditable, secret-free compatibility record."""

    if sys.version_info[:2] != (3, 11):
        raise RDAgentCompatibilityError(
            "The isolated RD-Agent v1.0.0 worker must run on Python 3.11; "
            f"found {sys.version_info.major}.{sys.version_info.minor}."
        )
    module = _load_rdagent()
    assert_no_qlib_modules_loaded()
    try:
        installed = importlib.metadata.version("rdagent")
    except importlib.metadata.PackageNotFoundError:
        installed = str(getattr(module, "__version__", RDAGENT_VERSION))
    actual_commit = getattr(module, "__commit__", None)
    if actual_commit is not None and str(actual_commit) != RDAGENT_COMMIT:
        raise RDAgentCompatibilityError(
            f"RD-Agent commit {actual_commit} does not match required {RDAGENT_COMMIT}."
        )
    return {
        "distribution": "rdagent",
        "version": installed,
        "required_version": RDAGENT_VERSION,
        "required_commit": RDAGENT_COMMIT,
        "reported_commit": actual_commit,
        "qlib_loaded": False,
    }


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("experiment contract must be finite, JSON-serializable data") from exc


def _json_object(value: Any, *, name: str) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    payload = dict(value)
    _canonical_json(payload)
    return payload


def _reject_generated_code(value: Any, path: str = "experiment") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key in _FORBIDDEN_SPEC_KEYS:
                raise ValueError(f"generated code/command field is forbidden: {path}.{raw_key}")
            _reject_generated_code(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_generated_code(child, f"{path}[{index}]")


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if _is_sensitive_key(key)
                else _redact(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact(child) for child in value]
    return value


def _is_sensitive_key(key: object) -> bool:
    normalised = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    return normalised in _SENSITIVE_KEYS or normalised.endswith("_password") or normalised.endswith(
        "_secret"
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON contract: {path}") from exc
    return _json_object(value, name=str(path))


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> str:
    data = _canonical_json(dict(payload)) + b"\n"
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    except FileExistsError:
        existing = path.read_bytes()
        if existing != data:
            raise FileExistsError(f"immutable contract already exists with different content: {path}")
    return digest


@dataclass(slots=True)
class BridgeExperiment:
    """A duck-typed RD-Agent experiment carrying only a structured contract."""

    experiment_id: str
    spec: dict[str, Any]
    hypothesis: Any = None
    based_experiments: Sequence[Any] = ()
    sub_tasks: Sequence[Any] = ()
    result: dict[str, Any] | None = None
    request_path: Path | None = None
    response_path: Path | None = None
    request_sha256: str | None = None
    running_seconds: float | None = None

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.experiment_id):
            raise ValueError("experiment_id must be a safe portable identifier")
        self.spec = _json_object(self.spec, name="experiment spec")
        _reject_generated_code(self.spec)


@dataclass(frozen=True, slots=True)
class BridgeFeedback:
    reason: str
    decision: bool
    observations: str
    hypothesis_evaluation: str
    new_hypothesis: str
    code_change_summary: str = "Structured contract only; no generated code executed."


class ZyQuantScenario:
    """RD-Agent Scenario backed by a frozen, redacted ZyQuant descriptor."""

    def __init__(self, descriptor: Mapping[str, Any] | Path | str | None = None) -> None:
        if descriptor is None:
            configured = os.environ.get("ZYQUANT_RDAGENT_SCENARIO")
            if not configured:
                raise ValueError(
                    "scenario descriptor is required (or set ZYQUANT_RDAGENT_SCENARIO)"
                )
            descriptor = Path(configured)
        if isinstance(descriptor, (str, Path)):
            descriptor = _read_json_object(Path(descriptor))
        self._descriptor = _redact(_json_object(descriptor, name="scenario descriptor"))

    @property
    def descriptor(self) -> dict[str, Any]:
        return json.loads(_canonical_json(self._descriptor))

    @property
    def background(self) -> str:
        return str(self._descriptor.get("background", "ZyQuant PIT quantitative research"))

    def get_source_data_desc(self, task: Any = None) -> str:  # noqa: ARG002
        source = self._descriptor.get("source_data", self._descriptor.get("dataset", {}))
        return json.dumps(source, sort_keys=True, ensure_ascii=False)

    @property
    def source_data(self) -> str:
        return self.get_source_data_desc()

    @property
    def rich_style_description(self) -> str:
        return json.dumps(self._descriptor, sort_keys=True, ensure_ascii=False, indent=2)

    def get_scenario_all_desc(
        self,
        task: Any = None,  # noqa: ARG002
        filtered_tag: str | None = None,  # noqa: ARG002
        simple_background: bool | None = None,
    ) -> str:
        if simple_background:
            return self.background
        return self.rich_style_description

    def get_runtime_environment(self) -> str:
        runtime = self._descriptor.get(
            "runtime", {"engine": "ZyQuant worker", "contract": "canonical JSON"}
        )
        return json.dumps(runtime, sort_keys=True, ensure_ascii=False)

    @property
    def experiment_setting(self) -> str | None:
        setting = self._descriptor.get("experiment_setting")
        return None if setting is None else json.dumps(setting, sort_keys=True, ensure_ascii=False)


class ZyQuantHypothesis2Experiment:
    """Convert an RD-Agent hypothesis containing a structured spec to an experiment."""

    def __init__(
        self,
        validator: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._validator = validator

    def convert(self, hypothesis: Any, trace: Any = None) -> BridgeExperiment:  # noqa: ARG002
        if isinstance(hypothesis, Mapping):
            source = dict(hypothesis)
        else:
            source = vars(hypothesis)
        raw_spec = source.get("experiment_spec", source.get("spec"))
        if raw_spec is None:
            raise ValueError(
                "hypothesis must include a structured experiment_spec; free-form code is not accepted"
            )
        spec = _json_object(raw_spec, name="experiment_spec")
        _reject_generated_code(spec)
        if self._validator is not None:
            validated = self._validator(spec)
            spec = _json_object(validated, name="validated experiment_spec")
        experiment_id = str(source.get("experiment_id") or spec.get("experiment_id") or "")
        if not experiment_id:
            experiment_id = hashlib.sha256(_canonical_json(spec)).hexdigest()
        return BridgeExperiment(experiment_id=experiment_id, spec=spec, hypothesis=hypothesis)


class ZyQuantFactorModelRunner:
    """Execute a contract with ``zyq`` using argv and ``shell=False`` only."""

    def __init__(
        self,
        scen: ZyQuantScenario | None = None,
        *,
        run_dir: Path | str | None = None,
        executable: str | Path | None = None,
        timeout_seconds: int = 21_600,
        worker_subcommand: Sequence[str] = ("research", "agent", "worker"),
    ) -> None:
        self.scen = scen
        descriptor = scen.descriptor if scen is not None else {}
        bridge = descriptor.get("bridge", {}) if isinstance(descriptor.get("bridge", {}), Mapping) else {}
        configured_dir = run_dir or bridge.get("run_dir") or os.environ.get("ZYQUANT_RDAGENT_RUN_DIR")
        if configured_dir is None:
            raise ValueError("run_dir is required for the immutable experiment contract")
        self.run_dir = Path(configured_dir).expanduser().resolve()
        self.executable = str(
            executable or bridge.get("executable") or os.environ.get("ZYQUANT_EXECUTABLE", "zyq")
        )
        if not self.executable or any(character in self.executable for character in "\n\r\0"):
            raise ValueError("invalid ZyQuant executable")
        self.timeout_seconds = int(timeout_seconds)
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self.worker_subcommand = tuple(str(part) for part in worker_subcommand)
        if not self.worker_subcommand or any(
            not part or part.startswith("-") or any(c in part for c in "\n\r\0")
            for part in self.worker_subcommand
        ):
            raise ValueError("worker_subcommand must contain safe positional argv components")

    def command_for(self, request_path: Path, response_path: Path) -> tuple[str, ...]:
        return (
            self.executable,
            *self.worker_subcommand,
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        )

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        allowed = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "PYTHONPATH", "SYSTEMROOT", "WINDIR")
        return {key: os.environ[key] for key in allowed if key in os.environ}

    def develop(self, exp: BridgeExperiment) -> BridgeExperiment:
        if not isinstance(exp, BridgeExperiment):
            raise TypeError("runner accepts only BridgeExperiment contracts")
        request_path = self.run_dir / "requests" / f"{exp.experiment_id}.json"
        response_path = self.run_dir / "responses" / f"{exp.experiment_id}.json"
        request = {
            "contract_version": "1.0",
            "experiment_id": exp.experiment_id,
            "spec": exp.spec,
        }
        exp.request_sha256 = _write_immutable(request_path, request)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        if response_path.exists():
            response = _read_json_object(response_path)
        else:
            started = time.monotonic()
            completed = subprocess.run(
                self.command_for(request_path, response_path),
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=self._worker_environment(),
            )
            exp.running_seconds = time.monotonic() - started
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "worker failed").strip()[-2000:]
                raise RuntimeError(
                    f"ZyQuant worker failed with exit code {completed.returncode}: {detail}"
                )
            if not response_path.is_file():
                raise RuntimeError("ZyQuant worker completed without a response contract")
            response = _read_json_object(response_path)
            response_path.chmod(0o444)
        if response.get("experiment_id") != exp.experiment_id:
            raise ValueError("response experiment_id does not match request")
        exp.request_path = request_path
        exp.response_path = response_path
        exp.result = response
        return exp


class RDAgentProcessClient:
    """Core-side AgentClient that never imports RD-Agent.

    The configured command must point at the isolated Python 3.11 worker.  It
    receives only paths to immutable contracts and implements the same two
    methods as ZyQuant's deterministic ReplayAgentClient.
    """

    def __init__(
        self,
        run_dir: Path | str,
        *,
        command: Sequence[str] | None = None,
        timeout_seconds: int = 180,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        if command is None:
            isolated_python = os.environ.get("ZYQUANT_RDAGENT_PYTHON")
            if not isolated_python:
                raise ValueError(
                    "command or ZYQUANT_RDAGENT_PYTHON is required; the core environment "
                    "must not import RD-Agent"
                )
            command = (
                isolated_python,
                "-m",
                "zyquant_rdagent.worker",
            )
        self.command = tuple(str(part) for part in command)
        if not self.command or any(
            not part or any(character in part for character in "\n\r\0") for part in self.command
        ):
            raise ValueError("RD-Agent worker command contains an invalid argv component")
        self.timeout_seconds = int(timeout_seconds)
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")

    @staticmethod
    def _assert_no_secrets(value: Any, path: str = "request") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).casefold()
                if _is_sensitive_key(lowered):
                    raise ValueError(f"secret-bearing field is forbidden in Agent contract: {path}.{key}")
                RDAgentProcessClient._assert_no_secrets(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                RDAgentProcessClient._assert_no_secrets(child, f"{path}[{index}]")

    def _exchange(self, operation: str, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = {"contract_version": "1.0", "operation": operation, **dict(request)}
        self._assert_no_secrets(payload)
        digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
        request_path = self.run_dir / "agent_requests" / f"{digest}.json"
        response_path = self.run_dir / "agent_responses" / f"{digest}.json"
        _write_immutable(request_path, payload)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        if not response_path.exists():
            completed = subprocess.run(
                (*self.command, "--request", str(request_path), "--response", str(response_path)),
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=self._agent_environment(),
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "worker failed").strip()[-2000:]
                raise RuntimeError(
                    f"RD-Agent worker failed with exit code {completed.returncode}: {detail}"
                )
            if not response_path.is_file():
                raise RuntimeError("RD-Agent worker completed without a response contract")
            response_path.chmod(0o444)
        response = _read_json_object(response_path)
        if response.get("request_id") != digest:
            raise ValueError("RD-Agent response request_id does not match the immutable request")
        return response

    @staticmethod
    def _agent_environment() -> dict[str, str]:
        # Secrets are inherited by the isolated LLM process but are never put
        # in argv, request JSON, audit records, stdout, or error messages.
        allowed = (
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "PYTHONPATH",
            "SYSTEMROOT",
            "WINDIR",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "ZYQUANT_RDAGENT_HANDLER",
        )
        return {key: os.environ[key] for key in allowed if key in os.environ}

    def propose_factors(
        self, *, context: Mapping[str, Any], count: int, iteration: int
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
        response = self._exchange(
            "propose_factors",
            {"context": dict(context), "count": count, "iteration": iteration},
        )
        from pydantic import TypeAdapter

        from zyquant.research.agent_discovery.models import (
            FactorCandidateSpec,
            ResearchHypothesis,
        )

        hypotheses = TypeAdapter(tuple[ResearchHypothesis, ...]).validate_python(
            response.get("hypotheses", ())
        )
        candidates = TypeAdapter(tuple[FactorCandidateSpec, ...]).validate_python(
            response.get("candidates", ())
        )
        if len(candidates) > count:
            raise ValueError("RD-Agent returned more candidates than requested")
        audit = _json_object(response.get("audit", {}), name="agent audit")
        audit.setdefault("provider", "rdagent")
        audit.setdefault("request_id", response["request_id"])
        return hypotheses, candidates, audit

    def propose_experiment(
        self, *, context: Mapping[str, Any], experiment_kind: str, ordinal: int
    ) -> tuple[Any, dict[str, Any]]:
        response = self._exchange(
            "propose_experiment",
            {
                "context": dict(context),
                "experiment_kind": experiment_kind,
                "ordinal": ordinal,
            },
        )
        from zyquant.research.agent_discovery.models import JointExperimentSpec

        experiment = JointExperimentSpec.model_validate(response.get("experiment"))
        if experiment.experiment_kind != experiment_kind:
            raise ValueError("RD-Agent experiment kind does not match the request")
        audit = _json_object(response.get("audit", {}), name="agent audit")
        audit.setdefault("provider", "rdagent")
        audit.setdefault("request_id", response["request_id"])
        return experiment, audit


class ZyQuantExperimentSummarizer:
    """Turn candidate-level ZyQuant evidence into RD-Agent feedback."""

    def __init__(self, scen: ZyQuantScenario | None = None) -> None:
        self.scen = scen

    @staticmethod
    def _feedback_fields(exp: BridgeExperiment, exception: Exception | None) -> dict[str, Any]:
        if exception is not None:
            return {
                "reason": f"Experiment failed: {exception}",
                "decision": False,
                "observations": str(exception),
                "hypothesis_evaluation": "The structured experiment did not complete.",
                "new_hypothesis": "Repair the structured specification without generating code.",
            }
        result = _json_object(exp.result or {}, name="experiment result")
        status = str(result.get("status", "unknown"))
        decision = bool(result.get("decision", status in {"accepted", "complete", "success"}))
        evidence = result.get("candidate_feedback", result.get("evidence", result))
        observations = json.dumps(_redact(evidence), sort_keys=True, ensure_ascii=False)
        return {
            "reason": str(result.get("reason", f"ZyQuant worker status: {status}")),
            "decision": decision,
            "observations": observations,
            "hypothesis_evaluation": str(
                result.get("hypothesis_evaluation", "See candidate-level ZyQuant evidence.")
            ),
            "new_hypothesis": str(
                result.get("new_hypothesis", "Use the evidence to propose a distinct structured experiment.")
            ),
        }

    def generate_feedback(
        self,
        exp: BridgeExperiment,
        trace: Any = None,  # noqa: ARG002
        exception: Exception | None = None,
    ) -> Any:
        fields = self._feedback_fields(exp, exception)
        try:
            _load_rdagent()
            module = importlib.import_module("rdagent.core.proposal")
            feedback_class = getattr(module, "HypothesisFeedback")
        except RDAgentUnavailableError:
            return BridgeFeedback(**fields)
        feedback = feedback_class(code_change_summary="", **fields)
        assert_no_qlib_modules_loaded()
        return feedback


def _component(setting: Any, name: str, scenario: Any = None) -> Any:
    value = getattr(setting, name, None)
    if value is None:
        raise ValueError(f"RD loop setting is missing component: {name}")
    if isinstance(value, str):
        value = _import_reference(value)
    if isinstance(value, type):
        return value(scenario) if scenario is not None else value()
    return value


def _import_reference(reference: str) -> Any:
    normalised = reference.replace(":", ".")
    if re.search(r"(?:^|\.)qlib(?:[_.]|$)", normalised):
        raise RDAgentCompatibilityError(f"Qlib component is forbidden: {reference}")
    try:
        module_name, attribute = normalised.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(f"component must be a module-qualified reference: {reference}") from exc
    value = getattr(importlib.import_module(module_name), attribute)
    assert_no_qlib_modules_loaded()
    return value


def _build_zyquant_loop_class() -> type[Any]:
    """Build the loop lazily from upstream's Qlib-free workflow primitives."""

    workflow = importlib.import_module("rdagent.utils.workflow")
    proposal = importlib.import_module("rdagent.core.proposal")
    loop_base = getattr(workflow, "LoopBase")
    loop_meta = getattr(workflow, "LoopMeta")
    trace_class = getattr(proposal, "Trace")

    class _ZyQuantRDLoop(loop_base, metaclass=loop_meta):  # type: ignore[valid-type,misc,metaclass]
        def __init__(self, setting: Any) -> None:
            scenario = getattr(setting, "scenario", None) or getattr(setting, "scen", None)
            if scenario is None:
                scenario = ZyQuantScenario()
            elif isinstance(scenario, str):
                scenario = _import_reference(scenario)
            elif isinstance(scenario, type):
                scenario = scenario()
            if isinstance(scenario, type):
                scenario = scenario()
            self.scenario = scenario
            self.hypothesis_gen = _component(setting, "hypothesis_gen", scenario)
            self.hypothesis2experiment = _component(setting, "hypothesis2experiment")
            self.runner = _component(setting, "runner", scenario)
            self.summarizer = _component(setting, "summarizer", scenario)
            self.trace = trace_class(scen=scenario)
            super().__init__()

        def propose(self, prev_out: dict[str, Any]) -> Any:  # noqa: ARG002
            return self.hypothesis_gen.gen(self.trace)

        def convert(self, prev_out: dict[str, Any]) -> Any:
            return self.hypothesis2experiment.convert(prev_out["propose"], self.trace)

        def running(self, prev_out: dict[str, Any]) -> Any:
            return self.runner.develop(prev_out["convert"])

        def feedback(self, prev_out: dict[str, Any]) -> Any:
            return self.summarizer.generate_feedback(
                prev_out["running"],
                self.trace,
                exception=prev_out.get(self.EXCEPTION_KEY),
            )

        def record(self, prev_out: dict[str, Any]) -> None:
            experiment = prev_out.get("running") or prev_out.get("convert")
            self.trace.sync_dag_parent_and_hist(
                (experiment, prev_out["feedback"]),
                prev_out[self.LOOP_IDX_KEY],
            )

    _ZyQuantRDLoop.__name__ = "ZyQuantRDLoop"
    _ZyQuantRDLoop.__qualname__ = "ZyQuantRDLoop"
    return _ZyQuantRDLoop


class ZyQuantRDLoop:
    """Lazy constructor for a Qlib-free loop built on RD-Agent LoopBase/Trace."""

    def __new__(cls, setting: Any) -> Any:
        del cls
        _load_rdagent()
        assert_no_qlib_modules_loaded()
        loop = _build_zyquant_loop_class()(setting)
        assert_no_qlib_modules_loaded()
        return loop


def create_rdagent_loop(prop_setting: Any, *, loop_class: type[Any] | None = None) -> Any:
    """Instantiate the ZyQuant loop from upstream Qlib-free core primitives.

    ``loop_class`` is an injection seam for offline tests.  Production never
    imports upstream's ``components.workflow.rd_loop`` because v1.0.0 couples
    that otherwise-generic class to ``rdagent.utils.qlib``.
    """

    _load_rdagent()
    assert_no_qlib_modules_loaded()
    if loop_class is None:
        return ZyQuantRDLoop(prop_setting)
    module_name = str(getattr(loop_class, "__module__", ""))
    if module_name == "qlib" or module_name.startswith("qlib.") or ".qlib" in module_name:
        raise RDAgentCompatibilityError(f"Qlib loop class is forbidden: {module_name}")
    loop = loop_class(prop_setting)
    assert_no_qlib_modules_loaded()
    return loop
