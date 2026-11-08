from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import AgentCampaignPlan, ExperimentFeedback, JointExperimentSpec


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, payload: Any, *, immutable: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    if destination.exists():
        if immutable and destination.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"immutable artifact differs: {destination}")
        if immutable:
            return
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def redact_payload(value: Any) -> Any:
    """Recursively remove values that must never enter a campaign artifact."""
    sensitive = ("api_key", "apikey", "authorization", "bearer", "password", "secret", "token")
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]" if any(term in str(key).lower() for term in sensitive)
                else redact_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_payload(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"sk-(?:proj-)?[A-Za-z0-9_-]{12,}", "[REDACTED]", value)
        value = re.sub(
            r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[REDACTED]", value
        )
        return value
    return value


class ZyQuantExperimentExecutor:
    """Execute a frozen experiment through an immutable filesystem contract.

    A worker receives ``--request <json> --output-dir <directory>``. Commands are
    argv sequences and are always launched with ``shell=False``. The worker must
    atomically publish ``feedback.json`` plus any Parquet artifacts before writing
    ``SUCCESS.json``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        worker_argv: Sequence[str] | None = None,
        timeout_seconds: int = 86_400,
        runner: Callable[[JointExperimentSpec, Path], ExperimentFeedback | Mapping[str, Any]] | None = None,
    ) -> None:
        if worker_argv is not None and not worker_argv:
            raise ValueError("worker_argv cannot be empty")
        self.root = Path(root).expanduser().resolve()
        self.worker_argv = tuple(worker_argv) if worker_argv is not None else None
        self.timeout_seconds = timeout_seconds
        self.runner = runner

    def execute(self, spec: JointExperimentSpec) -> ExperimentFeedback:
        experiment_dir = self.root / "experiments" / spec.experiment_id
        request_path = experiment_dir / "request.json"
        atomic_write_json(request_path, spec.model_dump(mode="json"), immutable=True)

        success_path = experiment_dir / "SUCCESS.json"
        if success_path.exists():
            return self._load_verified(experiment_dir, spec)

        if self.runner is not None:
            feedback = self.runner(spec, experiment_dir)
            feedback = ExperimentFeedback.model_validate(feedback)
            atomic_write_json(
                experiment_dir / "feedback.json",
                redact_payload(feedback.model_dump(mode="json")),
                immutable=True,
            )
        elif self.worker_argv is not None:
            replacements = {
                "request": str(request_path),
                "output_dir": str(experiment_dir),
                "experiment_path": str(experiment_dir),
                "spec_path": str(request_path),
                "run_dir": str(self.root),
                "plan_path": str(self.root / "plan.json"),
                "config_path": str(self.root / "plan.json"),
            }
            argv = [argument.format(**replacements) for argument in self.worker_argv]
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
                env=os.environ.copy(),
            )
            atomic_write_json(experiment_dir / "worker_audit.json", redact_payload({
                "argv_program": self.worker_argv[0],
                "returncode": completed.returncode,
                "stdout": completed.stdout[-20_000:],
                "stderr": completed.stderr[-20_000:],
            }))
            if completed.returncode:
                raise RuntimeError(
                    f"experiment worker failed for {spec.experiment_id}: "
                    f"exit {completed.returncode}"
                )
        else:
            raise RuntimeError("no experiment runner configured")

        feedback_path = experiment_dir / "feedback.json"
        if not feedback_path.exists():
            raise RuntimeError("experiment worker did not publish feedback.json")
        feedback = ExperimentFeedback.model_validate_json(
            feedback_path.read_text(encoding="utf-8")
        )
        if feedback.experiment_id != spec.experiment_id:
            raise ValueError("experiment feedback ID does not match frozen request")
        manifest = self._manifest(experiment_dir, spec)
        atomic_write_json(success_path, manifest, immutable=True)
        return feedback

    def execute_locked(self, spec: JointExperimentSpec, window: str) -> dict[str, Any]:
        if window not in {"diagnostic", "reference", "formal_account"}:
            raise ValueError("locked evaluation window is not registered")
        destination = self.root / "locked_evaluations" / window / spec.experiment_id
        request_path = destination / "request.json"
        atomic_write_json(request_path, {
            "schema_version": "2.2",
            "experiment_id": spec.experiment_id,
            "evaluation_window": window,
            "feedback_allowed": False,
            "spec": spec.model_dump(mode="json"),
        }, immutable=True)
        success_path = destination / "SUCCESS.json"
        if success_path.exists():
            return json.loads(success_path.read_text(encoding="utf-8"))
        if self.worker_argv is None:
            raise RuntimeError("no locked-evaluation worker configured")
        replacements = {
            "request": str(request_path), "output_dir": str(destination),
            "experiment_path": str(destination), "spec_path": str(request_path),
            "run_dir": str(self.root), "plan_path": str(self.root / "plan.json"),
            "config_path": str(self.root / "plan.json"),
        }
        argv = [argument.format(**replacements) for argument in self.worker_argv]
        completed = subprocess.run(
            argv, check=False, capture_output=True, text=True,
            timeout=self.timeout_seconds, shell=False, env=os.environ.copy(),
        )
        atomic_write_json(destination / "worker_audit.json", redact_payload({
            "argv_program": argv[0], "returncode": completed.returncode,
            "stdout": completed.stdout[-20_000:], "stderr": completed.stderr[-20_000:],
        }))
        if completed.returncode:
            raise RuntimeError(
                f"locked {window} worker failed for {spec.experiment_id}: "
                f"exit {completed.returncode}"
            )
        result_path = destination / "result.json"
        if not result_path.exists():
            alternative = destination / "feedback.json"
            if alternative.exists():
                result_path = alternative
            else:
                raise RuntimeError(f"locked {window} worker did not publish result.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = {
            "schema_version": "2.2", "experiment_id": spec.experiment_id,
            "evaluation_window": window, "feedback_allowed": False,
            "result_sha256": sha256_file(result_path),
        }
        atomic_write_json(success_path, manifest, immutable=True)
        return {**manifest, "result": redact_payload(result)}

    def _load_verified(
        self, experiment_dir: Path, spec: JointExperimentSpec
    ) -> ExperimentFeedback:
        success = json.loads((experiment_dir / "SUCCESS.json").read_text(encoding="utf-8"))
        expected = self._manifest(experiment_dir, spec)
        if success != expected:
            raise ValueError(f"experiment artifact hash mismatch: {spec.experiment_id}")
        return ExperimentFeedback.model_validate_json(
            (experiment_dir / "feedback.json").read_text(encoding="utf-8")
        )

    @staticmethod
    def _manifest(experiment_dir: Path, spec: JointExperimentSpec) -> dict[str, Any]:
        artifacts: dict[str, str] = {}
        for path in sorted(experiment_dir.rglob("*")):
            if path.is_file() and path.name != "SUCCESS.json":
                artifacts[str(path.relative_to(experiment_dir))] = sha256_file(path)
        return {
            "schema_version": "2.2",
            "experiment_id": spec.experiment_id,
            "feedback_window": "research_feedback",
            "artifacts": artifacts,
        }


__all__ = [
    "ZyQuantExperimentExecutor",
    "atomic_write_json",
    "redact_payload",
    "sha256_file",
    "sha256_json",
    "run_worker_contract",
]


def run_worker_contract(request_path: str | Path, response_path: str | Path) -> dict[str, Any]:
    """Hidden core worker used by the isolated RD-Agent runner."""
    request_source = Path(request_path).expanduser().resolve()
    request = json.loads(request_source.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or request.get("contract_version") != "1.0":
        raise ValueError("unsupported experiment worker contract")
    spec = JointExperimentSpec.model_validate(request.get("spec"))
    if request.get("experiment_id") != spec.experiment_id:
        raise ValueError("worker request experiment_id mismatch")
    run_dir = request_source.parent.parent
    plan = AgentCampaignPlan.model_validate_json(
        (run_dir / "plan.json").read_text(encoding="utf-8")
    )
    if spec.campaign_id != plan.config.campaign_id or spec.pool_id != plan.config.pool_id:
        raise ValueError("worker experiment differs from frozen campaign")
    settings = plan.config.experiment_executor
    executor = ZyQuantExperimentExecutor(
        run_dir, worker_argv=settings.argv, timeout_seconds=settings.timeout_seconds
    )
    feedback = executor.execute(spec)
    response = {
        "contract_version": "1.0",
        "experiment_id": spec.experiment_id,
        "status": "complete" if not feedback.errors else "failed",
        "decision": not feedback.errors,
        **feedback.model_dump(mode="json"),
    }
    atomic_write_json(response_path, response, immutable=True)
    return response
