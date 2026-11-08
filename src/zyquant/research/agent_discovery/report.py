from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .executor import sha256_file, sha256_json
from .models import AgentCampaignPlan, AgentCampaignResult, ExperimentFeedback


_SECRET_PATTERNS = (
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+\S+"),
    re.compile(r"(?i)(?:api[_-]?key|password|access[_-]?token)\s*[:=]\s*[^\s\"']+"),
)


def verify_agent_campaign(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        plan = AgentCampaignPlan.model_validate_json(
            (root / "plan.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        return {"status": "invalid", "errors": [f"plan: {type(exc).__name__}: {exc}"]}
    expected_config_hash = sha256_json(plan.config.model_dump(mode="json"))
    if expected_config_hash != plan.config_hash:
        errors.append("plan config hash mismatch")

    result: AgentCampaignResult | None = None
    success_path = root / "SUCCESS.json"
    if success_path.exists():
        try:
            result = AgentCampaignResult.model_validate_json(
                success_path.read_text(encoding="utf-8")
            )
            if result.plan_id != plan.plan_id:
                errors.append("result plan_id mismatch")
            manifest_path = root / "result_manifest.json"
            if sha256_file(manifest_path) != result.result_manifest_hash:
                errors.append("result manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected in manifest.get("artifacts", {}).items():
                path = root / relative
                if not path.exists() or sha256_file(path) != expected:
                    errors.append(f"artifact hash mismatch: {relative}")
        except Exception as exc:
            errors.append(f"result: {type(exc).__name__}: {exc}")
    else:
        warnings.append("campaign has no terminal SUCCESS.json")

    feedback_path = root / "experiment_feedback.jsonl"
    if feedback_path.exists():
        for number, line in enumerate(feedback_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                continue
            try:
                feedback = ExperimentFeedback.model_validate_json(line)
                if feedback.feedback_window != "research_feedback":
                    errors.append(f"non-development feedback at line {number}")
            except Exception as exc:
                errors.append(f"invalid feedback line {number}: {exc}")

    pareto_path = root / "PARETO_LOCK.json"
    if pareto_path.exists():
        pareto = json.loads(pareto_path.read_text(encoding="utf-8"))
        if pareto.get("selection_window") != "research_feedback":
            errors.append("Pareto routes were not locked from research_feedback")
        if pareto.get("diagnostic_feedback_allowed") is not False:
            errors.append("diagnostic feedback barrier is not frozen")
        if pareto.get("reference_feedback_allowed") is not False:
            errors.append("reference feedback barrier is not frozen")

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".parquet", ".feather"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            errors.append(f"possible secret in artifact: {path.relative_to(root)}")

    return {
        "status": "valid" if not errors else "invalid",
        "plan_id": plan.plan_id,
        "campaign_id": plan.config.campaign_id,
        "terminal": result is not None,
        "errors": errors,
        "warnings": warnings,
    }


def write_agent_report(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    plan = AgentCampaignPlan.model_validate_json(
        (root / "plan.json").read_text(encoding="utf-8")
    )
    success_path = root / "SUCCESS.json"
    result = (
        AgentCampaignResult.model_validate_json(success_path.read_text(encoding="utf-8"))
        if success_path.exists() else None
    )
    factor_rows = _read_jsonl(root / "accepted_factors.jsonl")
    feedback_rows = _read_jsonl(root / "experiment_feedback.jsonl")
    pareto = (
        json.loads((root / "PARETO_LOCK.json").read_text(encoding="utf-8"))
        if (root / "PARETO_LOCK.json").exists() else {"experiment_ids": []}
    )
    verification = verify_agent_campaign(root)
    rows = [
        "# ZyQuant RD-Agent Research Report",
        "",
        f"- Campaign: `{plan.config.campaign_id}`",
        f"- Pool: `{plan.config.pool_id}`",
        f"- Status: `{result.status if result else 'running'}`",
        f"- Raw proposals: {result.raw_proposals if result else 'running'}",
        f"- Accepted unique factors: {len(factor_rows)}",
        f"- Completed development experiments: {len(feedback_rows)}",
        f"- Locked Pareto routes: {len(pareto.get('experiment_ids', []))}",
        f"- Integrity verification: `{verification['status']}`",
        "",
        "## Evidence boundaries",
        "",
        "- Only `research_feedback` evidence was supplied to RD-Agent.",
        "- Diagnostic (2023–2025) and 2026 reference results are locked evaluations and never selection feedback.",
        "- 2023–2026 is labelled `previously_observed_reference`, not a fresh holdout.",
        "- Monotonicity, HAC, and FDR are evidence fields, not solitary rejection gates.",
        "",
        "## Pareto routes",
        "",
    ]
    ids = pareto.get("experiment_ids", [])
    rows.extend([f"- `{item}`" for item in ids] or ["- No development-eligible route was locked."])
    if result and result.status == "incomplete":
        rows.extend([
            "", "## Incomplete campaign", "",
            "The factor target was not reached within the frozen proposal budget; "
            "the run is intentionally reported as incomplete.",
        ])
    destination = root / "REPORT.md"
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return destination


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


__all__ = ["verify_agent_campaign", "write_agent_report"]
