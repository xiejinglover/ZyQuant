"""Entrypoint for the isolated RD-Agent v1.0.0 process."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .bridge import (
    _canonical_json,
    _read_json_object,
    _write_immutable,
    assert_no_qlib_modules_loaded,
    verify_rdagent_install,
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _load_handler() -> Any:
    target = os.environ.get("ZYQUANT_RDAGENT_HANDLER")
    if not target:
        from .handler import StructuredRDAgentHandler

        return StructuredRDAgentHandler()
    if ":" not in target:
        raise RuntimeError("ZYQUANT_RDAGENT_HANDLER must use module:attribute syntax")
    module_name, attribute = target.split(":", 1)
    if module_name == "qlib" or module_name.startswith("qlib.") or ".qlib" in module_name:
        raise RuntimeError("Qlib handler modules are forbidden")
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory() if callable(factory) else factory


def execute(request: Mapping[str, Any], handler: Any) -> dict[str, Any]:
    request_id = str(
        request.get("request_id") or hashlib.sha256(_canonical_json(dict(request))).hexdigest()
    )
    operation = request.get("operation")
    phase = request.get("phase")
    if operation == "propose_factors" or phase == "factor_discovery":
        result = handler.propose_factors(
            context=request["context"],
            count=request.get("count", request.get("requested_candidates")),
            iteration=request.get("iteration", 1),
        )
        if isinstance(result, tuple) and len(result) == 3:
            hypotheses, candidates, audit = result
            result = {
                "hypotheses": hypotheses,
                "candidates": candidates,
                "audit": audit,
            }
    elif operation == "propose_experiment" or phase in {
        "incremental_experiment",
        "joint_experiment",
    }:
        experiment_kind = request.get("experiment_kind")
        if experiment_kind is None and isinstance(phase, str):
            experiment_kind = phase.removesuffix("_experiment")
        result = handler.propose_experiment(
            context=request["context"],
            experiment_kind=experiment_kind,
            ordinal=request.get("ordinal", 1),
        )
        if isinstance(result, tuple) and len(result) == 2:
            experiment, audit = result
            result = {"experiment": experiment, "audit": audit}
    else:
        raise ValueError(f"unsupported RD-Agent operation: {operation}")
    result = _jsonable(result)
    if not isinstance(result, Mapping):
        raise TypeError("RD-Agent handler must return a JSON object")
    payload = json.loads(_canonical_json(dict(result)))
    payload["request_id"] = request_id
    payload.setdefault("audit", {})
    payload["audit"].update(
        {
            "provider": "rdagent",
            "rdagent_version": "1.0.0",
            "qlib_loaded": False,
        }
    )
    payload.setdefault("response_id", payload["audit"].get("response_id"))
    payload.setdefault("usage", payload["audit"].get("usage", {}))
    payload.setdefault("status", payload["audit"].get("status", "completed"))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path)
    args = parser.parse_args(argv)
    verify_rdagent_install()
    assert_no_qlib_modules_loaded()
    request = _read_json_object(args.request)
    # Bind the response to the exact immutable request before enriching the
    # in-memory context with settings from the separately frozen plan.
    request.setdefault(
        "request_id", hashlib.sha256(_canonical_json(dict(request))).hexdigest()
    )
    if args.plan_path is not None:
        plan = _read_json_object(args.plan_path)
        config = plan.get("config", {})
        agent = config.get("agent", {}) if isinstance(config, Mapping) else {}
        budgets = config.get("budgets", {}) if isinstance(config, Mapping) else {}
        context = request.get("context", {})
        if isinstance(context, Mapping) and isinstance(agent, Mapping) and "agent" not in context:
            request = dict(request)
            request["context"] = {
                **dict(context),
                "agent": dict(agent),
                "budgets": dict(budgets) if isinstance(budgets, Mapping) else {},
            }
    response = execute(request, _load_handler())
    assert_no_qlib_modules_loaded()
    _write_immutable(args.response, response)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
