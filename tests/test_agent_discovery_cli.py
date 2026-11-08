from __future__ import annotations

import json
from pathlib import Path

import yaml

from zyquant.cli.main import main, parser

from tests.test_agent_discovery_workflow import workflow_config


def test_agent_cli_shape() -> None:
    args = parser().parse_args([
        "research", "agent", "verify", "--run-dir", "/tmp/campaign"
    ])
    assert args.command == "research"
    assert args.research_command == "agent"
    assert args.agent_command == "verify"


def test_agent_cli_plan(tmp_path: Path, capsys) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"factor_rounds": [], "experiments": []}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(workflow_config(replay).model_dump(mode="json")), encoding="utf-8"
    )
    output = tmp_path / "plan"
    assert main([
        "research", "agent", "plan", "--config", str(config),
        "--output-dir", str(output),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "planned"
    assert (output / "frozen_spec.json").exists()

