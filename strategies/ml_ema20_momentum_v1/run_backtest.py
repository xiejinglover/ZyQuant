from __future__ import annotations

import argparse
import json
from pathlib import Path

from zyquant.config import load_config
from zyquant.experiment import ExperimentStore
from zyquant.workflow import WorkflowRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the phase-1 EMA20 momentum backtest."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("server_backtest.yaml"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    with ExperimentStore(config.experiment_database) as store:
        result = WorkflowRunner(
            config.output_root,
            store,
            project_root=Path(__file__).resolve().parents[2],
        ).run(config)
    print(json.dumps({
        "run_id": result.run_id,
        "metrics": result.metrics,
        "output": str(config.output_root / result.run_id),
    }, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
