#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from zyquant.connectors.jqdata import JQDataAdapter, JQDataRequest
from zyquant.data import SnapshotPublisher


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Publish a bounded JQData request as a ZyQuant snapshot",
    )
    command.add_argument("--root", type=Path, default=Path("data"))
    command.add_argument("--dataset-id", default="jqdata-sample-2025")
    command.add_argument("--request", type=Path)
    return command


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        request = (
            JQDataRequest.from_file(args.request)
            if args.request is not None
            else JQDataRequest.sample_2025()
        )
        snapshot = SnapshotPublisher(args.root).publish_adapter(
            args.dataset_id,
            JQDataAdapter(),
            request=request,
        )
        payload = {
            "status": "published",
            "dataset_id": snapshot.metadata.dataset_id,
            "as_of_date": snapshot.metadata.as_of_date.isoformat(),
            "fingerprint": snapshot.metadata.fingerprint,
            "path": str(snapshot.path),
        }
        status = 0
    except Exception as exc:
        message = str(exc)
        for name in ("JQDATA_USERNAME", "JQDATA_PASSWORD"):
            secret = os.environ.get(name)
            if secret:
                message = message.replace(secret, "***REDACTED***")
        payload = {
            "status": "error",
            "type": type(exc).__name__,
            "message": message,
        }
        status = 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
