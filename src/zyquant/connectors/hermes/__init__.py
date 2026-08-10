from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping

from zyquant.core.exceptions import DataContractError
from zyquant.core.plugins import PluginMetadata

from .acquisition import (
    AcquisitionState,
    HermesAcquisitionRequest,
    HermesCredentials,
    HermesDataAdapter,
    HermesMySQLClient,
    HermesResourceLimits,
)
from .normalize import HermesAcquisitionPublisher, HermesCanonicalizer


def _date(value: Any, default: date) -> date:
    if value is None:
        return default
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def request_from_mapping(
    value: Mapping[str, Any] | None,
) -> HermesAcquisitionRequest:
    payload = dict(value or {})
    limits = dict(payload.pop("limits", {}))
    for source, target in (
        ("connections", "max_connections"),
        ("target_memory_gib", "target_memory_gib"),
        ("hard_memory_gib", "hard_memory_gib"),
    ):
        if source in payload:
            limits[target] = payload.pop(source)
    return HermesAcquisitionRequest(
        job_id=str(payload.get("job_id", "hermes-cn-a-2010-20260724")),
        start_date=_date(payload.get("start_date"), date(2010, 1, 1)),
        end_date=_date(payload.get("end_date"), date(2026, 7, 24)),
        financial_warmup_start=_date(
            payload.get("financial_warmup_start"), date(2009, 1, 1)
        ),
        exchanges=tuple(payload.get("exchanges", ("XSHG", "XSHE", "XBEI"))),
        root=Path(payload.get("root", "data")),
        limits=HermesResourceLimits(**limits),
    )


class HermesConnector:
    """Resumable Hermes connector exposed through the generic data CLI."""

    def ingest(self, request: Mapping[str, Any] | None = None):
        raise DataContractError(
            "Hermes is resumable; use 'zyq data acquire --source hermes'"
        )

    def acquire(
        self, action: str, request: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        resolved = request_from_mapping(request)
        if action == "status":
            return HermesDataAdapter.status(resolved.root, resolved.job_id)
        if action not in {"run", "resume"}:
            raise DataContractError(f"unsupported Hermes action: {action}")
        resume = action == "resume"
        acquisition = HermesDataAdapter().run(resolved, resume=resume)
        normalization = HermesCanonicalizer(resolved).run(resume=resume)
        return {
            "status": "normalized",
            "acquisition": acquisition,
            "normalization": normalization,
        }

    def publish(
        self,
        data_root: str | Path,
        dataset_id: str,
        request: Mapping[str, Any] | None = None,
    ):
        job_id = str(dict(request or {}).get("job_id", "")).strip()
        if not job_id:
            raise DataContractError("Hermes publish request requires 'job_id'")
        return HermesAcquisitionPublisher(data_root).publish(job_id, dataset_id)


def create_adapter(
    request: Mapping[str, Any] | None = None,
) -> HermesConnector:
    return HermesConnector()


setattr(create_adapter, "plugin_metadata", PluginMetadata(
    name="hermes",
    version="2.0.0",
    kind="data",
    input_schema="data.source/hermes@1",
    output_schema="data.canonical_batch@1",
    optional_dependencies=("PyMySQL>=1.1,<2",),
))


__all__ = [
    "AcquisitionState", "HermesAcquisitionPublisher",
    "HermesAcquisitionRequest", "HermesCanonicalizer", "HermesConnector",
    "HermesCredentials", "HermesDataAdapter", "HermesMySQLClient",
    "HermesResourceLimits", "create_adapter", "request_from_mapping",
]
