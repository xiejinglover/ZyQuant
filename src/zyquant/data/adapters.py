from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import pandas as pd

from zyquant.core.exceptions import DataContractError
from .contracts import FINANCIAL_TABLES


@dataclass(frozen=True)
class CanonicalBatch:
    tables: Mapping[str, pd.DataFrame]
    vendor_factors: pd.DataFrame | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)


class DataSourceAdapter(Protocol):
    def ingest(self, request: Mapping[str, Any] | None = None) -> CanonicalBatch: ...


DataSourceFactory = Callable[
    [Mapping[str, Any] | None], DataSourceAdapter
]


class DirectoryDataAdapter:
    """Read canonical CSV/Parquet inputs before snapshot publication."""

    SOURCE_TABLES = (
        "instruments", "trade_calendar", "daily_raw", "corporate_actions",
        "universe_membership", "industry_membership", "market_rules",
    )

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()

    def ingest(self, request=None) -> CanonicalBatch:
        tables = {name: self._read(name) for name in self.SOURCE_TABLES}
        for name in FINANCIAL_TABLES:
            optional = self._read(name, required=False)
            if optional is not None:
                tables[name] = optional
        factors = self._read("adjustment_factors", required=False)
        return CanonicalBatch(
            tables, factors,
            {"adapter": type(self).__name__, "source_path": str(self.path)},
        )

    def _read(self, name: str, required: bool = True):
        for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
            path = self.path / f"{name}{suffix}"
            if path.exists():
                return reader(path)
        directory = self.path / name
        if directory.exists():
            return pd.read_parquet(directory)
        if required:
            raise DataContractError(f"source adapter missing table: {name}")
        return None
