from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from zyquant.core.exceptions import DataContractError
from zyquant.core.plugins import PluginMetadata
from zyquant.data.adapters import CanonicalBatch, DirectoryDataAdapter
from zyquant.data.contracts import FINANCIAL_TABLES


class SQLDataAdapter:
    """Read canonical tables through SQLAlchemy without coupling the core."""

    def __init__(
        self,
        url: str,
        table_mapping: Mapping[str, str] | None = None,
        connect_args: Mapping[str, Any] | None = None,
    ):
        self.url = url
        self.table_mapping = dict(table_mapping or {})
        self.connect_args = dict(connect_args or {})

    def ingest(self, request: Mapping[str, Any] | None = None) -> CanonicalBatch:
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise DataContractError(
                "SQL connector requires: pip install 'zyquant[sql]'"
            ) from exc
        request = dict(request or {})
        queries = request.get("queries", {})
        engine = create_engine(self.url, connect_args=self.connect_args)
        tables: dict[str, pd.DataFrame] = {}
        try:
            for canonical in DirectoryDataAdapter.SOURCE_TABLES:
                query = queries.get(canonical)
                if query:
                    tables[canonical] = pd.read_sql_query(query, engine)
                else:
                    source = self.table_mapping.get(canonical, canonical)
                    tables[canonical] = pd.read_sql_table(source, engine)
            for canonical in FINANCIAL_TABLES:
                query = queries.get(canonical)
                source = self.table_mapping.get(canonical, canonical)
                if query:
                    tables[canonical] = pd.read_sql_query(query, engine)
                elif canonical in self.table_mapping:
                    tables[canonical] = pd.read_sql_table(source, engine)
            vendor_query = queries.get("adjustment_factors")
            vendor = pd.read_sql_query(vendor_query, engine) if vendor_query else None
        finally:
            engine.dispose()
        return CanonicalBatch(
            tables,
            vendor,
            {
                "adapter": type(self).__name__,
                "source": self.url.split("@")[-1],
                "tables": self.table_mapping,
            },
        )


def create_adapter(request: Mapping[str, Any] | None = None) -> SQLDataAdapter:
    payload = dict(request or {})
    url = str(payload.get("url", "")).strip()
    if not url:
        raise DataContractError("SQL connector request requires 'url'")
    return SQLDataAdapter(
        url,
        table_mapping=payload.get("table_mapping"),
        connect_args=payload.get("connect_args"),
    )


setattr(create_adapter, "plugin_metadata", PluginMetadata(
    name="sql",
    version="2.0.0",
    kind="data",
    input_schema="data.source/sql@1",
    output_schema="data.canonical_batch@1",
    optional_dependencies=("SQLAlchemy>=2.0",),
))
