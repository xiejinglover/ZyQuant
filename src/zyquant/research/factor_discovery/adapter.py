from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class DiscoveryDataset:
    """Pool-local PIT observations and expression leaf values for one period."""

    observations: pd.DataFrame
    fields: dict[str, pd.DataFrame]
    label_metadata: pd.DataFrame
    exposures: pd.DataFrame | None = None


class DiscoveryAdapter(Protocol):
    def metadata(self) -> dict[str, Any]: ...

    def load_period(self, period: str, start: date, end: date) -> DiscoveryDataset: ...


def load_adapter(reference: str, parameters: dict[str, Any]) -> DiscoveryAdapter:
    module_name, attribute = reference.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    adapter = factory(**parameters) if callable(factory) else factory
    if not callable(getattr(adapter, "metadata", None)):
        raise TypeError("discovery adapter must implement metadata()")
    if not callable(getattr(adapter, "load_period", None)):
        raise TypeError("discovery adapter must implement load_period()")
    return adapter

